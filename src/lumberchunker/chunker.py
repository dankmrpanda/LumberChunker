"""
Main LumberChunker class for semantic document segmentation.

This implementation preserves the exact algorithm from the original research paper:
"LumberChunker: Long-Form Narrative Document Segmentation"
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from lumberchunker.providers.base import BaseLLMProvider, UsageStats
from lumberchunker.providers.gemini import GeminiProvider
from lumberchunker.providers.openai import OpenAIProvider
from lumberchunker.providers.anthropic import AnthropicProvider
from lumberchunker.providers.ollama import OllamaProvider
from lumberchunker.utils import (
    count_words,
    split_into_paragraphs,
    extract_id_from_response,
    read_file_or_epub,
)


# System prompt from the original implementation
SYSTEM_PROMPT = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: 
- If you find a content shift, return the ID of that paragraph: 'Answer: ID XXXX'
- If the content is cohesive with no clear shift, respond with: 'Answer: NO SPLIT'

Important: Only identify a split if there is a genuine content shift. Do NOT force a split if the paragraphs discuss the same topic or flow naturally together.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


class LumberChunker:
    """
    LLM-powered semantic document segmentation.
    
    LumberChunker dynamically segments documents into semantically independent chunks
    by iteratively prompting an LLM to identify points where content begins to shift.
    
    Example:
        >>> from lumberchunker import LumberChunker
        >>> 
        >>> # Using Gemini (default)
        >>> chunker = LumberChunker(api_key="your-gemini-api-key")
        >>> chunks = chunker.chunk("Your long document text here...")
        >>> 
        >>> # Using OpenAI
        >>> chunker = LumberChunker(provider="openai", api_key="your-openai-key")
        >>> chunks = chunker.chunk("Your long document text here...")
        >>> 
        >>> # Using Ollama (local)
        >>> chunker = LumberChunker(provider="ollama", model="llama3.3")
        >>> chunks = chunker.chunk("Your long document text here...")
    """
    
    SUPPORTED_PROVIDERS = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }
    
    def __init__(
        self,
        provider: Union[str, BaseLLMProvider] = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        target_chunk_tokens: int = 550,
        temperature: float = 0.1,
        timeout: float = 120.0,
        verbose: bool = True,
    ):
        """
        Initialize the LumberChunker.
        
        Args:
            provider: Either a string ("gemini", "openai", "anthropic", "ollama")
                     or a pre-configured BaseLLMProvider instance.
            api_key: API key for the provider (not needed for Ollama or pre-configured providers).
            model: Model name to use. If None, uses provider default.
            target_chunk_tokens: Target token count per chunk window (default: 550).
            temperature: LLM sampling temperature (default: 0.1).
            timeout: HTTP request timeout in seconds (default: 120).
            verbose: If True, print progress to stdout (default: True).
        """
        self.target_chunk_tokens = target_chunk_tokens
        self.verbose = verbose
        self._log_fh = None  # File handle for log file output

        # Configure library-wide logging level based on verbose flag.
        # All lumberchunker submodules use logging.getLogger(__name__) so
        # setting the level on the root "lumberchunker" logger controls them all.
        lib_logger = logging.getLogger("lumberchunker")
        if verbose:
            lib_logger.setLevel(logging.DEBUG)
            # Add a stdout handler only if one hasn't been added yet
            if not lib_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                lib_logger.addHandler(handler)
        else:
            lib_logger.setLevel(logging.WARNING)

        # Initialize provider
        if isinstance(provider, BaseLLMProvider):
            self._provider = provider
        elif isinstance(provider, str):
            provider_name = provider.lower()
            if provider_name not in self.SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Supported: {list(self.SUPPORTED_PROVIDERS.keys())}"
                )
            
            provider_class = self.SUPPORTED_PROVIDERS[provider_name]
            
            # Build provider kwargs
            kwargs = {"temperature": temperature, "timeout": timeout}
            if model:
                kwargs["model"] = model
            if api_key:
                kwargs["api_key"] = api_key
            
            # Ollama doesn't use api_key
            if provider_name == "ollama" and "api_key" in kwargs:
                del kwargs["api_key"]
            
            self._provider = provider_class(**kwargs)
        else:
            raise TypeError(
                f"provider must be a string or BaseLLMProvider instance, got {type(provider)}"
            )
    
    @property
    def usage(self) -> UsageStats:
        """Return cumulative LLM usage statistics from the underlying provider.

        The returned :class:`UsageStats` object has attributes:
        ``prompt_count``, ``total_input_tokens``, ``total_output_tokens``,
        ``total_tokens``, ``total_duration_ms``.  Call ``.to_dict()`` for a
        plain dict representation.  Use ``.reset()`` to zero the counters.
        """
        return self._provider.usage

    def _log(self, message: str = ""):
        """Log a message to stdout (if verbose) and/or log file (if open).
        
        Args:
            message: The message to log.
        """
        if self.verbose:
            print(message)
        if self._log_fh:
            self._log_fh.write(message + "\n")
            self._log_fh.flush()

    def _open_log(self, log_path: Union[str, Path]):
        """Open a log file for writing.
        
        Args:
            log_path: Path to the log file.
        """
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(log_path, "w", encoding="utf-8")
        
        # Log model information at the top of the file
        self._log(f"Model: {self._provider.model}")

    def _close_log(self):
        """Close the log file if open."""
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    # ------------------------------------------------------------------
    # Checkpoint helpers (stop / resume support)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(
        checkpoint_path: Path,
        paragraphs_hash: str,
    ) -> Optional[Dict]:
        """Load a checkpoint file and validate it matches the current document.

        Args:
            checkpoint_path: Path to the ``.checkpoint.json`` file.
            paragraphs_hash: SHA-256 hex digest of the current paragraphs.

        Returns:
            A dict with ``chunk_boundaries`` and ``chunk_number`` if the
            checkpoint is valid for this document, or ``None`` otherwise.
        """
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if data.get("paragraphs_hash") != paragraphs_hash:
                return None
            return {
                "chunk_boundaries": data["chunk_boundaries"],
                "chunk_number": data["chunk_number"],
            }
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _save_checkpoint_if_needed(
        self,
        checkpoint_path: Optional[Path],
        paragraphs_hash: Optional[str],
        chunk_number: int,
        chunk_boundaries: List[int],
        id_chunks: List[str],
        return_metadata: bool,
        output_path: Optional[Path],
    ):
        """Persist checkpoint and incremental output after an LLM call.

        Does nothing when *checkpoint_path* is ``None`` (no output path was
        given by the caller).

        Args:
            checkpoint_path: Where to write the checkpoint JSON.
            paragraphs_hash: Document fingerprint stored in the checkpoint.
            chunk_number: Current loop position (next paragraph to process).
            chunk_boundaries: Boundaries discovered so far.
            id_chunks: Full list of ID-prefixed paragraphs.
            return_metadata: Whether the caller requested metadata dicts.
            output_path: Path for the incremental output file.
        """
        if checkpoint_path is None:
            return

        # 1. Write checkpoint JSON atomically (write to temp then rename)
        tmp = checkpoint_path.with_suffix(".tmp")
        data = {
            "paragraphs_hash": paragraphs_hash,
            "chunk_number": chunk_number,
            "chunk_boundaries": chunk_boundaries,
        }
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(checkpoint_path)

        # 2. Write incremental output (chunks discovered so far)
        if output_path is not None:
            self._save_incremental_output(
                id_chunks, chunk_boundaries, chunk_number,
                return_metadata, output_path,
            )

    @staticmethod
    def _epub_fingerprint(epub_path: Path) -> str:
        """Return a fingerprint for an EPUB file (path + mtime).

        Used to detect when the source file has changed between runs so
        a stale checkpoint is not accidentally reused.
        """
        mtime = os.path.getmtime(epub_path)
        raw = f"{epub_path.resolve()}:{mtime}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_epub_checkpoint(
        checkpoint_path: Path,
        epub_path: Path,
        chapters_to_process: List[Dict],
    ) -> Dict[str, Dict]:
        """Load an epub-level checkpoint and return completed chapters.

        Validates that the checkpoint matches the current EPUB file and
        the same chapter list.

        Args:
            checkpoint_path: Path to the ``epub_checkpoint.json`` file.
            epub_path: Path to the source EPUB file.
            chapters_to_process: The current list of chapter dicts.

        Returns:
            A dict mapping chapter names to their result dicts
            (``{'chapter': ..., 'chunks': ...}``).  Returns an empty
            dict if the checkpoint is invalid or missing.
        """
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

        # Validate fingerprint
        expected_fp = LumberChunker._epub_fingerprint(epub_path)
        if data.get("epub_fingerprint") != expected_fp:
            return {}

        # Validate chapter list
        current_names = [ch["chapter"] for ch in chapters_to_process]
        if data.get("chapter_names") != current_names:
            return {}

        completed = data.get("completed_chapters", [])
        return {entry["chapter"]: entry for entry in completed}

    @staticmethod
    def _save_epub_checkpoint(
        checkpoint_path: Path,
        epub_path: Path,
        chapters_to_process: List[Dict],
        results: List[Dict],
    ) -> None:
        """Persist epub-level checkpoint after a chapter completes.

        Args:
            checkpoint_path: Where to write the checkpoint JSON.
            epub_path: Path to the source EPUB file.
            chapters_to_process: Full ordered list of chapter dicts.
            results: Completed chapter results so far.
        """
        data = {
            "epub_fingerprint": LumberChunker._epub_fingerprint(epub_path),
            "chapter_names": [ch["chapter"] for ch in chapters_to_process],
            "completed_chapters": results,
        }
        tmp = checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(checkpoint_path)

    @staticmethod
    def _save_incremental_output(
        id_chunks: List[str],
        chunk_boundaries: List[int],
        chunk_number: int,
        return_metadata: bool,
        output_path: Path,
    ):
        """Write the chunks discovered so far to *output_path*.

        This includes all completed chunks (those before *chunk_number*) plus
        a "pending" section showing the not-yet-chunked tail of the document.

        Args:
            id_chunks: Full list of ID-prefixed paragraphs.
            chunk_boundaries: Boundaries discovered so far.
            chunk_number: Current loop position.
            return_metadata: Whether metadata format is wanted.
            output_path: Path to write the output file.
        """
        clean = [re.sub(r'^ID \d+:\s*', '', c) for c in id_chunks]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("LumberChunker Output (in progress)\n")
            f.write("=" * 50 + "\n\n")

            prev = 0
            chunk_idx = 1
            for boundary in chunk_boundaries:
                if boundary <= prev:
                    continue
                chunk_text = "\n".join(clean[prev:boundary])
                f.write(f"--- Chunk {chunk_idx} ---\n")
                f.write(chunk_text)
                f.write("\n\n")
                prev = boundary
                chunk_idx += 1

            # Show remaining paragraphs not yet assigned to a chunk
            if prev < len(clean):
                f.write(f"--- Pending (paragraphs {prev}–{len(clean) - 1}) ---\n")
                f.write("\n".join(clean[prev:]))
                f.write("\n")

    def _llm_prompt(self, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and get a response.
        
        All providers support the system_prompt parameter, so we always pass it.
        
        Args:
            user_prompt: The user prompt to send.
            
        Returns:
            The LLM response text.
        """
        return self._provider.generate(user_prompt, system_prompt=SYSTEM_PROMPT)
    
    def chunk(self, text: str, return_metadata: bool = False, output_path: Optional[Union[str, Path]] = None) -> Union[List[str], List[Dict]]:
        """
        Chunk a document into semantically coherent segments.
        
        Args:
            text: The document text to chunk.
            return_metadata: If True, returns dicts with chunk text and metadata.
            output_path: If provided, saves chunks to this file path
                         incrementally after each LLM call. Also enables
                         stop/resume: if a checkpoint file exists from a
                         previous interrupted run on the same text, chunking
                         resumes automatically.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
        """
        # Split text into paragraphs first
        paragraphs = split_into_paragraphs(text)
        result = self.chunk_paragraphs(paragraphs, return_metadata=return_metadata, output_path=output_path)
        if output_path:
            self._save_chunks_to_file(result, output_path)
        return result
    
    def chunk_file(
        self,
        file_path: Union[str, Path],
        return_metadata: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[List[str], List[Dict]]:
        """
        Chunk a document file into semantically coherent segments.
        
        Automatically detects and converts EPUB files to text.
        Supports: .epub, .txt, and other plain text formats.
        
        Args:
            file_path: Path to the document file (txt, epub, etc.).
            return_metadata: If True, returns dicts with chunk text and metadata.
            output_path: If provided, saves chunks to this file path
                         incrementally after each LLM call. Also enables
                         stop/resume: if a checkpoint file exists from a
                         previous interrupted run on the same text, chunking
                         resumes automatically.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
            
        Example:
            >>> chunker = LumberChunker(api_key="your-key")
            >>> 
            >>> # Chunk an EPUB file
            >>> chunks = chunker.chunk_file("book.epub")
            >>> 
            >>> # Chunk a text file with output saved
            >>> chunks = chunker.chunk_file("document.txt", output_path="chunks.txt")
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ImportError: If EPUB support libraries aren't installed (for .epub files).
        """
        text = read_file_or_epub(file_path)
        return self.chunk(text, return_metadata=return_metadata, output_path=output_path)
    
    def chunk_paragraphs(
        self,
        paragraphs: List[str],
        return_metadata: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[List[str], List[Dict]]:
        """
        Chunk a list of paragraphs into semantically coherent segments.

        This method implements the LumberChunker algorithm:
        1. Add IDs to each paragraph (per-call, starting from 0)
        2. Iteratively build windows of paragraphs up to target token count (~550)
        3. Ask LLM to identify where content shifts (model outputs only the split ID)
        4. If SPLIT: save chunk from last boundary to split ID, start next window at split ID
        5. If NO SPLIT: carry over the last paragraph ID into the next window
        6. Always send the last window to the LLM, even if under target token count

        Edge cases handled:
        - Single paragraph: returned as-is without LLM call
        - Last paragraphs under target: still fed to LLM
        - Multiple consecutive NO SPLITs: paragraphs accumulate into one chunk
        - Model returns unparseable response: advance window, paragraphs stay in current chunk
        - Content safety flag: advance window
        - Very long single paragraph (>target words): sent alone, model can't split within

        Args:
            paragraphs: List of paragraph strings.
            return_metadata: If True, returns dicts with chunk text and metadata.
            output_path: If provided, saves progress incrementally to this file
                         and enables stop/resume via a sibling checkpoint file.

        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
        """
        if not paragraphs:
            return []

        # Single paragraph — no split possible
        if len(paragraphs) == 1:
            if return_metadata:
                return [{"text": paragraphs[0], "start_paragraph": 0,
                         "end_paragraph": 1, "paragraph_count": 1}]
            return [paragraphs[0]]

        # Add IDs to paragraphs
        id_chunks = [f"ID {i}: {para}" for i, para in enumerate(paragraphs)]

        # --- Checkpoint / resume setup ---
        checkpoint_path = None
        paragraphs_hash = None
        if output_path is not None:
            output_path = Path(output_path)
            checkpoint_path = output_path.parent / (output_path.stem + ".checkpoint.json")
            paragraphs_hash = hashlib.sha256(
                "\n".join(paragraphs).encode("utf-8")
            ).hexdigest()

        # Track chunk boundaries (list of paragraph indices where new chunks start)
        chunk_boundaries = []

        # Current window start position
        chunk_number = 0

        # Attempt to resume from checkpoint
        if checkpoint_path and checkpoint_path.exists():
            resumed = self._load_checkpoint(checkpoint_path, paragraphs_hash)
            if resumed is not None:
                chunk_boundaries = resumed["chunk_boundaries"]
                chunk_number = resumed["chunk_number"]
                self._log(
                    f"[RESUME] Loaded checkpoint – resuming from paragraph {chunk_number} "
                    f"with {len(chunk_boundaries)} boundaries already found"
                )
            else:
                self._log("[RESUME] Checkpoint found but does not match current document, starting fresh")

        # Main chunking loop — need at least 2 paragraphs remaining for a split decision
        while chunk_number < len(id_chunks) - 1:

            # --- Step 1: Expand window until ~target_chunk_tokens words ---
            word_count = 0
            i = 0
            while word_count < self.target_chunk_tokens and (i + chunk_number) < len(id_chunks):
                i += 1
                window_text = "\n".join(id_chunks[chunk_number : chunk_number + i])
                word_count = count_words(window_text)

            # Safety: ensure at least 1 paragraph
            if i == 0:
                i = 1

            # --- Step 2: Back off by 1 ONLY if we actually exceeded the target ---
            # If word_count < target, we ran out of paragraphs — don't back off.
            if i == 1:
                # Only 1 paragraph in window, can't back off
                effective_end = chunk_number + 1
            elif word_count >= self.target_chunk_tokens:
                # Exceeded target, back off by 1 to keep window under target
                effective_end = chunk_number + i - 1
                # But never back off below 2 paragraphs — LLM needs at least 2
                # to identify where content shifts
                if effective_end - chunk_number < 2:
                    effective_end = chunk_number + 2
            else:
                # Didn't exceed target (ran out of paragraphs), use all we have
                effective_end = chunk_number + i

            # Clamp to array bounds
            effective_end = min(effective_end, len(id_chunks))

            # --- Step 3: If near the end, include ALL remaining paragraphs ---
            # User requirement: "if the last IDs don't meet 550, still feed to model"
            if effective_end >= len(id_chunks) or chunk_number + i >= len(id_chunks):
                effective_end = len(id_chunks)

            final_document = "\n".join(id_chunks[chunk_number:effective_end])

            # Need at least 2 paragraphs to ask for a split — if only 1 remains, exit
            if effective_end - chunk_number < 2:
                break

            # --- Step 4: Compute fallback advance position BEFORE LLM call ---
            fallback_next = max(chunk_number + 1, effective_end)

            # Log what we're sending to the model
            last_id_in_window = effective_end - 1
            num_paras = effective_end - chunk_number
            self._log(f"\n{'='*60}")
            self._log(f"[LLM INPUT] Sending IDs {chunk_number} to {last_id_in_window} ({num_paras} paragraphs)")
            self._log(f"{'='*60}")
            self._log(final_document)
            self._log(f"{'='*60}")

            # --- Step 5: Send to LLM ---
            prompt = f"\nDocument:\n{final_document}"
            gpt_output = self._llm_prompt(prompt)

            # Log token usage for this call
            u = self._provider.usage
            self._log(f"[TOKENS] in={u.last_input_tokens}  out={u.last_output_tokens}  "
                      f"(cumulative: {u.prompt_count} calls, {u.total_input_tokens} in, "
                      f"{u.total_output_tokens} out, {u.total_tokens} total)")

            # Log what the model returned
            self._log(f"\n[LLM OUTPUT] Raw response: {gpt_output}")

            # --- Step 6: Handle response ---

            # 6a: Content safety flag (legacy support)
            if gpt_output == "content_flag_increment":
                self._log("Content flag triggered, skipping forward")
                chunk_number = fallback_next
                self._save_checkpoint_if_needed(
                    checkpoint_path, paragraphs_hash, chunk_number,
                    chunk_boundaries, id_chunks, return_metadata, output_path,
                )
                continue

            # 6b: NO SPLIT — content is cohesive, carry over last paragraph
            if "NO SPLIT" in gpt_output.upper():
                self._log("Answer: NO SPLIT (content is cohesive)")
                # Per spec: next prompt combines the last paragraph ID from this
                # window with new paragraphs. Start next window from last ID in
                # current window (re-include it for context).
                chunk_number = max(effective_end - 1, chunk_number + 1)

                self._save_checkpoint_if_needed(
                    checkpoint_path, paragraphs_hash, chunk_number,
                    chunk_boundaries, id_chunks, return_metadata, output_path,
                )

                # If we've reached the end (all remaining sent and cohesive), exit
                if chunk_number >= len(id_chunks) - 1:
                    break
                continue

            # 6c: Try to extract split ID from response
            extracted_id = extract_id_from_response(gpt_output)

            if extracted_id == -1:
                self._log("Could not parse ID, moving forward")
                chunk_number = fallback_next
                self._save_checkpoint_if_needed(
                    checkpoint_path, paragraphs_hash, chunk_number,
                    chunk_boundaries, id_chunks, return_metadata, output_path,
                )
                continue

            # 6d: Validate extracted_id is within sensible range
            # Must be after the window start (can't split at or before first paragraph)
            if extracted_id <= chunk_number:
                self._log(f"ID {extracted_id} is at or before window start {chunk_number}, moving forward")
                chunk_number = fallback_next
                self._save_checkpoint_if_needed(
                    checkpoint_path, paragraphs_hash, chunk_number,
                    chunk_boundaries, id_chunks, return_metadata, output_path,
                )
                continue

            # If ID is past the window but still within document, clamp to effective_end
            if extracted_id > effective_end and extracted_id < len(id_chunks):
                self._log(f"ID {extracted_id} past window end {effective_end}, clamping")
                extracted_id = effective_end

            # If ID is past the entire document, reject
            if extracted_id >= len(id_chunks):
                self._log(f"ID {extracted_id} past document end, moving forward")
                chunk_number = fallback_next
                self._save_checkpoint_if_needed(
                    checkpoint_path, paragraphs_hash, chunk_number,
                    chunk_boundaries, id_chunks, return_metadata, output_path,
                )
                continue

            self._log(f"Answer: ID {extracted_id}")

            # 6e: Record boundary and advance to the split point
            chunk_boundaries.append(extracted_id)
            chunk_number = extracted_id

            # Anti-stall: if last two boundaries are identical, force advance
            if len(chunk_boundaries) >= 2 and chunk_boundaries[-1] == chunk_boundaries[-2]:
                chunk_number = extracted_id + 1

            self._save_checkpoint_if_needed(
                checkpoint_path, paragraphs_hash, chunk_number,
                chunk_boundaries, id_chunks, return_metadata, output_path,
            )

        # Always add final boundary to capture remaining paragraphs
        chunk_boundaries.append(len(id_chunks))

        # --- Assembly: merge paragraphs between boundaries into final chunks ---
        clean_paragraphs = [
            re.sub(r'^ID \d+:\s*', '', chunk) for chunk in id_chunks
        ]

        final_chunks = []
        prev_boundary = 0
        for boundary in chunk_boundaries:
            if boundary <= prev_boundary:
                continue  # Skip invalid/duplicate boundaries

            chunk_text = '\n'.join(clean_paragraphs[prev_boundary:boundary])

            if return_metadata:
                final_chunks.append({
                    "text": chunk_text,
                    "start_paragraph": prev_boundary,
                    "end_paragraph": boundary,
                    "paragraph_count": boundary - prev_boundary,
                })
            else:
                final_chunks.append(chunk_text)

            prev_boundary = boundary

        # Log usage summary
        u = self._provider.usage
        self._log(f"\n{'='*60}")
        self._log(f"Chunking complete: {len(final_chunks)} chunks from {len(paragraphs)} paragraphs")
        self._log(f"LLM usage: {u.prompt_count} prompts, "
                  f"{u.total_input_tokens:,} input tokens, "
                  f"{u.total_output_tokens:,} output tokens, "
                  f"{u.total_tokens:,} total tokens")
        if u.total_duration_ms:
            self._log(f"Total LLM time: {u.total_duration_ms / 1000:.1f}s")
        self._log(f"{'='*60}")

        # Remove checkpoint file on successful completion
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            self._log("[CHECKPOINT] Removed checkpoint file (chunking complete)")

        return final_chunks

    def list_chapters(
        self,
        epub_path: Union[str, Path],
        use_llm_extraction: bool = False,
        extraction_provider: Optional[Union[str, BaseLLMProvider]] = None,
        extraction_api_key: Optional[str] = None,
        extraction_model: Optional[str] = None,
    ) -> List[Dict]:
        """
        List chapters in an EPUB file without chunking them.

        Useful for inspecting which chapters are available before calling
        :meth:`chunk_epub` with a ``chapters`` filter.

        Args:
            epub_path: Path to the EPUB file.
            use_llm_extraction: If True, uses LLM for chapter boundary detection.
            extraction_provider: Provider for extraction (if use_llm_extraction).
            extraction_api_key: API key for extraction provider.
            extraction_model: Model for extraction provider.

        Returns:
            List of dicts, each with:
            - 'index': 1-based chapter number (int)
            - 'chapter': Chapter title (str)
            - 'word_count': Number of words in the chapter (int)

        Example:
            >>> chunker = LumberChunker(api_key="...")
            >>> for ch in chunker.list_chapters("book.epub"):
            ...     print(f"{ch['index']}. {ch['chapter']} ({ch['word_count']:,} words)")
        """
        from lumberchunker.utils import epub_to_chapters_simple, epub_to_chapters

        epub_path = Path(epub_path)
        if not epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")

        if use_llm_extraction:
            chapters = epub_to_chapters(
                epub_path,
                provider=extraction_provider or self._provider,
                api_key=extraction_api_key,
                model=extraction_model,
            )
        else:
            chapters = epub_to_chapters_simple(epub_path)

        return [
            {
                "index": i + 1,
                "chapter": ch["chapter"],
                "word_count": len(ch["text"].split()),
            }
            for i, ch in enumerate(chapters)
        ]

    def chunk_epub(
        self,
        epub_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        chapters: Optional[Union[List[int], List[str]]] = None,
        use_llm_extraction: bool = False,
        extraction_provider: Optional[Union[str, BaseLLMProvider]] = None,
        extraction_api_key: Optional[str] = None,
        extraction_model: Optional[str] = None,
        trim_front_back_matter: bool = False,
    ) -> List[Dict]:
        """
        Chunk an EPUB file by chapters, producing semantically coherent segments.

        This is the main entry point for EPUB processing:
        1. Extracts chapters from the EPUB (using TOC structure by default)
        2. Optionally filters to specific chapters
        3. Runs LumberChunker on each chapter to find semantic boundaries
        4. Optionally saves all outputs to a directory

        Args:
            epub_path: Path to the EPUB file.
            output_dir: Directory to save outputs. Options:
                - None: No files saved (default).
                - "auto": Creates a folder named after the EPUB file
                  (e.g., ``my_book/``). If the folder already exists,
                  appends an incrementing number (``my_book_1/``, etc.).
                - Any path string/Path: Uses that directory (created if needed).
                When provided, saves three files:
                - ``chapters_with_ids.txt`` — extracted chapters with paragraph IDs
                - ``chunks.txt`` — final chunking output
                - ``llm_log.txt`` — full LLM input/output log
            chapters: Which chapters to chunk. Options:
                - None: Chunk **all** chapters (default).
                - List of ints: 1-based chapter indices
                  (e.g., ``[1, 3, 5]`` for the 1st, 3rd, and 5th chapters).
                - List of strings: Chapter titles to match. A chapter is included
                  if any provided string appears as a substring (case-insensitive).
                Use :meth:`list_chapters` to see available chapters and indices.
            use_llm_extraction: If True, uses LLM for chapter boundary detection
                               and opening sentence cleaning (slower but more accurate
                               for front/back matter removal).
            extraction_provider: Provider for extraction (if use_llm_extraction=True).
                                Defaults to the chunker's own provider.
            extraction_api_key: API key for extraction provider.
            extraction_model: Model for extraction provider.
            trim_front_back_matter: If True, runs a post-processing step after
                chunking that feeds the start and end chapters to the LLM to
                classify them as narrative vs. auxiliary front/back matter
                (e.g., title page, copyright, acknowledgements, about the
                author). Non-narrative chapters are trimmed from both ends.
                Default: False.

        Returns:
            List of dicts, each with:
            - 'chapter': Chapter title
            - 'chunks': List of chunk strings

        Example:
            >>> chunker = LumberChunker(provider="openai", api_key="...")
            >>>
            >>> # Chunk all chapters
            >>> results = chunker.chunk_epub("book.epub", output_dir="auto")
            >>>
            >>> # Chunk only chapters 1, 3, and 5
            >>> results = chunker.chunk_epub("book.epub", chapters=[1, 3, 5])
            >>>
            >>> # Chunk chapters by title
            >>> results = chunker.chunk_epub("book.epub", chapters=["Prologue", "Chapter 1"])
            >>>
            >>> for r in results:
            ...     print(f"{r['chapter']}: {len(r['chunks'])} chunks")
        """
        from lumberchunker.utils import epub_to_chapters_simple, epub_to_chapters

        epub_path = Path(epub_path)
        if not epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")

        chapters_filter = chapters  # stash before local reuse

        # Set up output directory if requested
        out_dir = None
        if output_dir is not None:
            out_dir = self._resolve_output_dir(epub_path, output_dir)
            self._log(f"Output folder: {out_dir}")
            self._open_log(out_dir / "llm_log.txt")

        try:
            # Step 1: Extract chapters
            if use_llm_extraction:
                all_chapters = epub_to_chapters(
                    epub_path,
                    provider=extraction_provider or self._provider,
                    api_key=extraction_api_key,
                    model=extraction_model,
                )
            else:
                all_chapters = epub_to_chapters_simple(epub_path)

            self._log(f"\nExtracted {len(all_chapters)} chapters from EPUB")
            for i, ch in enumerate(all_chapters):
                word_count = len(ch['text'].split())
                self._log(f"  {i+1}. {ch['chapter']} ({word_count:,} words)")

            # Filter chapters if requested
            if chapters_filter is not None:
                selected = self._filter_chapters(all_chapters, chapters_filter)
                if not selected:
                    available = [f"  {i+1}. {ch['chapter']}" for i, ch in enumerate(all_chapters)]
                    raise ValueError(
                        f"No chapters matched the filter {chapters_filter!r}.\n"
                        f"Available chapters:\n" + "\n".join(available)
                    )
                self._log(f"\nFiltered to {len(selected)}/{len(all_chapters)} chapters")
                chapters_to_process = selected
            else:
                chapters_to_process = all_chapters

            # Save chapters with IDs (save all extracted, not just filtered)
            if out_dir:
                self._save_chapters_with_ids(chapters_to_process, out_dir / "chapters_with_ids.txt")

            # Step 2: Chunk each chapter
            results = []

            # Load epub checkpoint if available (skip already-completed chapters)
            epub_checkpoint_path = (out_dir / "epub_checkpoint.json") if out_dir else None
            completed_map = {}  # chapter_name -> result dict
            if epub_checkpoint_path and epub_checkpoint_path.exists():
                completed_map = self._load_epub_checkpoint(
                    epub_checkpoint_path, epub_path, chapters_to_process,
                )

            total_chunks = 0
            for ch in chapters_to_process:
                chapter_name = ch['chapter']
                chapter_text = ch['text']

                # Skip if already completed in a previous run
                if chapter_name in completed_map:
                    self._log(f"\n[RESUME] Skipping already-completed chapter: {chapter_name}")
                    results.append(completed_map[chapter_name])
                    total_chunks += len(completed_map[chapter_name]['chunks'])
                    continue

                if not chapter_text.strip():
                    self._log(f"  Skipping empty chapter: {chapter_name}")
                    continue

                self._log(f"\n{'#'*60}")
                self._log(f"# PROCESSING CHAPTER: {chapter_name}")
                self._log(f"{'#'*60}")

                chunks = self.chunk(chapter_text)
                result = {
                    'chapter': chapter_name,
                    'chunks': chunks,
                }
                results.append(result)
                total_chunks += len(chunks)
                self._log(f"\n  {chapter_name}: {len(chunks)} chunks created")

                # Save epub checkpoint after each chapter completes
                if epub_checkpoint_path:
                    self._save_epub_checkpoint(
                        epub_checkpoint_path, epub_path,
                        chapters_to_process, results,
                    )

            # Step 3: Post-processing — trim front/back matter if requested
            if trim_front_back_matter and len(results) > 2:
                before_count = len(results)
                results = self._trim_boundary_chapters(results)
                trimmed = before_count - len(results)
                if trimmed:
                    total_chunks = sum(len(r['chunks']) for r in results)
                    self._log(f"\n[TRIM] Removed {trimmed} non-narrative chapter(s), "
                              f"{len(results)} chapters remain")
                else:
                    self._log("\n[TRIM] All chapters appear to be narrative, nothing trimmed")

            self._log(f"\nTotal: {total_chunks} chunks from {len(results)} chapters")

            # Log usage summary for entire EPUB
            u = self._provider.usage
            self._log(f"\n{'='*60}")
            self._log(f"EPUB chunking complete")
            self._log(f"LLM usage: {u.prompt_count} prompts, "
                      f"{u.total_input_tokens:,} input tokens, "
                      f"{u.total_output_tokens:,} output tokens, "
                      f"{u.total_tokens:,} total tokens")
            if u.total_duration_ms:
                self._log(f"Total LLM time: {u.total_duration_ms / 1000:.1f}s")
            self._log(f"{'='*60}")

            # Save final chunks
            if out_dir:
                self._save_epub_output(results, out_dir / "chunks.txt")
                self._log(f"\nAll outputs saved to: {out_dir}")

            # Remove epub checkpoint on successful completion
            if epub_checkpoint_path and epub_checkpoint_path.exists():
                epub_checkpoint_path.unlink()
                self._log("[CHECKPOINT] Removed epub checkpoint (all chapters complete)")

            return results
        finally:
            self._close_log()

    # Prompt for post-processing boundary trimming
    _BOUNDARY_TRIM_PROMPT = (
        "You are classifying whether a chapter from a book is part of the "
        "main narrative or is auxiliary front/back matter.\n"
        "\n"
        "Front matter examples: title page, copyright, dedication, table of "
        "contents, epigraph, author's note (if short), legal notice.\n"
        "Back matter examples: acknowledgements, about the author, also by "
        "this author, discussion questions, preview of another book, index.\n"
        "Narrative examples: prologue, chapter 1, epilogue, any chapter that "
        "contains story prose, dialogue, or plot content.\n"
        "\n"
        "Given the chapter title and opening text below, respond with ONLY "
        "valid JSON on a single line:\n"
        '{"is_narrative": true, "reason": "brief explanation"}\n'
        "or\n"
        '{"is_narrative": false, "reason": "brief explanation"}\n'
    )

    def _trim_boundary_chapters(self, results: List[Dict]) -> List[Dict]:
        """Trim non-narrative chapters from the start and end of results.

        Feeds each chapter's title and first ~200 words to the LLM to
        classify it as narrative vs. front/back matter.  Stops trimming
        as soon as a narrative chapter is found (does not remove chapters
        from the middle of the book).

        Args:
            results: List of ``{'chapter': str, 'chunks': [...]}`` dicts.

        Returns:
            The same list with leading/trailing non-narrative chapters removed.
        """
        if len(results) <= 2:
            return results

        def _first_n_words(chunks: List, n: int = 200) -> str:
            """Extract the first *n* words from a chapter's chunks."""
            text = ""
            for c in chunks:
                text += (c["text"] if isinstance(c, dict) else c) + "\n"
                if len(text.split()) >= n:
                    break
            words = text.split()[:n]
            return " ".join(words)

        def _classify(chapter_name: str, opening: str) -> bool:
            """Return True if the chapter looks like narrative content."""
            user_msg = f"Chapter title: {chapter_name}\n\nOpening text:\n{opening}"
            try:
                raw = self._provider.generate(
                    user_msg, system_prompt=self._BOUNDARY_TRIM_PROMPT,
                )
                # Parse JSON from the response (tolerant of extra text)
                raw = raw.strip()
                # Try to find a JSON object in the response
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    parsed = json.loads(raw[start:end])
                    is_narr = parsed.get("is_narrative", True)
                    reason = parsed.get("reason", "")
                    self._log(f"  [TRIM] {chapter_name}: "
                              f"{'NARRATIVE' if is_narr else 'NON-NARRATIVE'} "
                              f"— {reason}")
                    return bool(is_narr)
            except Exception as e:
                self._log(f"  [TRIM] {chapter_name}: classification error ({e}), keeping")
            # Default to keeping the chapter on any failure
            return True

        self._log(f"\n{'='*60}")
        self._log("[TRIM] Post-processing: checking start chapters for front matter")
        self._log(f"{'='*60}")

        # --- Trim from the front ---
        max_check = min(5, len(results) - 1)  # never trim ALL chapters
        front_trim = 0
        for i in range(max_check):
            ch = results[i]
            opening = _first_n_words(ch["chunks"])
            if _classify(ch["chapter"], opening):
                break  # found narrative, stop
            front_trim = i + 1

        self._log(f"\n{'='*60}")
        self._log("[TRIM] Post-processing: checking end chapters for back matter")
        self._log(f"{'='*60}")

        # --- Trim from the back ---
        back_trim = 0
        for i in range(max_check):
            idx = len(results) - 1 - i
            if idx <= front_trim:
                break  # don't overlap with front trim
            ch = results[idx]
            opening = _first_n_words(ch["chunks"])
            if _classify(ch["chapter"], opening):
                break  # found narrative, stop
            back_trim = i + 1

        # Apply trims
        end_idx = len(results) - back_trim if back_trim else len(results)
        return results[front_trim:end_idx]

    @staticmethod
    def _resolve_output_dir(epub_path: Path, output_dir: Union[str, Path]) -> Path:
        """Resolve and create the output directory.

        When *output_dir* is ``"auto"`` and a directory with the EPUB's base
        name already exists **and** contains an ``epub_checkpoint.json``, that
        directory is reused so that a resumed run continues into the same
        folder instead of creating ``_1``, ``_2``, etc.

        Args:
            epub_path: Path to the source EPUB file.
            output_dir: ``"auto"`` to create a folder named after the EPUB,
                        or an explicit directory path.

        Returns:
            Path to the created output directory.
        """
        if isinstance(output_dir, str) and output_dir.lower() == "auto":
            base_name = epub_path.stem
            parent_dir = epub_path.parent
            candidate = parent_dir / base_name

            # Reuse existing directory if it contains an epub checkpoint
            if candidate.exists() and (candidate / "epub_checkpoint.json").exists():
                return candidate

            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            counter = 1
            while True:
                candidate = parent_dir / f"{base_name}_{counter}"
                # Also check numbered dirs for a checkpoint to resume
                if candidate.exists() and (candidate / "epub_checkpoint.json").exists():
                    return candidate
                if not candidate.exists():
                    candidate.mkdir(parents=True, exist_ok=True)
                    return candidate
                counter += 1
        else:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            return out

    @staticmethod
    def _filter_chapters(
        all_chapters: List[Dict],
        selector: Union[List[int], List[str]],
    ) -> List[Dict]:
        """Return the subset of *all_chapters* that match *selector*.

        Args:
            all_chapters: Full list of chapters (each a dict with ``'chapter'`` key).
            selector: Either a list of 1-based indices or a list of title
                      substrings (case-insensitive).

        Returns:
            Filtered list of chapter dicts preserving original order.
        """
        if not selector:
            return all_chapters

        # Determine type from first element
        if isinstance(selector[0], int):
            index_set = set(selector)
            return [
                ch for i, ch in enumerate(all_chapters)
                if (i + 1) in index_set
            ]
        else:
            # String matching — include chapter if any selector is a
            # case-insensitive substring of its title
            needles = [s.lower() for s in selector]
            return [
                ch for ch in all_chapters
                if any(n in ch["chapter"].lower() for n in needles)
            ]

    @staticmethod
    def _save_chapters_with_ids(chapters: List[Dict], output_path: Path):
        """Save chapters with paragraph ID markers to a text file.

        Uses ``text_with_ids`` if available (from LLM extraction), otherwise
        generates IDs from the plain text paragraphs.

        Args:
            chapters: List of chapter dicts with 'chapter' and 'text' keys.
            output_path: Path to save the output file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("EPUB Chapters with Paragraph IDs\n")
            f.write("=" * 50 + "\n\n")

            for i, ch in enumerate(chapters):
                chapter_name = ch['chapter']
                word_count = len(ch['text'].split())

                # Use pre-generated text_with_ids if available, otherwise generate
                if ch.get('text_with_ids'):
                    chapter_content = ch['text_with_ids']
                else:
                    paras = split_into_paragraphs(ch['text'])
                    chapter_content = "\n\n".join(
                        f"ID {idx}: {para}" for idx, para in enumerate(paras) if para.strip()
                    )

                f.write(f"=== Chapter {i}: {chapter_name} ===\n")
                f.write(f"Word Count: {word_count:,}\n")
                f.write("-" * 40 + "\n\n")
                f.write(chapter_content)
                f.write("\n\n" + "=" * 50 + "\n\n")

    @staticmethod
    def _save_chunks_to_file(chunks: List, output_path: Union[str, Path]):
        """Save chunks to a text file.

        Args:
            chunks: List of chunk strings or dicts with 'text' key.
            output_path: Path to save the file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("LumberChunker Output\n")
            f.write("=" * 50 + "\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                content = chunk['text'] if isinstance(chunk, dict) else chunk
                f.write(content)
                f.write("\n\n")

    def _save_epub_output(
        self,
        results: List[Dict],
        output_path: Union[str, Path],
    ):
        """Save EPUB chunking results to a text file.

        Output format::

            === Chapter Title ===

            --- Chunk 1 ---
            <chunk text>

            --- Chunk 2 ---
            <chunk text>

        Args:
            results: List of dicts with 'chapter' and 'chunks' keys.
            output_path: Path to save the file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("LumberChunker Output\n")
            f.write("=" * 50 + "\n\n")

            for chapter_result in results:
                chapter_name = chapter_result['chapter']
                chunks = chapter_result['chunks']

                f.write(f"=== {chapter_name} ===\n\n")
                for i, chunk in enumerate(chunks):
                    f.write(f"--- Chunk {i+1} ---\n")
                    content = chunk['text'] if isinstance(chunk, dict) else chunk
                    f.write(content)
                    f.write("\n\n")

        total_chunks = sum(len(r['chunks']) for r in results)
        self._log(f"Saved {total_chunks} chunks from {len(results)} chapters to: {output_path}")
