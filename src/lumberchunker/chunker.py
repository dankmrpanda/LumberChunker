"""
Main LumberChunker class for semantic document segmentation.

This implementation preserves the exact algorithm from the original research paper:
"LumberChunker: Long-Form Narrative Document Segmentation"
"""

import logging
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
            output_path: If provided, saves chunks to this file path.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
        """
        # Split text into paragraphs first
        paragraphs = split_into_paragraphs(text)
        result = self.chunk_paragraphs(paragraphs, return_metadata=return_metadata)
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
            output_path: If provided, saves chunks to this file path.
            
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
        return_metadata: bool = False
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

        # Track chunk boundaries (list of paragraph indices where new chunks start)
        chunk_boundaries = []

        # Current window start position
        chunk_number = 0

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
                continue

            # 6b: NO SPLIT — content is cohesive, carry over last paragraph
            if "NO SPLIT" in gpt_output.upper():
                self._log("Answer: NO SPLIT (content is cohesive)")
                # Per spec: next prompt combines the last paragraph ID from this
                # window with new paragraphs. Start next window from last ID in
                # current window (re-include it for context).
                chunk_number = max(effective_end - 1, chunk_number + 1)

                # If we've reached the end (all remaining sent and cohesive), exit
                if chunk_number >= len(id_chunks) - 1:
                    break
                continue

            # 6c: Try to extract split ID from response
            extracted_id = extract_id_from_response(gpt_output)

            if extracted_id == -1:
                self._log("Could not parse ID, moving forward")
                chunk_number = fallback_next
                continue

            # 6d: Validate extracted_id is within sensible range
            # Must be after the window start (can't split at or before first paragraph)
            if extracted_id <= chunk_number:
                self._log(f"ID {extracted_id} is at or before window start {chunk_number}, moving forward")
                chunk_number = fallback_next
                continue

            # If ID is past the window but still within document, clamp to effective_end
            if extracted_id > effective_end and extracted_id < len(id_chunks):
                self._log(f"ID {extracted_id} past window end {effective_end}, clamping")
                extracted_id = effective_end

            # If ID is past the entire document, reject
            if extracted_id >= len(id_chunks):
                self._log(f"ID {extracted_id} past document end, moving forward")
                chunk_number = fallback_next
                continue

            self._log(f"Answer: ID {extracted_id}")

            # 6e: Record boundary and advance to the split point
            chunk_boundaries.append(extracted_id)
            chunk_number = extracted_id

            # Anti-stall: if last two boundaries are identical, force advance
            if len(chunk_boundaries) >= 2 and chunk_boundaries[-1] == chunk_boundaries[-2]:
                chunk_number = extracted_id + 1

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
            total_chunks = 0
            for ch in chapters_to_process:
                chapter_name = ch['chapter']
                chapter_text = ch['text']

                if not chapter_text.strip():
                    self._log(f"  Skipping empty chapter: {chapter_name}")
                    continue

                self._log(f"\n{'#'*60}")
                self._log(f"# PROCESSING CHAPTER: {chapter_name}")
                self._log(f"{'#'*60}")

                chunks = self.chunk(chapter_text)
                results.append({
                    'chapter': chapter_name,
                    'chunks': chunks,
                })
                total_chunks += len(chunks)
                self._log(f"\n  {chapter_name}: {len(chunks)} chunks created")

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

            return results
        finally:
            self._close_log()

    @staticmethod
    def _resolve_output_dir(epub_path: Path, output_dir: Union[str, Path]) -> Path:
        """Resolve and create the output directory.

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
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            counter = 1
            while True:
                candidate = parent_dir / f"{base_name}_{counter}"
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
