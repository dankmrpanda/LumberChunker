"""
Main LumberChunker class for semantic document segmentation.

This implementation preserves the exact algorithm from the original research paper:
"LumberChunker: Long-Form Narrative Document Segmentation"
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from lumberchunker.providers.base import BaseLLMProvider
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
        """
        self.target_chunk_tokens = target_chunk_tokens
        
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
            kwargs = {"temperature": temperature}
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
    
    def _llm_prompt(self, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and get a response.
        
        This method handles the system prompt and provider-specific logic.
        
        Args:
            user_prompt: The user prompt to send.
            
        Returns:
            The LLM response text.
        """
        # Check if provider supports system_prompt parameter
        if hasattr(self._provider.generate, '__code__'):
            params = self._provider.generate.__code__.co_varnames
            if 'system_prompt' in params:
                return self._provider.generate(user_prompt, system_prompt=SYSTEM_PROMPT)
        
        # For providers that don't support separate system prompt, combine them
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
        return self._provider.generate(full_prompt)
    
    def chunk(self, text: str, return_metadata: bool = False) -> Union[List[str], List[Dict]]:
        """
        Chunk a document into semantically coherent segments.
        
        Args:
            text: The document text to chunk.
            return_metadata: If True, returns dicts with chunk text and metadata.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
        """
        # Split text into paragraphs first
        paragraphs = split_into_paragraphs(text)
        return self.chunk_paragraphs(paragraphs, return_metadata=return_metadata)
    
    def chunk_file(
        self,
        file_path: Union[str, Path],
        return_metadata: bool = False
    ) -> Union[List[str], List[Dict]]:
        """
        Chunk a document file into semantically coherent segments.
        
        Automatically detects and converts EPUB files to text.
        Supports: .epub, .txt, and other plain text formats.
        
        Args:
            file_path: Path to the document file (txt, epub, etc.).
            return_metadata: If True, returns dicts with chunk text and metadata.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
            
        Example:
            >>> chunker = LumberChunker(api_key="your-key")
            >>> 
            >>> # Chunk an EPUB file
            >>> chunks = chunker.chunk_file("book.epub")
            >>> 
            >>> # Chunk a text file
            >>> chunks = chunker.chunk_file("document.txt")
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ImportError: If EPUB support libraries aren't installed (for .epub files).
        """
        text = read_file_or_epub(file_path)
        return self.chunk(text, return_metadata=return_metadata)
    
    def chunk_paragraphs(
        self,
        paragraphs: List[str],
        return_metadata: bool = False
    ) -> Union[List[str], List[Dict]]:
        """
        Chunk a list of paragraphs into semantically coherent segments.
        
        This method implements the LumberChunker algorithm:
        1. Add IDs to each paragraph
        2. Iteratively build groups of paragraphs up to target token count
        3. Ask LLM to identify where content shifts (model outputs only the ID)
        4. Concatenate paragraphs from start to split ID into a chunk
        5. If "NO SPLIT": carry over the last ID and add more paragraphs
        6. Always ensure at least 2 IDs are sent to the model
        
        Args:
            paragraphs: List of paragraph strings.
            return_metadata: If True, returns dicts with chunk text and metadata.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
        """
        if not paragraphs:
            return []
        
        # Add IDs to paragraphs
        id_chunks = [f"ID {i}: {para}" for i, para in enumerate(paragraphs)]
        
        # Track chunk boundaries (list of end IDs for each chunk)
        chunk_boundaries = []
        
        # Current window start position
        window_start = 0
        
        # Main chunking loop
        while window_start < len(id_chunks):
            # Calculate window end: expand until we reach ~550 words or end of document
            word_count = 0
            window_end = window_start
            
            while word_count < self.target_chunk_tokens and window_end < len(id_chunks):
                window_end += 1
                window_text = "\n".join(id_chunks[window_start:window_end])
                word_count = count_words(window_text)
            
            # Ensure we have at least 2 IDs in the window (requirement)
            if window_end - window_start < 2:
                window_end = min(window_start + 2, len(id_chunks))
            
            # If we've reached the end of the document, this is the last chunk
            if window_end >= len(id_chunks):
                # Add remaining paragraphs as the final chunk
                chunk_boundaries.append(len(id_chunks))
                break
            
            # Build the document for LLM analysis
            # Back off by 1 to not overshoot the target token count
            if window_end - window_start > 2:
                window_end -= 1
            
            final_document = "\n".join(id_chunks[window_start:window_end])
            
            # Log what we're sending to the model
            print(f"\n{'='*60}")
            print(f"[LLM INPUT] Sending IDs {window_start} to {window_end - 1} ({window_end - window_start} paragraphs)")
            print(f"{'='*60}")
            print(final_document[:500] + "..." if len(final_document) > 500 else final_document)
            print(f"{'='*60}")
            
            # Build prompt and get LLM response
            prompt = f"\nDocument:\n{final_document}"
            gpt_output = self._llm_prompt(prompt)
            
            # Log what the model returned
            print(f"\n[LLM OUTPUT] Raw response: {gpt_output}")
            
            # Handle "NO SPLIT" response
            if "NO SPLIT" in gpt_output.upper():
                print("Answer: NO SPLIT (content is cohesive)")
                # Per spec: next request should include the last ID from this window
                # plus at least 1 more paragraph (or until ~550 words)
                # Set window_start to last ID in current window (window_end - 1)
                # This means we'll include window_end-1 again in next iteration
                window_start = window_end - 1
                
                # Ensure we can still make progress
                if window_start >= len(id_chunks) - 1:
                    # We're at the end, add everything as final chunk
                    chunk_boundaries.append(len(id_chunks))
                    break
                continue
            
            # Handle content flag (legacy support)
            if gpt_output == "content_flag_increment":
                window_start = window_end - 1
                continue
            
            # Extract ID from response
            extracted_id = extract_id_from_response(gpt_output)
            
            if extracted_id == -1:
                print("Could not parse ID, moving forward")
                # Move forward to avoid infinite loop
                window_start = window_end - 1
                continue
            
            # Validate extracted_id is within current window
            if extracted_id < window_start or extracted_id >= window_end:
                print(f"ID {extracted_id} outside window [{window_start}, {window_end}), adjusting")
                # Clamp to valid range
                extracted_id = max(window_start + 1, min(extracted_id, window_end - 1))
            
            print(f"Answer: ID {extracted_id}")
            
            # Record the boundary: chunk goes from previous boundary (or 0) to extracted_id
            chunk_boundaries.append(extracted_id)
            
            # Next window starts at the split point (extracted_id)
            window_start = extracted_id
            
            # Prevent infinite loop: if we're stuck at the same position
            if chunk_boundaries and len(chunk_boundaries) >= 2:
                if chunk_boundaries[-1] == chunk_boundaries[-2]:
                    window_start = extracted_id + 1
        
        # Handle edge case: if no boundaries were found, treat entire document as one chunk
        if not chunk_boundaries:
            chunk_boundaries = [len(id_chunks)]
        
        # Remove IDs from paragraphs for final output
        clean_paragraphs = [
            re.sub(r'^ID \d+:\s*', '', chunk) for chunk in id_chunks
        ]
        
        # Create final chunks by merging paragraphs between boundaries
        final_chunks = []
        prev_boundary = 0
        for boundary in chunk_boundaries:
            if boundary <= prev_boundary:
                continue  # Skip invalid boundaries
            
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
        
        return final_chunks
