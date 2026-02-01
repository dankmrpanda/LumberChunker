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

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

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
        
        This method implements the original LumberChunker algorithm exactly:
        1. Add IDs to each paragraph
        2. Iteratively build groups of paragraphs up to target token count
        3. Ask LLM to identify where content shifts
        4. Merge paragraphs between shift points into final chunks
        
        Args:
            paragraphs: List of paragraph strings.
            return_metadata: If True, returns dicts with chunk text and metadata.
            
        Returns:
            List of chunk strings, or list of dicts if return_metadata is True.
        """
        if not paragraphs:
            return []
        
        # Add IDs to paragraphs (matching original implementation)
        id_chunks = [f"ID {i}: {para}" for i, para in enumerate(paragraphs)]
        
        chunk_number = 0
        new_id_list = []
        
        # Main chunking loop (preserves original algorithm exactly)
        while chunk_number < len(id_chunks) - 5:
            word_count = 0
            i = 0
            
            # Build up paragraphs until we reach target token count
            while word_count < self.target_chunk_tokens and i + chunk_number < len(id_chunks) - 1:
                i += 1
                final_document = "\n".join(
                    id_chunks[k] for k in range(chunk_number, i + chunk_number)
                )
                word_count = count_words(final_document)
            
            # Adjust the document (from original implementation)
            if i == 1:
                final_document = "\n".join(
                    id_chunks[k] for k in range(chunk_number, i + chunk_number)
                )
            else:
                final_document = "\n".join(
                    id_chunks[k] for k in range(chunk_number, i - 1 + chunk_number)
                )
            
            chunk_number = chunk_number + i - 1
            
            # Build prompt and get LLM response
            question = f"\nDocument:\n{final_document}"
            prompt = question  # System prompt is handled in _llm_prompt
            
            gpt_output = self._llm_prompt(prompt)
            
            # Handle content flag (from original implementation)
            if gpt_output == "content_flag_increment":
                chunk_number = chunk_number + 1
            else:
                # Extract ID from response
                extracted_id = extract_id_from_response(gpt_output)
                
                if extracted_id == -1:
                    print("repeat this one")  # Matching original debug output
                else:
                    print(f"Answer: ID {extracted_id}")  # Matching original debug output
                    chunk_number = extracted_id
                    new_id_list.append(chunk_number)
                    
                    # Increment to avoid infinite loop (from original implementation)
                    if new_id_list[-1] == chunk_number:
                        chunk_number = chunk_number + 1
        
        # Add the last chunk to the list
        new_id_list.append(len(id_chunks))
        
        # Remove IDs from chunks (they no longer make sense here)
        clean_paragraphs = [
            re.sub(r'^ID \d+:\s*', '', chunk) for chunk in id_chunks
        ]
        
        # Create final chunks by merging paragraphs between shift points
        final_chunks = []
        for i in range(len(new_id_list)):
            start_idx = new_id_list[i - 1] if i > 0 else 0
            end_idx = new_id_list[i]
            
            chunk_text = '\n'.join(clean_paragraphs[start_idx:end_idx])
            
            if return_metadata:
                final_chunks.append({
                    "text": chunk_text,
                    "start_paragraph": start_idx,
                    "end_paragraph": end_idx,
                    "paragraph_count": end_idx - start_idx,
                })
            else:
                final_chunks.append(chunk_text)
        
        return final_chunks
