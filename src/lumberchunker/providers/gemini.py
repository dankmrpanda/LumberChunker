"""
Google Gemini provider for LumberChunker.

Uses the google-genai SDK (recommended as of 2025+).
"""

import os
from typing import Optional

from lumberchunker.providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider.
    
    This is the default provider for LumberChunker.
    
    Example:
        >>> provider = GeminiProvider(api_key="your-api-key")
        >>> response = provider.generate("Hello, world!")
    """
    
    DEFAULT_MODEL = "gemini-2.0-flash-lite"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 60.0,
        timeout: float = 120.0,
    ):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: The Gemini API key. If None, uses GEMINI_API_KEY env var.
            model: The model to use. Defaults to "gemini-2.0-flash".
            temperature: Sampling temperature (0.0-1.0).
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay in seconds between retries.
            timeout: HTTP request timeout in seconds (default: 120).
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. "
                "Pass api_key parameter or set GEMINI_API_KEY environment variable."
            )
        
        # Initialize the client
        self._client = None
    
    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            from google import genai
            from google.genai import types as genai_types
            http_options = genai_types.HttpOptions(timeout=int(self.timeout * 1000))
            self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The input prompt.
            system_prompt: Optional system prompt for context.
            
        Returns:
            The generated text response.
        """
        import time as _time

        def _generate():
            client = self._get_client()
            
            # Combine system prompt with user content for Gemini
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            t0 = _time.perf_counter()
            response = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "temperature": self.temperature,
                }
            )
            elapsed_ms = (_time.perf_counter() - t0) * 1000

            # Record token usage from Gemini's usage_metadata
            input_tok = 0
            output_tok = 0
            meta = getattr(response, "usage_metadata", None)
            if meta:
                input_tok = getattr(meta, "prompt_token_count", 0) or 0
                output_tok = getattr(meta, "candidates_token_count", 0) or 0
            self.usage.record(input_tokens=input_tok, output_tokens=output_tok, duration_ms=elapsed_ms)

            return response.text
        
        return self._retry_with_backoff(_generate)
