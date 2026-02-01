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
    
    DEFAULT_MODEL = "gemini-2.0-flash"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: The Gemini API key. If None, uses GEMINI_API_KEY env var.
            model: The model to use. Defaults to "gemini-2.0-flash".
            temperature: Sampling temperature (0.0-1.0).
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay in seconds between retries.
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
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
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            The generated text response.
        """
        def _generate():
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                }
            )
            return response.text
        
        return self._retry_with_backoff(_generate)
