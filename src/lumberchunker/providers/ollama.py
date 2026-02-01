"""
Ollama provider for LumberChunker.

Uses the Ollama Python library for local LLM inference.
"""

import os
from typing import Optional

from lumberchunker.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Ollama local LLM provider.
    
    Supports any Ollama-compatible model (llama3.3, mistral, etc.).
    
    Example:
        >>> provider = OllamaProvider(model="llama3.3")
        >>> response = provider.generate("Hello, world!")
    """
    
    DEFAULT_MODEL = "llama3.3"
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ):
        """
        Initialize the Ollama provider.
        
        Args:
            model: The model to use. Defaults to "llama3.3".
            host: The Ollama server host. Defaults to OLLAMA_HOST env var or localhost.
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
        
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self._client = None
    
    def _get_client(self):
        """Lazily initialize the Ollama client."""
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host)
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Ollama.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.
            
        Returns:
            The generated text response.
        """
        def _generate():
            client = self._get_client()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                },
            )
            return response["message"]["content"]
        
        return self._retry_with_backoff(_generate)
