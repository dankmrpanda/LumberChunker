"""
OpenAI provider for LumberChunker.

Uses the OpenAI Python SDK v2.x.
"""

import os
from typing import Optional

from lumberchunker.providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider.
    
    Supports GPT-3.5, GPT-4, and GPT-4o models.
    
    Example:
        >>> provider = OpenAIProvider(api_key="your-api-key")
        >>> response = provider.generate("Hello, world!")
    """
    
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: The OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: The model to use. Defaults to "gpt-4o".
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
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Pass api_key parameter or set OPENAI_API_KEY environment variable."
            )
        
        self._client = None
    
    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using OpenAI.
        
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
            
            completion = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            return completion.choices[0].message.content
        
        return self._retry_with_backoff(_generate)
