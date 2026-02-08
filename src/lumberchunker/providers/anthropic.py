"""
Anthropic Claude provider for LumberChunker.

Uses the Anthropic Python SDK.
"""

import os
from typing import Optional

from lumberchunker.providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider.
    
    Supports Claude Sonnet and Opus models.
    
    Example:
        >>> provider = AnthropicProvider(api_key="your-api-key")
        >>> response = provider.generate("Hello, world!")
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: The Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: The model to use. Defaults to "claude-sonnet-4-20250514".
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay in seconds between retries.
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Pass api_key parameter or set ANTHROPIC_API_KEY environment variable."
            )
        
        self._client = None
    
    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Anthropic Claude.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.
            
        Returns:
            The generated text response.
        """
        import time as _time

        def _generate():
            client = self._get_client()
            
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt

            t0 = _time.perf_counter()
            message = client.messages.create(**kwargs)
            elapsed_ms = (_time.perf_counter() - t0) * 1000

            # Record token usage from Anthropic's usage object
            input_tok = 0
            output_tok = 0
            if message.usage:
                input_tok = getattr(message.usage, "input_tokens", 0) or 0
                output_tok = getattr(message.usage, "output_tokens", 0) or 0
            self.usage.record(input_tokens=input_tok, output_tokens=output_tok, duration_ms=elapsed_ms)

            return message.content[0].text
        
        return self._retry_with_backoff(_generate)
