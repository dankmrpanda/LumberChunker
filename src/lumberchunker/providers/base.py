"""
Base class for LLM providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Cumulative LLM usage statistics."""

    prompt_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0

    # Last-call metrics (reset on each call)
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_duration_ms: float = 0.0

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
    ):
        """Record metrics from a single LLM call."""
        self.prompt_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.total_duration_ms += duration_ms
        self.last_input_tokens = input_tokens
        self.last_output_tokens = output_tokens
        self.last_duration_ms = duration_ms

    def reset(self):
        """Reset all counters to zero."""
        self.prompt_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_duration_ms = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_duration_ms = 0.0

    def to_dict(self) -> dict:
        """Return stats as a plain dict."""
        return {
            "prompt_count": self.prompt_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_ms": round(self.total_duration_ms, 1),
        }


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement the `generate` method which takes a prompt
    and returns a text response.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 60.0,
        timeout: float = 120.0,
    ):
        """
        Initialize the base LLM provider.
        
        Args:
            model: The model identifier to use. If None, uses provider default.
            temperature: Sampling temperature (0.0-1.0). Lower is more deterministic.
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Delay in seconds between retries.
            timeout: HTTP request timeout in seconds (default: 120).
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.usage = UsageStats()
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt to send to the LLM.
            
        Returns:
            The generated text response.
            
        Raises:
            Exception: If generation fails after all retries.
        """
        pass
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function call.
            
        Raises:
            Exception: If all retries are exhausted.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check for content safety/blocking errors - don't retry these
                if "list index out of range" in error_str:
                    return "content_flag_increment"
                
                logger.warning(f"An error occurred: {e}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        raise last_exception
