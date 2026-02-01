"""
Base class for LLM providers.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional


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
    ):
        """
        Initialize the base LLM provider.
        
        Args:
            model: The model identifier to use. If None, uses provider default.
            temperature: Sampling temperature (0.0-1.0). Lower is more deterministic.
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Delay in seconds between retries.
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
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
                
                print(f"An error occurred: {e}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        raise last_exception
