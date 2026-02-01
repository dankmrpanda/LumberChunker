"""
LumberChunker - LLM-powered semantic document segmentation.

This package provides tools for dynamically segmenting long-form documents
into semantically coherent chunks using large language models.
"""

from lumberchunker.chunker import LumberChunker
from lumberchunker.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
)
from lumberchunker.utils import epub_to_text, read_file_or_epub

__version__ = "0.1.0"
__all__ = [
    "LumberChunker",
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "epub_to_text",
    "read_file_or_epub",
]

