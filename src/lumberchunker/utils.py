"""
Utility functions for LumberChunker.

These functions are derived from the original LumberChunker research implementation.
"""

import re
from pathlib import Path
from typing import List, Union


def count_words(input_string: str) -> int:
    """
    Count words and estimate token count.
    
    This function approximates the number of tokens in the sentence.
    Assuming 1 word ~ 1.2 tokens (from original implementation).
    
    Args:
        input_string: The text to count tokens for.
        
    Returns:
        Estimated token count.
    """
    words = input_string.split()
    return round(1.2 * len(words))


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.
    
    Splits on double newlines or multiple consecutive newlines.
    
    Args:
        text: The text to split.
        
    Returns:
        List of paragraph strings.
    """
    # Split on multiple newlines
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    
    # Filter out empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def add_ids_to_chunks(chunks: List[str]) -> List[str]:
    """
    Add ID prefixes to each chunk.
    
    Args:
        chunks: List of text chunks.
        
    Returns:
        List of chunks with "ID X: " prefix.
    """
    return [f"ID {i}: {chunk}" for i, chunk in enumerate(chunks)]


def remove_ids_from_chunks(chunks: List[str]) -> List[str]:
    """
    Remove ID prefixes from chunks.
    
    Args:
        chunks: List of chunks with ID prefixes.
        
    Returns:
        List of chunks without ID prefixes.
    """
    pattern = r'^ID \d+:\s*'
    return [re.sub(pattern, '', chunk) for chunk in chunks]


def extract_id_from_response(response: str) -> int:
    """
    Extract the paragraph ID from an LLM response.
    
    The response should contain "Answer: ID XXXX" format.
    Also attempts to find just "ID XXXX" if the strict format is missing.
    
    Args:
        response: The LLM response text.
        
    Returns:
        The extracted ID number, or -1 if not found.
    """
    # 1. Try strict format: "Answer: ID XXXX"
    pattern = r"Answer: ID \w+"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        id_text = match.group(0)
        id_pattern = r'\d+'
        id_match = re.search(id_pattern, id_text)
        if id_match:
            return int(id_match.group())
            
    # 2. Fallback: Look for any "ID XXXX" occurrence
    # We prioritize IDs appearing at the start of a line or sentence help
    pattern_loose = r"ID\s*(\d+)"
    match = re.search(pattern_loose, response, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    
    return -1


def is_epub(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is an EPUB based on extension.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        True if the file has .epub extension.
    """
    path = Path(file_path)
    return path.suffix.lower() == '.epub'


def epub_to_text(epub_path: Union[str, Path]) -> str:
    """
    Convert an EPUB file to plain text.
    
    Extracts text content from all document items in the EPUB,
    preserving paragraph structure.
    
    Args:
        epub_path: Path to the EPUB file.
        
    Returns:
        Extracted text content as a string.
        
    Raises:
        ImportError: If ebooklib or bs4 are not installed.
        FileNotFoundError: If the EPUB file doesn't exist.
        
    Example:
        >>> text = epub_to_text("book.epub")
        >>> chunks = chunker.chunk(text)
    """
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise ImportError(
            "ebooklib is required for EPUB support. "
            "Install it with: pip install ebooklib"
        )
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for EPUB support. "
            "Install it with: pip install beautifulsoup4"
        )
    
    path = Path(epub_path)
    if not path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")
    
    # Read the EPUB file
    book = epub.read_epub(str(path))
    
    text_parts = []
    
    # Iterate through all document items
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Extract text from paragraphs and other text elements
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']):
                text = element.get_text(strip=True)
                if text:
                    text_parts.append(text)
    
    # Join with double newlines to preserve paragraph structure
    return '\n\n'.join(text_parts)


def read_file_or_epub(file_path: Union[str, Path]) -> str:
    """
    Read text from a file, automatically converting EPUB if needed.
    
    Args:
        file_path: Path to the file (txt, epub, or other text format).
        
    Returns:
        Text content of the file.
        
    Example:
        >>> text = read_file_or_epub("book.epub")
        >>> # or
        >>> text = read_file_or_epub("document.txt")
    """
    path = Path(file_path)
    
    if is_epub(path):
        return epub_to_text(path)
    else:
        # Read as plain text
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

