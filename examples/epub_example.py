#!/usr/bin/env python3
"""
LumberChunker EPUB Example

Demonstrates how to chunk EPUB files using the LumberChunker library.

Usage:
    python epub_example.py <path_to_epub_file>
"""

import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
except ImportError:
    pass

from lumberchunker import LumberChunker, epub_to_text

# Provider configuration
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.0-flash"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "ollama": (None, "OLLAMA_MODEL", "llama3.3"),
}

def get_provider():
    """Auto-select a provider based on available API keys."""
    for name, (key_var, _, _) in PROVIDERS.items():
        if key_var is None:  # ollama
            continue
        if os.environ.get(key_var):
            return name
    # Fallback to ollama if no API keys are set
    return "ollama"

def save_chunks(chunks, output_path):
    """Save chunks to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            content = chunk['text'] if isinstance(chunk, dict) else chunk
            f.write(content)
            f.write("\n\n")
    print(f"Saved {len(chunks)} chunks to: {output_path}")

def main():
    # Check for EPUB file argument
    if len(sys.argv) < 2:
        print("Usage: python epub_example.py <path_to_epub_file>")
        print("\nExample:")
        print("  python epub_example.py my_book.epub")
        sys.exit(1)

    epub_path = Path(sys.argv[1])
    
    if not epub_path.exists():
        print(f"Error: File not found: {epub_path}")
        sys.exit(1)
    
    if not epub_path.suffix.lower() == ".epub":
        print(f"Warning: File does not have .epub extension: {epub_path}")

    # Select provider
    provider_id = get_provider()
    key_var, model_var, default_model = PROVIDERS[provider_id]
    
    api_key = os.environ.get(key_var) if key_var else None
    model = os.environ.get(model_var, default_model)
    
    print(f"EPUB Chunker")
    print(f"------------")
    print(f"Input:    {epub_path}")
    print(f"Provider: {provider_id}")
    print(f"Model:    {model}")

    # Extract text from EPUB
    print("\nExtracting text from EPUB...")
    try:
        text = epub_to_text(epub_path)
        word_count = len(text.split())
        print(f"Extracted {word_count:,} words")
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required packages: pip install ebooklib beautifulsoup4")
        sys.exit(1)
    except Exception as e:
        print(f"\nError reading EPUB: {e}")
        sys.exit(1)

    # Chunk the text
    print("\nChunking text...")
    try:
        chunker = LumberChunker(
            provider=provider_id,
            api_key=api_key,
            model=model
        )
        chunks = chunker.chunk(text)
    except Exception as e:
        print(f"\nError during chunking: {e}")
        if provider_id == "ollama":
            print("Tip: Ensure Ollama is running and the model is pulled.")
        sys.exit(1)

    # Results
    print(f"\nCreated {len(chunks)} chunks")
    
    # Show preview of first few chunks
    print("\nPreview (first 3 chunks):")
    for i, chunk in enumerate(chunks[:3]):
        content = chunk['text'] if isinstance(chunk, dict) else chunk
        preview = content[:100].replace('\n', ' ').strip()
        if len(content) > 100:
            preview += "..."
        print(f"  {i+1}: {preview}")
    
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")

    # Save to file
    output_path = epub_path.with_suffix(".chunks.txt")
    save_chunks(chunks, output_path)

if __name__ == "__main__":
    main()
