#!/usr/bin/env python3
"""
LumberChunker EPUB Example

Demonstrates how to chunk EPUB files using the LumberChunker library
with intelligent chapter extraction.

Usage:
    python epub_example.py <path_to_epub_file>
"""

import os
import sys
from pathlib import Path
from io import StringIO

# Load environment variables fresh from .env file each time
# Using override=True ensures .env values take precedence over system environment variables
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
except ImportError:
    pass

from lumberchunker import LumberChunker, epub_to_chapters

# Provider configuration
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.0-flash"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "ollama": (None, "OLLAMA_MODEL", "llama3.3"),
}

def get_available_providers():
    """Get list of providers with available API keys."""
    available = []
    for name, (key_var, _, _) in PROVIDERS.items():
        if key_var is None:  # ollama - always available
            available.append((name, True))
        elif os.environ.get(key_var):
            available.append((name, True))
        else:
            available.append((name, False))
    return available

def select_provider():
    """Interactive provider selection."""
    available = get_available_providers()
    
    print("\nSelect LLM Provider:")
    print("--------------------")
    for i, (name, has_key) in enumerate(available, 1):
        status = "" if has_key else " (no API key)"
        print(f"  {i}. {name}{status}")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                name, has_key = available[idx]
                if not has_key:
                    key_var = PROVIDERS[name][0]
                    print(f"Warning: {key_var} not set. Provider may fail.")
                return name
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Please enter 1-4.")

def create_output_folder(epub_path: Path) -> Path:
    """Create output folder for EPUB processing results.
    
    Creates a folder based on the EPUB filename. If a folder with the same name
    already exists, appends a number (1, 2, 3, etc.) to create a unique folder.
    
    Args:
        epub_path: Path to the EPUB file.
        
    Returns:
        Path to the created output folder.
    """
    base_name = epub_path.stem  # Get filename without extension
    parent_dir = epub_path.parent
    
    # Try the base folder name first
    output_folder = parent_dir / base_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder
    
    # If it exists, add incrementing numbers until we find a unique name
    counter = 1
    while True:
        output_folder = parent_dir / f"{base_name}_{counter}"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
            return output_folder
        counter += 1

def save_epub_chapters_with_ids(chapters: list, output_path: Path):
    """Save EPUB chapters with paragraph ID markers to a text file.
    
    This saves the output from the EPUB converter, with IDs assigned to each paragraph.
    
    Args:
        chapters: List of chapter dicts with 'chapter', 'text', and 'text_with_ids' keys.
        output_path: Path to save the output file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("EPUB Converter Output - Paragraphs with IDs\n")
        f.write("=" * 50 + "\n\n")
        
        for i, ch in enumerate(chapters):
            chapter_name = ch['chapter']
            # Use text_with_ids which has paragraph-level IDs
            chapter_text_with_ids = ch.get('text_with_ids', ch['text'])
            word_count = len(ch['text'].split())
            
            f.write(f"=== Chapter {i}: {chapter_name} ===\n")
            f.write(f"Word Count: {word_count:,}\n")
            f.write("-" * 40 + "\n\n")
            f.write(chapter_text_with_ids)
            f.write("\n\n" + "=" * 50 + "\n\n")
    
    print(f"Saved {len(chapters)} chapters with paragraph IDs to: {output_path}")

def save_final_chunks(all_chunks: list, output_path: Path):
    """Save final chunking results to a text file.
    
    Args:
        all_chunks: List of (chapter_name, chunks) tuples.
        output_path: Path to save the output file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Final Chunking Output\n")
        f.write("=" * 50 + "\n\n")
        
        for chapter_name, chunks in all_chunks:
            f.write(f"=== {chapter_name} ===\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                content = chunk['text'] if isinstance(chunk, dict) else chunk
                f.write(content)
                f.write("\n\n")
    
    total_chunks = sum(len(chunks) for _, chunks in all_chunks)
    print(f"Saved {total_chunks} chunks from {len(all_chunks)} chapters to: {output_path}")


class TeeLogger:
    """A class that writes to both stdout and a file simultaneously."""
    
    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.close()
        return False

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

    print(f"EPUB Chapter Chunker")
    print(f"====================")
    print(f"Input: {epub_path}")

    # Create output folder for this EPUB
    output_folder = create_output_folder(epub_path)
    print(f"Output folder: {output_folder}")

    # Let user select provider
    provider_id = select_provider()
    key_var, model_var, default_model = PROVIDERS[provider_id]
    
    api_key = os.environ.get(key_var) if key_var else None
    model = os.environ.get(model_var, default_model)
    
    print(f"\nUsing: {provider_id} / {model}")

    # Step 1: Extract chapters from EPUB
    print("\nStep 1: Extracting chapters from EPUB...")
    print("(Using LLM to detect narrative boundaries and clean chapter starts)")
    try:
        chapters = epub_to_chapters(
            epub_path,
            provider=provider_id,
            api_key=api_key,
            model=model
        )
        print(f"Extracted {len(chapters)} chapters")
        
        for i, ch in enumerate(chapters):
            word_count = len(ch['text'].split())
            print(f"  {i+1}. {ch['chapter']} ({word_count:,} words)")
        
        # Save epub converter output with IDs
        epub_converter_output = output_folder / "epub_chapters_with_ids.txt"
        save_epub_chapters_with_ids(chapters, epub_converter_output)
            
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required packages: pip install ebooklib pydantic")
        sys.exit(1)
    except Exception as e:
        print(f"\nError extracting chapters: {e}")
        sys.exit(1)

    # Step 2: Chunk each chapter with logging
    print("\nStep 2: Chunking each chapter...")
    
    # Set up logging to file
    log_file_path = output_folder / "llm_log.txt"
    print(f"Logging LLM inputs/outputs to: {log_file_path}")
    
    try:
        chunker = LumberChunker(
            provider=provider_id,
            api_key=api_key,
            model=model
        )
        
        all_chunks = []
        total_chunks = 0
        
        # Use TeeLogger to capture all output to both console and file
        with TeeLogger(log_file_path) as logger:
            for ch in chapters:
                chapter_name = ch['chapter']
                chapter_text = ch['text']
                
                if not chapter_text.strip():
                    print(f"  Skipping empty chapter: {chapter_name}")
                    continue
                
                print(f"\n{'#'*60}")
                print(f"# PROCESSING CHAPTER: {chapter_name}")
                print(f"{'#'*60}")
                
                chunks = chunker.chunk(chapter_text)
                all_chunks.append((chapter_name, chunks))
                total_chunks += len(chunks)
                print(f"\n  {chapter_name}: {len(chunks)} chunks created")
            
    except Exception as e:
        print(f"\nError during chunking: {e}")
        if provider_id == "ollama":
            print("Tip: Ensure Ollama is running and the model is pulled.")
        sys.exit(1)

    # Results summary
    print(f"\nResults")
    print(f"-------")
    print(f"Total chapters: {len(all_chunks)}")
    print(f"Total chunks:   {total_chunks}")
    
    # Show preview of first few chunks from first chapter
    if all_chunks:
        first_chapter, first_chunks = all_chunks[0]
        print(f"\nPreview (first 3 chunks from '{first_chapter}'):")
        for i, chunk in enumerate(first_chunks[:3]):
            content = chunk['text'] if isinstance(chunk, dict) else chunk
            preview = content[:100].replace('\n', ' ').strip()
            if len(content) > 100:
                preview += "..."
            print(f"  {i+1}: {preview}")
        
        if len(first_chunks) > 3:
            print(f"  ... and {len(first_chunks) - 3} more chunks in this chapter")

    # Save final chunks to the output folder
    final_chunks_output = output_folder / "final_chunks.txt"
    save_final_chunks(all_chunks, final_chunks_output)
    
    print(f"\nAll outputs saved to: {output_folder}")

if __name__ == "__main__":
    main()

