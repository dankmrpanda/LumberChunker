#!/usr/bin/env python3
"""
LumberChunker EPUB Example

Demonstrates how to chunk EPUB files using the LumberChunker library.
The library handles all output: chapter extraction, chunking, and saving.

Usage:
    python epub_example.py <path_to_epub_file>
    python epub_example.py <path_to_epub_file> --chapters 1 3 5
    python epub_example.py <path_to_epub_file> --list-chapters
"""

import argparse
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
except ImportError:
    pass

from lumberchunker import LumberChunker

# Provider configuration: (api_key_env_var, model_env_var, default_model)
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.0-flash"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "ollama": (None, "OLLAMA_MODEL", "llama3.3"),
}


def select_provider():
    """Interactive provider selection."""
    print("\nSelect LLM Provider:")
    print("--------------------")
    names = list(PROVIDERS.keys())
    for i, name in enumerate(names, 1):
        key_var = PROVIDERS[name][0]
        has_key = key_var is None or os.environ.get(key_var)
        status = "" if has_key else " (no API key)"
        print(f"  {i}. {name}{status}")

    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                name = names[idx]
                key_var = PROVIDERS[name][0]
                if key_var and not os.environ.get(key_var):
                    print(f"Warning: {key_var} not set. Provider may fail.")
                return name
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Please enter 1-4.")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk an EPUB file into semantically coherent segments.",
    )
    parser.add_argument("epub", type=Path, help="Path to the .epub file")
    parser.add_argument(
        "--chapters", nargs="+",
        help="Chapters to chunk. Pass numbers (1-based) or title substrings. "
             "Omit to chunk all chapters.",
    )
    parser.add_argument(
        "--list-chapters", action="store_true",
        help="List available chapters and exit (no chunking).",
    )
    args = parser.parse_args()

    epub_path = args.epub

    if not epub_path.exists():
        print(f"Error: File not found: {epub_path}")
        sys.exit(1)

    if not epub_path.suffix.lower() == ".epub":
        print(f"Warning: File does not have .epub extension: {epub_path}")

    print(f"EPUB Chapter Chunker")
    print(f"====================")
    print(f"Input: {epub_path}")

    # Select provider
    provider_id = select_provider()
    key_var, model_var, default_model = PROVIDERS[provider_id]
    api_key = os.environ.get(key_var) if key_var else None
    model = os.environ.get(model_var, default_model)

    print(f"\nUsing: {provider_id} / {model}")

    # Chunk the EPUB — the library handles output folder, file saving, and logging
    try:
        chunker = LumberChunker(
            provider=provider_id,
            api_key=api_key,
            model=model,
        )

        # --list-chapters: show available chapters and exit
        if args.list_chapters:
            print("\nAvailable chapters:")
            for ch in chunker.list_chapters(epub_path):
                print(f"  {ch['index']:3d}. {ch['chapter']}  ({ch['word_count']:,} words)")
            print("\nTip: pass --chapters 1 3 5 to chunk specific chapters.")
            sys.exit(0)

        # Parse --chapters argument (ints or strings)
        chapter_filter = None
        if args.chapters:
            # Try to interpret as ints first; fall back to string matching
            try:
                chapter_filter = [int(c) for c in args.chapters]
                print(f"Chapters: {chapter_filter} (by index)")
            except ValueError:
                chapter_filter = args.chapters
                print(f"Chapters: {chapter_filter} (by title)")

        results = chunker.chunk_epub(
            epub_path,
            output_dir="auto",        # auto-creates folder from EPUB filename
            chapters=chapter_filter,   # None = all, or list of ints/strings
            use_llm_extraction=False,   # use LLM for accurate chapter detection
        )

        # Summary
        total_chunks = sum(len(r['chunks']) for r in results)
        print(f"\nDone! {total_chunks} chunks from {len(results)} chapters.")
        stats = chunker.usage.to_dict()
        print(f"LLM usage: {stats['prompt_count']} calls, "
              f"{stats['total_tokens']:,} total tokens")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required packages: pip install ebooklib pydantic")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if provider_id == "ollama":
            print("Tip: Ensure Ollama is running and the model is pulled.")
        sys.exit(1)


if __name__ == "__main__":
    main()

