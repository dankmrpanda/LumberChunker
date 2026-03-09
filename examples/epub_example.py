#!/usr/bin/env python3
"""
LumberChunker EPUB Example

Demonstrates how to chunk EPUB files using the LumberChunker library.
The library handles all output: chapter extraction, chunking, and saving.

Configure the variables below, then run:
    python epub_example.py
"""

import os
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these variables before running
# ============================================================

# Path to the EPUB file
EPUB_FILE = Path(__file__).parent / "sources" / "epubs" / "Autobiography of Benjamin Franklin - Benjamin Franklin.epub"

# Provider: "gemini", "openai", "anthropic", or "ollama"
PROVIDER = "gemini"

# Model override (set to None to use the provider's default)
MODEL = "gemini-3.1-flash-lite-preview"  # e.g. "gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514", "llama3.3"

# Base location for all output, relative to this script.
OUTPUT_LOCATION = Path(__file__).parent / "output"

# Output directory name: "auto" creates a subfolder from the EPUB filename
#   inside OUTPUT_LOCATION.  Set an explicit name to override.
#   If the folder already contains a checkpoint, the run resumes.
OUTPUT_DIR = "auto"

# Which chapters to chunk:
#   None          -> all chapters
#   [1, 3, 5]     -> chapters by 1-based index
#   ["Prologue"]  -> chapters by title substring (case-insensitive)
CHAPTERS = [1]

# Set to True to only list chapters (no chunking)
LIST_CHAPTERS_ONLY = False

# Use LLM for chapter boundary detection (more accurate, slower)
USE_LLM_EXTRACTION = True

# Post-processing: trim front/back matter chapters (title page, copyright, etc.)
TRIM_FRONT_BACK_MATTER = True

# ============================================================
# END CONFIGURATION
# ============================================================

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

# Resolve API key from environment
API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def main():
    epub_path = Path(EPUB_FILE)

    if not epub_path.exists():
        print(f"Error: File not found: {epub_path}")
        sys.exit(1)

    if epub_path.suffix.lower() != ".epub":
        print(f"Warning: File does not have .epub extension: {epub_path}")

    # Resolve output directory relative to this script
    output_location = Path(OUTPUT_LOCATION)
    output_location.mkdir(parents=True, exist_ok=True)

    if isinstance(OUTPUT_DIR, str) and OUTPUT_DIR.lower() == "auto":
        resolved_output = output_location / epub_path.stem
    else:
        resolved_output = output_location / OUTPUT_DIR

    print("EPUB Chapter Chunker")
    print("====================")
    print(f"Input    : {epub_path}")
    print(f"Output   : {resolved_output}")

    # Resolve API key
    key_var = API_KEY_ENV_VARS.get(PROVIDER)
    api_key = os.environ.get(key_var) if key_var else None

    model_display = MODEL or "(provider default)"
    print(f"Provider : {PROVIDER}")
    print(f"Model    : {model_display}")

    try:
        chunker = LumberChunker(
            provider=PROVIDER,
            api_key=api_key,
            model=MODEL,
        )

        # List chapters only
        if LIST_CHAPTERS_ONLY:
            print("\nAvailable chapters:")
            for ch in chunker.list_chapters(epub_path):
                print(f"  {ch['index']:3d}. {ch['chapter']}  ({ch['word_count']:,} words)")
            print("\nTip: set CHAPTERS = [1, 3, 5] to chunk specific chapters.")
            sys.exit(0)

        # Print active options
        if CHAPTERS:
            print(f"Chapters : {CHAPTERS}")
        if TRIM_FRONT_BACK_MATTER:
            print("Trim     : front/back matter trimming enabled")

        results = chunker.chunk_epub(
            epub_path,
            output_dir=resolved_output,
            chapters=CHAPTERS,
            use_llm_extraction=USE_LLM_EXTRACTION,
            trim_front_back_matter=TRIM_FRONT_BACK_MATTER,
        )

        # Summary
        total_chunks = sum(len(r['chunks']) for r in results)
        print(f"\nDone! {total_chunks} chunks from {len(results)} chapters.")
        if TRIM_FRONT_BACK_MATTER:
            print("(boundary trimming was enabled)")
        stats = chunker.usage.to_dict()
        print(f"LLM usage: {stats['prompt_count']} calls, "
              f"{stats['total_tokens']:,} total tokens")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required packages: pip install ebooklib pydantic")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if PROVIDER == "ollama":
            print("Tip: Ensure Ollama is running and the model is pulled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
