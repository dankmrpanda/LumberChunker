#!/usr/bin/env python3
"""
LumberChunker CLI — semantic document segmentation from the command line.

Run with:
    python -m lumberchunker -h
    lumberchunker -h          (if installed via pip)
"""

import argparse
import os
from pathlib import Path


def _load_dotenv():
    """Best-effort load of .env file."""
    try:
        from dotenv import load_dotenv
        for candidate in [Path.cwd() / ".env", Path.home() / ".env"]:
            if candidate.exists():
                load_dotenv(candidate, override=False)
                break
    except ImportError:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lumberchunker",
        description=(
            "LumberChunker — LLM-powered semantic document segmentation.\n\n"
            "Chunk plain-text or EPUB files into semantically coherent segments\n"
            "using large language models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  lumberchunker document.txt\n"
            "  lumberchunker book.epub --chapters 1 3 5\n"
            "  lumberchunker book.epub --trim --provider gemini\n"
            "  lumberchunker book.epub --list-chapters\n"
            "  lumberchunker book.epub --output-dir my_output\n"
        ),
    )

    # --- Positional ---
    parser.add_argument(
        "file", type=Path,
        help="Path to the input file (.txt, .epub, or other text format).",
    )

    # --- Provider options ---
    parser.add_argument(
        "--provider", type=str, default="gemini",
        choices=["gemini", "openai", "anthropic", "ollama"],
        help="LLM provider to use (default: gemini).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the provider. Falls back to the standard env var "
             "(GEMINI_API_KEY, OPENAI_API_KEY, etc.) if not specified.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name to use. If omitted, uses the provider's default.",
    )

    # --- Chunking options ---
    parser.add_argument(
        "--target-tokens", type=int, default=550,
        help="Target token count per chunk window (default: 550).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path for plain-text chunking, or output directory "
             "for EPUB chunking. For EPUBs, 'auto' (default when omitted for "
             "EPUBs) creates a folder from the filename.",
    )

    # --- EPUB-specific options ---
    epub_group = parser.add_argument_group("EPUB options")
    epub_group.add_argument(
        "--chapters", nargs="+",
        help="Chapters to chunk. Pass 1-based numbers (e.g., 1 3 5) or title "
             "substrings. Omit to chunk all chapters.",
    )
    epub_group.add_argument(
        "--list-chapters", action="store_true",
        help="List available chapters in the EPUB and exit.",
    )
    epub_group.add_argument(
        "--use-llm-extraction", action="store_true",
        help="Use LLM for chapter boundary detection (slower but more "
             "accurate). By default, uses heuristic TOC-based extraction.",
    )
    epub_group.add_argument(
        "--trim", action="store_true",
        help="Post-processing: use the LLM to detect and remove front/back "
             "matter chapters (e.g., title page, copyright, acknowledgements).",
    )

    # --- General ---
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output (only print final results).",
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Include metadata (paragraph ranges) in each chunk.",
    )

    args = parser.parse_args(argv)

    # --- Validate ---
    if not args.file.exists():
        parser.error(f"File not found: {args.file}")

    _load_dotenv()

    # Resolve API key from env if not passed directly
    api_key = args.api_key
    if api_key is None:
        env_var_map = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_var_map.get(args.provider)
        if env_var:
            api_key = os.environ.get(env_var)

    # --- Build chunker ---
    from lumberchunker import LumberChunker

    chunker = LumberChunker(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        target_chunk_tokens=args.target_tokens,
        verbose=not args.quiet,
    )

    is_epub = args.file.suffix.lower() == ".epub"

    # --- EPUB path ---
    if is_epub:
        # --list-chapters
        if args.list_chapters:
            print("\nAvailable chapters:")
            for ch in chunker.list_chapters(args.file):
                print(f"  {ch['index']:3d}. {ch['chapter']}  "
                      f"({ch['word_count']:,} words)")
            print("\nTip: pass --chapters 1 3 5 to chunk specific chapters.")
            return 0

        # Parse --chapters
        chapter_filter = None
        if args.chapters:
            try:
                chapter_filter = [int(c) for c in args.chapters]
            except ValueError:
                chapter_filter = args.chapters

        output_dir = args.output if args.output else "auto"

        results = chunker.chunk_epub(
            args.file,
            output_dir=output_dir,
            chapters=chapter_filter,
            use_llm_extraction=args.use_llm_extraction,
            trim_front_back_matter=args.trim,
        )

        total_chunks = sum(len(r["chunks"]) for r in results)
        print(f"\nDone! {total_chunks} chunks from {len(results)} chapters.")

    # --- Plain text path ---
    else:
        output_path = args.output
        results = chunker.chunk_file(
            args.file,
            return_metadata=args.metadata,
            output_path=output_path,
        )
        print(f"\nDone! {len(results)} chunks.")

    # Usage summary
    stats = chunker.usage.to_dict()
    print(f"LLM usage: {stats['prompt_count']} calls, "
          f"{stats['total_tokens']:,} total tokens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
