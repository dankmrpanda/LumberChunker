#!/usr/bin/env python3
"""
LumberChunker Example Script

Demonstrates how to use different LLM providers (Gemini, OpenAI, Anthropic, Ollama)
to chunk text using the LumberChunker library.

Configure the variables below, then run:
    python example.py
"""

import os
from pathlib import Path

# ============================================================
# CONFIGURATION — edit these variables before running
# ============================================================

# Input file (set to None to use the built-in SAMPLE_TEXT instead)
INPUT_FILE = Path(__file__).parent / "sources" / "5af1a7340e89__And_Then_There_Were_None_by_Agatha_Christie_CH1.txt"

# Provider: "gemini", "openai", "anthropic", or "ollama"
PROVIDER = "gemini"

# Model override (set to None to use the provider's default)
MODEL = None  # e.g. "gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514", "llama3.3"

# Output file for the chunked results
OUTPUT_FILE = Path(__file__).parent / "output" / "chunks.txt"

# Target tokens per chunk window
TARGET_CHUNK_TOKENS = 550

# ============================================================
# END CONFIGURATION
# ============================================================

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

SAMPLE_TEXT = """
"What about a story?" I said.

"Could you very sweetly tell Winnie-the-Pooh one?"

"I suppose I could," I said. "What sort of stories does he like?"

"About himself. Because he's that sort of Bear."

"Oh, I see."

"So could you very sweetly?"

"I'll try," I said.

So I tried.

Once upon a time, a very long time ago now, about last Friday, Winnie-the-Pooh lived in a forest all by himself under the name of Sanders.

"What does 'under the name' mean?" asked Christopher Robin.
"""

# Resolve API key from environment
API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}

from lumberchunker import LumberChunker


def main():
    print("LumberChunker Example")
    print("---------------------")

    # Resolve API key
    key_var = API_KEY_ENV_VARS.get(PROVIDER)
    api_key = os.environ.get(key_var) if key_var else None

    if key_var and not api_key:
        print(f"\nError: {key_var} not set in environment.")
        print("Please set it in your .env file or shell environment.")
        return

    model_display = MODEL or "(provider default)"
    print(f"Provider : {PROVIDER}")
    print(f"Model    : {model_display}")

    try:
        chunker = LumberChunker(
            provider=PROVIDER,
            api_key=api_key,
            model=MODEL,
            target_chunk_tokens=TARGET_CHUNK_TOKENS,
        )

        # Ensure output folder exists
        OUTPUT_FILE.parent.mkdir(exist_ok=True)

        # Chunk from file or built-in sample text
        if INPUT_FILE and Path(INPUT_FILE).exists():
            print(f"Input    : {INPUT_FILE}")
            chunks = chunker.chunk_file(str(INPUT_FILE), output_path=str(OUTPUT_FILE))
        else:
            if INPUT_FILE:
                print(f"Warning: file not found ({INPUT_FILE}), using built-in sample text")
            else:
                print("Input    : (built-in sample text)")
            chunks = chunker.chunk(SAMPLE_TEXT, output_path=str(OUTPUT_FILE))

        # Show results
        print(f"\nCreated {len(chunks)} chunks (saved to {OUTPUT_FILE}):")
        for i, chunk in enumerate(chunks):
            content = chunk['text'] if isinstance(chunk, dict) else chunk
            preview = content[:80].replace('\n', ' ')
            if len(content) > 80:
                preview += "..."
            print(f"  {i+1}: {preview}")

    except Exception as e:
        print(f"\nError: {e}")
        if PROVIDER == "ollama":
            print("Tip: Ensure Ollama is running ('ollama serve') and the model has been pulled.")


if __name__ == "__main__":
    main()
