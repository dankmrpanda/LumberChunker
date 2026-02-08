#!/usr/bin/env python3
"""
LumberChunker Example Script

Demonstrates how to use different LLM providers (Gemini, OpenAI, Anthropic, Ollama)
to chunk text using the LumberChunker library.
"""

import os
from pathlib import Path
from lumberchunker import LumberChunker

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
            choice = input(f"\nEnter choice (1-{len(names)}, q to quit): ").strip().lower()
            if choice == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
        except (ValueError, IndexError):
            pass
        print("Invalid choice.")


def main():
    print("LumberChunker Example")
    print("---------------------")

    provider_id = select_provider()
    if provider_id is None:
        return

    key_var, model_var, default_model = PROVIDERS[provider_id]
    api_key = os.environ.get(key_var) if key_var else None
    model = os.environ.get(model_var, default_model)

    if key_var and not api_key:
        print(f"\nError: {key_var} not set in environment.")
        print("Please check your .env file or environment variables.")
        return

    print(f"\nUsing: {provider_id} / {model}")

    try:
        chunker = LumberChunker(
            provider=provider_id,
            api_key=api_key,
            model=model,
        )

        # Chunk and save — the library handles file output
        output_file = f"chunks_{provider_id}.txt"
        chunks = chunker.chunk(SAMPLE_TEXT, output_path=output_file)

        # Show results
        print(f"\nCreated {len(chunks)} chunks (saved to {output_file}):")
        for i, chunk in enumerate(chunks):
            content = chunk['text'] if isinstance(chunk, dict) else chunk
            preview = content[:80].replace('\n', ' ')
            if len(content) > 80:
                preview += "..."
            print(f"  {i+1}: {preview}")

    except Exception as e:
        print(f"\nError: {e}")
        if provider_id == "ollama":
            print("Tip: Ensure Ollama is running ('ollama serve') and the model has been pulled.")


if __name__ == "__main__":
    main()
