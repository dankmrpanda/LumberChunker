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

# Configuration for supported providers
PROVIDERS = {
    "gemini": {
        "name": "Google Gemini",
        "api_key_var": "GEMINI_API_KEY",
        "model_var": "GEMINI_MODEL",
        "default_model": "gemini-2.0-flash"
    },
    "openai": {
        "name": "OpenAI",
        "api_key_var": "OPENAI_API_KEY",
        "model_var": "OPENAI_MODEL",
        "default_model": "gpt-4o"
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "api_key_var": "ANTHROPIC_API_KEY",
        "model_var": "ANTHROPIC_MODEL",
        "default_model": "claude-sonnet-4-20250514"
    },
    "ollama": {
        "name": "Ollama (Local)",
        "api_key_var": None,
        "model_var": "OLLAMA_MODEL",
        "default_model": "llama3.3"
    }
}

def save_chunks(chunks, provider):
    """Save chunks to a text file."""
    filename = f"chunks_{provider}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                # Handle both string chunks and dictionary chunks (with metadata)
                if isinstance(chunk, dict):
                    f.write(str(chunk.get('text', chunk)))
                else:
                    f.write(str(chunk))
                f.write("\n\n")
        print(f"Saved chunks to: {filename}")
    except Exception as e:
        print(f"Failed to save chunks: {e}")

def main():
    print("LumberChunker Example")
    print("---------------------")
    
    # 1. Select Provider
    print("Select a provider:")
    provider_keys = list(PROVIDERS.keys())
    for i, key in enumerate(provider_keys):
        p = PROVIDERS[key]
        status = ""
        if p["api_key_var"]:
             if os.environ.get(p["api_key_var"]):
                 status = "(key found)"
             else:
                 status = "(no key)"
        elif key == "ollama":
            status = "(local)"
            
        print(f"{i+1}. {p['name']} {status}")
    
    choice = input("\nEnter choice (1-4, q to quit): ").strip().lower()
    if choice == 'q':
        return

    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(provider_keys)):
            print("Invalid selection.")
            return
        provider_id = provider_keys[idx]
    except ValueError:
        print("Invalid input.")
        return

    # 2. Configure Provider
    config = PROVIDERS[provider_id]
    api_key_var = config["api_key_var"]
    
    if api_key_var and not os.environ.get(api_key_var):
        print(f"\nError: {api_key_var} not set in environment.")
        print("Please check your .env file or environment variables.")
        return
        
    api_key = os.environ.get(api_key_var) if api_key_var else None
    model = os.environ.get(config["model_var"], config["default_model"])
    
    print(f"\nInitializing {config['name']} with model: {model}")
    
    # 3. Run Chunking
    try:
        chunker = LumberChunker(
            provider=provider_id,
            api_key=api_key,
            model=model
        )
        
        print("Chunking sample text...")
        chunks = chunker.chunk(SAMPLE_TEXT)
        
        # 4. Show Results
        print(f"\nCreated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            # Create a short preview of the chunk content
            content = chunk['text'] if isinstance(chunk, dict) else chunk
            preview = content[:80].replace('\n', ' ') + "..." if len(content) > 80 else content
            print(f"  {i+1}: {preview}")
            
        save_chunks(chunks, provider_id)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        if provider_id == "ollama":
            print("Tip: Ensure Ollama is running ('ollama serve') and the model has been pulled.")

if __name__ == "__main__":
    main()
