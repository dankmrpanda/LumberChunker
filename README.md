# LumberChunker

LLM-powered semantic document segmentation for long-form narrative documents.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Based on the research paper [LumberChunker: Long-Form Narrative Document Segmentation](https://arxiv.org/abs/2406.17526).

![LumberChunker Pipeline](LumberChunker_pipeline.png)

## Installation

```bash
pip install lumberchunker
```

## Quick Start

```python
from lumberchunker import LumberChunker

chunker = LumberChunker(api_key="your-gemini-api-key")
chunks = chunker.chunk("Your long document text here...")

# Save to file
chunks = chunker.chunk("Your text...", output_path="chunks.txt")
```

Other providers: `"openai"`, `"anthropic"`, `"ollama"` (local, no key needed).

```python
chunker = LumberChunker(provider="openai", api_key="your-key")
chunker = LumberChunker(provider="ollama", model="llama3.3")
```

## EPUB Support

```python
# Simple — whole book as one text
chunks = chunker.chunk_file("book.epub")

# Chapter-by-chapter with auto-saved output
results = chunker.chunk_epub("book.epub", output_dir="auto")

# Select specific chapters (by index or title)
chunker.list_chapters("book.epub")                          # see what's available
results = chunker.chunk_epub("book.epub", chapters=[1, 3])  # by index
results = chunker.chunk_epub("book.epub", chapters=["Prologue"])  # by title

# LLM-powered chapter detection (removes front/back matter)
results = chunker.chunk_epub("book.epub", use_llm_extraction=True)
```

## Token Usage

```python
chunks = chunker.chunk("Your text...")
print(chunker.usage.to_dict())
# {'prompt_count': 12, 'total_input_tokens': 8420, 'total_output_tokens': 156,
#  'total_tokens': 8576, 'total_duration_ms': 4230.5}

chunker.usage.reset()  # zero counters between runs
```

## Configuration

Set API keys via environment variables or a `.env` file:

```bash
GEMINI_API_KEY=your-key      # default provider
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

| Provider | Default Model |
|----------|---------------|
| Gemini (default) | `gemini-2.0-flash` |
| OpenAI | `gpt-4o` |
| Anthropic | `claude-sonnet-4-20250514` |
| Ollama | `llama3.3` |

## API Reference

| Method | Description |
|--------|-------------|
| `LumberChunker(provider, api_key, model, target_chunk_tokens=550, temperature=0.1, verbose=True)` | Create a chunker |
| `chunk(text, output_path=None, return_metadata=False)` | Chunk a string |
| `chunk_file(file_path, output_path=None, return_metadata=False)` | Chunk a text/EPUB file |
| `chunk_epub(epub_path, output_dir=None, chapters=None, use_llm_extraction=False)` | Chunk EPUB by chapters |
| `list_chapters(epub_path)` | List chapters without chunking |
| `usage` | `UsageStats` — prompt_count, total_input/output/total_tokens, total_duration_ms |

## Examples

```bash
# Chunk a text document (interactive provider selection)
python examples/example.py

# Chunk an EPUB file
python examples/epub_example.py path/to/book.epub

# List chapters in an EPUB
python examples/epub_example.py path/to/book.epub --list-chapters

# Chunk specific chapters only
python examples/epub_example.py path/to/book.epub --chapters 1 3 5
```

## Citation

```bibtex
@misc{duarte2024lumberchunker,
      title={LumberChunker: Long-Form Narrative Document Segmentation}, 
      author={André V. Duarte and João Marques and Miguel Graça and Miguel Freire and Lei Li and Arlindo L. Oliveira},
      year={2024},
      eprint={2406.17526},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17526}, 
}
```

## License

MIT License
