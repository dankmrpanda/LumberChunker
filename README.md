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

# Using Gemini (default)
chunker = LumberChunker(api_key="your-gemini-api-key")
chunks = chunker.chunk("Your long document text here...")

# Using OpenAI
chunker = LumberChunker(provider="openai", api_key="your-openai-key")

# Using Ollama (local, no API key needed)
chunker = LumberChunker(provider="ollama", model="llama3.3")
```

## EPUB Support

### Simple Extraction
```python
chunks = chunker.chunk_file("book.epub")
```

### Intelligent Chapter Extraction

Extract chapters with LLM-powered boundary detection (removes front/back matter automatically):

```python
from lumberchunker import LumberChunker, epub_to_chapters

# Both steps use the same provider
chapters = epub_to_chapters("book.epub", provider="gemini", api_key="...")
chunker = LumberChunker(provider="gemini", api_key="...")

for ch in chapters:
    print(f"Chapter: {ch['chapter']}")
    chunks = chunker.chunk(ch['text'])
```

## Configuration

Set API keys via environment variables or `.env` file:

```bash
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## Supported Providers

| Provider | Default Model |
|----------|---------------|
| Gemini (default) | `gemini-2.0-flash` |
| OpenAI | `gpt-4o` |
| Anthropic | `claude-sonnet-4-20250514` |
| Ollama | `llama3.3` |

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
