# LumberChunker

LLM-powered semantic document segmentation for long-form narrative documents.

[![PyPI version](https://badge.fury.io/py/lumberchunker.svg)](https://badge.fury.io/py/lumberchunker)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LumberChunker is a method leveraging an LLM to dynamically segment documents into semantically independent chunks. It iteratively prompts the LLM to identify the point within a group of sequential passages where the content begins to shift.

Based on the research paper [LumberChunker: Long-Form Narrative Document Segmentation](https://doi.org/10.48550/arXiv.2406.17526) by André V. Duarte, João D.S. Marques, Miguel Graça, Miguel Freire, Lei Li and Arlindo L. Oliveira.

![LumberChunker Pipeline](LumberChunker_pipeline.png)

---

## Installation

```bash
pip install lumberchunker
```

Or install from source:

```bash
git clone https://github.com/LumberChunker/LumberChunker.git
cd LumberChunker
pip install -e .
```

---

## Quick Start

### Using Google Gemini (Default)

```python
from lumberchunker import LumberChunker

chunker = LumberChunker(api_key="your-gemini-api-key")
chunks = chunker.chunk("Your long document text here...")
```

### Using OpenAI

```python
chunker = LumberChunker(provider="openai", api_key="your-openai-key")
chunks = chunker.chunk(text)
```

### Using Anthropic Claude

```python
chunker = LumberChunker(provider="anthropic", api_key="your-anthropic-key")
chunks = chunker.chunk(text)
```

### Using Ollama (Local LLMs)

```python
# No API key needed for local models
chunker = LumberChunker(provider="ollama", model="llama3.3")
chunks = chunker.chunk(text)
```

### Chunking Files

```python
# Directly chunk an EPUB file (automatically converts to text)
chunks = chunker.chunk_file("book.epub")

# Also works with text files
chunks = chunker.chunk_file("document.txt")
```

---

## Configuration

### Environment Variables

Create a `.env` file (see `examples/.env.example`) or set environment variables:

```bash
# API Keys
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Model Selection (optional - defaults shown)
GEMINI_MODEL=gemini-2.0-flash
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OLLAMA_MODEL=llama3.3

# Ollama Host (optional)
OLLAMA_HOST=http://localhost:11434
```

### Specifying Models

```python
# Specify model directly
chunker = LumberChunker(
    provider="openai",
    api_key="your-key",
    model="gpt-4o-mini"
)
```

### Advanced Configuration

```python
from lumberchunker import LumberChunker
from lumberchunker.providers import GeminiProvider

# Create a custom provider
provider = GeminiProvider(
    api_key="your-key",
    model="gemini-2.0-flash",
    temperature=0.1,
)

# Use with custom settings
chunker = LumberChunker(
    provider=provider,
    target_chunk_tokens=550,
)

# Get chunks with metadata
chunks = chunker.chunk(text, return_metadata=True)
for chunk in chunks:
    print(f"Paragraphs {chunk['start_paragraph']}-{chunk['end_paragraph']}")
```

---

## Running the Example

```bash
cd examples
cp .env.example .env
# Edit .env with your API keys
python example.py
```

The example script provides an interactive menu to select a provider and demonstrates chunking with output saved to a file.

---

## Supported Providers

| Provider | Package | Default Model |
|----------|---------|---------------|
| Gemini (default) | `google-genai` | `gemini-2.0-flash` |
| OpenAI | `openai` | `gpt-4o` |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` |
| Ollama | `ollama` | `llama3.3` |

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## Original Research Scripts

The original research implementation is preserved in the `Code/` directory:

```bash
python Code/LumberChunker-Segmentation.py --out_path <output_path> --model_type <Gemini|ChatGPT> --book_name <book_name>
```

---

## GutenQA Dataset

[GutenQA](https://huggingface.co/datasets/LumberChunker/GutenQA) consists of book passages manually extracted from Project Gutenberg and subsequently segmented using LumberChunker.

- 100 Public Domain Narrative Books
- 30 Question-Answer Pairs per Book

### Alternative Chunking Formats

- [Paragraph](https://huggingface.co/datasets/LumberChunker/GutenQA_Paragraphs)
- [Recursive Chunks](https://huggingface.co/datasets/LumberChunker/GutenQA_Recursive)
- [Semantic Chunks](https://huggingface.co/datasets/LumberChunker/GutenQA_Semantic)
- [Propositions](https://huggingface.co/datasets/LumberChunker/GutenQA_Propositions)

---

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

---

## License

MIT License - see [LICENSE](LICENSE) for details.
