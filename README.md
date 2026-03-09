# LumberChunker

**LLM-powered semantic document segmentation for long-form narrative documents.**

[![PyPI version](https://img.shields.io/pypi/v/lumberchunker.svg)](https://pypi.org/project/lumberchunker/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Based on the research paper [LumberChunker: Long-Form Narrative Document Segmentation](https://arxiv.org/abs/2406.17526).

![LumberChunker Pipeline](LumberChunker_pipeline.png)

LumberChunker uses an LLM to iteratively identify where content shifts, producing semantically coherent chunks that are logically independent — ideal for RAG pipelines and narrative analysis.

**Quick links:**  
[Paper](https://doi.org/10.48550/arXiv.2406.17526) | [Blog Post](https://avduarte333.github.io/projects/lumberchunker/) | [GutenQA Dataset](https://huggingface.co/datasets/LumberChunker/GutenQA)

---

## Installation

```bash
pip install lumberchunker
pip install -e .
```

Set your API key via environment variable or a `.env` file:

```bash
export GEMINI_API_KEY="your-key"   # default provider
```

---

## Quick Start

```python
from lumberchunker import LumberChunker

chunker = LumberChunker(api_key="your-gemini-api-key")
chunks = chunker.chunk("Your long document text here...")

# Each chunk is a dict with 'id', 'text', and paragraph range metadata
for chunk in chunks:
    print(chunk["text"][:100])
```

### Saving output (enables stop & resume)

```python
chunks = chunker.chunk("Your text...", output_path="chunks.txt")
```

---

## Configuration

| Provider | Constructor arg | Environment variable | Default model |
| :--- | :--- | :--- | :--- |
| **Gemini** (default) | `provider="gemini"` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| **OpenAI** | `provider="openai"` | `OPENAI_API_KEY` | `gpt-4o` |
| **Anthropic** | `provider="anthropic"` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| **Ollama** | `provider="ollama"` | — | `llama3.3` |

```python
chunker = LumberChunker(provider="openai", api_key="your-key")
chunker = LumberChunker(provider="anthropic", api_key="your-key")
chunker = LumberChunker(provider="ollama", model="llama3.3")  # local, no key needed
```

---

## API Reference

### `LumberChunker(provider, api_key, model, target_chunk_tokens, verbose)`

Creates a chunker instance.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `provider` | `str` | `"gemini"` | LLM provider: `"gemini"`, `"openai"`, `"anthropic"`, `"ollama"` |
| `api_key` | `str \| None` | `None` | API key. Falls back to the relevant environment variable. |
| `model` | `str \| None` | `None` | Model name. Uses provider default if omitted. |
| `target_chunk_tokens` | `int` | `550` | Target token window size per chunking call. |
| `verbose` | `bool` | `True` | Show progress output. |

---

### `chunk(text, output_path, return_metadata)`

Chunk a raw text string.

```python
chunks = chunker.chunk(text)
chunks = chunker.chunk(text, output_path="out.txt")         # incremental save
chunks = chunker.chunk(text, return_metadata=True)          # include paragraph ranges
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | `str` | required | The document text to segment. |
| `output_path` | `str \| None` | `None` | Path to save chunks incrementally. Enables stop & resume. |
| `return_metadata` | `bool` | `False` | Include `start_id`/`end_id` paragraph indices in each chunk dict. |

**Returns:** `list[dict]` — each item has `"id"` and `"text"` keys (plus range keys if `return_metadata=True`).

---

### `chunk_file(path, output_path, return_metadata)`

Chunk a plain-text or EPUB file (EPUB is read as a single string).

```python
chunks = chunker.chunk_file("document.txt")
chunks = chunker.chunk_file("book.epub", output_path="out.txt")
```

---

### `chunk_epub(epub_path, output_dir, chapters, use_llm_extraction, trim_front_back_matter)`

Process an EPUB chapter-by-chapter with automatic output management and checkpoint support.

```python
results = chunker.chunk_epub("book.epub", output_dir="auto")
```

```python
results = chunker.chunk_epub(
    "book.epub",
    output_dir="my_output/book",       # explicit folder
    chapters=[1, 3, 5],                # 1-based indices, or title substrings
    use_llm_extraction=True,           # LLM prefix cleaning
    trim_front_back_matter=True,       # remove copyright/credits pages
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `epub_path` | `str \| Path` | required | Path to the `.epub` file. |
| `output_dir` | `str \| Path` | `"auto"` | Output directory. `"auto"` creates a folder from the filename. |
| `chapters` | `list \| None` | `None` | Chapter filter — integers (1-based index) or title substrings. `None` processes all chapters. |
| `use_llm_extraction` | `bool` | `False` | Use an LLM to identify and trim non-narrative prefixes (e.g. "Chapter IV", baked-in headers). |
| `trim_front_back_matter` | `bool` | `False` | Use an LLM to detect and remove non-narrative chapters (copyright, dedications, etc.). |

**Returns:** `list[dict]` — one entry per chapter with `"chapter"`, `"index"`, and `"chunks"` keys.

---

### `list_chapters(epub_path)`

Inspect EPUB structure without chunking.

```python
chapters = chunker.list_chapters("book.epub")
for ch in chapters:
    print(ch["index"], ch["chapter"], ch["word_count"])
```

**Returns:** `list[dict]` with `"index"`, `"chapter"`, and `"word_count"` keys.

---

### `usage`

Token and call counter, accessible after any chunking operation.

```python
stats = chunker.usage.to_dict()
# {
#   "prompt_count": 42,
#   "input_tokens": 18000,
#   "output_tokens": 1200,
#   "total_tokens": 19200
# }

chunker.usage.reset()  # zero counters between runs
```

---

## CLI

LumberChunker ships a command-line interface:

```bash
# Chunk a plain-text file
lumberchunker document.txt

# Chunk a plain-text file with a specific provider
lumberchunker document.txt --provider anthropic

# Chunk an EPUB, auto-naming the output folder
lumberchunker book.epub

# List chapters without chunking
lumberchunker book.epub --list-chapters

# Chunk specific chapters only
lumberchunker book.epub --chapters 1 3 5

# Full narrative cleaning
lumberchunker book.epub --use-llm-extraction --trim

# Specify output location
lumberchunker book.epub --output my_output/
```

**All options:**

```
positional:
  file                  Path to input file (.txt, .epub, ...)

provider:
  --provider            gemini | openai | anthropic | ollama  (default: gemini)
  --api-key             API key (falls back to env var)
  --model               Model name override

chunking:
  --target-tokens       Target token window size (default: 550)
  -o, --output          Output file (text) or directory (EPUB)

EPUB:
  --chapters            1-based indices or title substrings
  --list-chapters       List chapters and exit
  --use-llm-extraction  LLM-based prefix cleaning
  --trim                Remove front/back matter chapters

general:
  -q, --quiet           Suppress progress output
  --metadata            Include paragraph range metadata in output
```

---

## Stop & Resume

For large books, chunking can take time. LumberChunker automatically writes a `.checkpoint.json` alongside your output. If a run is interrupted, re-run the exact same command and it will pick up where it left off — no duplicate API calls.

```python
# Run once — saves progress to chunks.txt.checkpoint.json
chunks = chunker.chunk(long_text, output_path="chunks.txt")

# Run again after interruption — resumes automatically
chunks = chunker.chunk(long_text, output_path="chunks.txt")
```

---

## Examples

```bash
# Basic text chunking
python examples/example.py

# EPUB chapter-by-chapter chunking
python examples/epub_example.py

# List chapters only
python examples/epub_example.py --list-chapters

# Process specific chapters
python examples/epub_example.py --chapters 1 3 5
```

---

## Citation

```bibtex
@inproceedings{duarte-etal-2024-lumberchunker,
    title = "{L}umber{C}hunker: Long-Form Narrative Document Segmentation",
    author = "Duarte, Andr{\'e} V.  and Marques, Jo{\~a}o DS  and Gra{\c{c}}a, Miguel  and Freire, Miguel  and Li, Lei  and Oliveira, Arlindo L.",
    editor = "Al-Onaizan, Yaser  and Bansal, Mohit  and Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.377/",
    doi = "10.18653/v1/2024.findings-emnlp.377",
    pages = "6473--6486"
}
```

## License

MIT License
