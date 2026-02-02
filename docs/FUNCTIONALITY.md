# LumberChunker Functionality Documentation

LumberChunker is an **LLM-powered semantic document segmentation library** that dynamically chunks long-form text into semantically coherent segments using large language models.

---

## Table of Contents

1. [Core Concept](#core-concept)
2. [The Chunking Algorithm](#the-chunking-algorithm)
3. [Main Components](#main-components)
4. [LLM Providers](#llm-providers)
5. [EPUB Support](#epub-support)
6. [Configuration Options](#configuration-options)
7. [API Reference](#api-reference)

---

## Core Concept

Traditional text chunking methods split documents by:
- Fixed character/token counts
- Sentence or paragraph boundaries
- Sliding windows with overlap

**LumberChunker is different.** It uses an LLM to identify where the *content semantically shifts*, producing chunks that are topically coherent rather than arbitrarily split.

### Why This Matters

When chunking for RAG (Retrieval-Augmented Generation) or embedding systems, semantic coherence improves:
- Retrieval accuracy (chunks are self-contained topics)
- Vector embedding quality (less noise from mixed topics)
- Response quality (retrieved chunks are contextually complete)

---

## The Chunking Algorithm

### Step-by-Step Process

```
┌──────────────────────────────────────────────────────────────────┐
│  1. PREPARATION: Text → Paragraphs → ID-labeled paragraphs      │
│     "ID 0: First paragraph..."                                   │
│     "ID 1: Second paragraph..."                                  │
│     "ID 2: Third paragraph..."                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  2. BUILD WINDOW: Accumulate paragraphs until ~550 tokens       │
│     Window = [ID 0, ID 1, ID 2, ID 3, ID 4, ID 5]               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  3. ASK LLM: "Find where content clearly changes"               │
│                                                                  │
│     Possible responses:                                          │
│     • "Answer: ID 4"     → Content shifts at paragraph 4        │
│     • "Answer: NO SPLIT" → Content is cohesive, no split needed │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  4. RECORD BOUNDARY: If shift found, mark as chunk boundary     │
│     Chunk 1 = paragraphs 0-3                                     │
│     Next window starts at paragraph 4                            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  5. REPEAT: Continue until all paragraphs are processed         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  6. MERGE: Combine paragraphs between boundaries into chunks    │
│     Final output: List of semantically coherent chunks          │
└──────────────────────────────────────────────────────────────────┘
```

### The System Prompt

The LLM receives this instruction:

```
You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes 
compared to the previous paragraphs.

Output: 
- If you find a content shift, return the ID of that paragraph: 'Answer: ID XXXX'
- If the content is cohesive with no clear shift, respond with: 'Answer: NO SPLIT'

Important: Only identify a split if there is a genuine content shift. 
Do NOT force a split if the paragraphs discuss the same topic or flow naturally together.
```

### Key Behaviors

| Scenario | Behavior |
|----------|----------|
| Content shifts at ID 7 | LLM responds "Answer: ID 7", paragraphs 0-6 become chunk 1 |
| Content is cohesive | LLM responds "Answer: NO SPLIT", algorithm moves forward without splitting |
| Short content (<3 paragraphs remaining) | Added to final chunk without LLM call |
| LLM response unrecognized | Algorithm moves forward to avoid infinite loops |
| Content triggers safety filter | Returns "content_flag_increment", skips that segment |

---

## Main Components

### LumberChunker Class

The primary class for chunking documents.

```python
from lumberchunker import LumberChunker

chunker = LumberChunker(
    provider="gemini",       # or "openai", "anthropic", "ollama"
    api_key="your-api-key",
    model="gemini-2.0-flash",
    target_chunk_tokens=550,
    temperature=0.1
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `chunk(text)` | Chunk a text string into segments |
| `chunk_file(path)` | Chunk a file (txt or epub) |
| `chunk_paragraphs(paragraphs)` | Chunk a pre-split list of paragraphs |

#### Returns

By default, returns `List[str]` (list of chunk text).

With `return_metadata=True`, returns `List[Dict]`:
```python
{
    "text": "The chunk text...",
    "start_paragraph": 0,
    "end_paragraph": 5,
    "paragraph_count": 5
}
```

---

## LLM Providers

LumberChunker supports multiple LLM backends:

### Gemini (Default)

```python
chunker = LumberChunker(
    provider="gemini",
    api_key="your-gemini-api-key",
    model="gemini-2.0-flash"  # default
)
```

### OpenAI

```python
chunker = LumberChunker(
    provider="openai",
    api_key="your-openai-api-key",
    model="gpt-4o"  # default
)
```

### Anthropic

```python
chunker = LumberChunker(
    provider="anthropic",
    api_key="your-anthropic-api-key",
    model="claude-sonnet-4-20250514"  # default
)
```

### Ollama (Local/Free)

```python
chunker = LumberChunker(
    provider="ollama",
    model="llama3.3"  # default
)
# Note: No API key needed, runs locally
```

### Custom Provider

Extend `BaseLLMProvider` to create your own:

```python
from lumberchunker import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        # Your implementation here
        return response_text

chunker = LumberChunker(provider=MyProvider())
```

### Provider Base Class

All providers inherit from `BaseLLMProvider`:

```python
class BaseLLMProvider:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ):
        ...
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
```

**Built-in retry logic:** Providers automatically retry failed API calls with exponential backoff (default: 3 retries, 60s delay).

---

## EPUB Support

LumberChunker has built-in EPUB support with two approaches:

### Simple: Chunk Entire EPUB

```python
# Treats entire EPUB as one document
chunks = chunker.chunk_file("book.epub")
```

**Behavior:** Extracts all text from the EPUB, then chunks it as a single document. Chapter boundaries are ignored.

### Advanced: Chapter-by-Chapter Extraction

```python
from lumberchunker import epub_to_chapters, LumberChunker

# Step 1: Extract chapters using LLM-powered boundary detection
chapters = epub_to_chapters(
    "book.epub",
    provider="gemini",
    api_key="your-api-key"
)

# Step 2: Chunk each chapter separately
chunker = LumberChunker(provider="gemini", api_key="your-api-key")
for ch in chapters:
    chapter_chunks = chunker.chunk(ch['text'])
    print(f"{ch['chapter']}: {len(chapter_chunks)} chunks")
```

**`epub_to_chapters()` does:**
1. Identifies narrative boundaries (skips front/back matter)
2. Extracts individual chapters with titles
3. Cleans chapter starts to the true opening sentence

**Returns:**
```python
[
    {"chapter": "Chapter 1: The Beginning", "text": "..."},
    {"chapter": "Chapter 2: Rising Action", "text": "..."},
    ...
]
```

### Required Dependencies for EPUB

```bash
pip install ebooklib beautifulsoup4 pydantic
```

---

## Configuration Options

### LumberChunker Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | `"gemini"` | LLM provider: "gemini", "openai", "anthropic", "ollama", or BaseLLMProvider instance |
| `api_key` | `None` | API key (not needed for Ollama) |
| `model` | Provider default | Model name to use |
| `target_chunk_tokens` | `550` | Target token count per analysis window |
| `temperature` | `0.1` | LLM temperature (lower = more deterministic) |

### Tuning Chunk Size

The `target_chunk_tokens` parameter controls how much text the LLM analyzes at once:

```python
# Smaller windows = potentially more chunks
chunker = LumberChunker(target_chunk_tokens=300)

# Larger windows = potentially fewer, larger chunks
chunker = LumberChunker(target_chunk_tokens=800)
```

> **Note:** This is NOT a hard maximum chunk size. It's the window size for LLM analysis. Actual chunk sizes depend on where the LLM identifies semantic shifts.

### Token Estimation

Tokens are estimated as: `tokens ≈ words × 1.2`

This approximation is used for building analysis windows, not for precise tokenization.

---

## API Reference

### Core Functions

#### `LumberChunker.chunk(text, return_metadata=False)`

Chunk a text string into semantically coherent segments.

```python
chunks = chunker.chunk("Your long document text here...")
# Returns: ["Chunk 1 text...", "Chunk 2 text...", ...]

chunks = chunker.chunk(text, return_metadata=True)
# Returns: [{"text": "...", "start_paragraph": 0, ...}, ...]
```

#### `LumberChunker.chunk_file(file_path, return_metadata=False)`

Chunk a file (automatically detects EPUB vs plain text).

```python
chunks = chunker.chunk_file("document.txt")
chunks = chunker.chunk_file("book.epub")
```

#### `LumberChunker.chunk_paragraphs(paragraphs, return_metadata=False)`

Chunk a pre-split list of paragraphs.

```python
paragraphs = ["Paragraph 1...", "Paragraph 2...", ...]
chunks = chunker.chunk_paragraphs(paragraphs)
```

### Utility Functions

#### `epub_to_text(epub_path)`

Convert an EPUB file to plain text.

```python
from lumberchunker import epub_to_text
text = epub_to_text("book.epub")
```

#### `epub_to_chapters(epub_path, provider, api_key, model)`

Extract chapters from an EPUB using LLM-powered boundary detection.

```python
from lumberchunker import epub_to_chapters
chapters = epub_to_chapters("book.epub", provider="openai", api_key="...")
```

#### `read_file_or_epub(file_path)`

Read text from any file, automatically converting EPUB if needed.

```python
from lumberchunker import read_file_or_epub
text = read_file_or_epub("document.txt")  # or .epub
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Unknown provider` | Invalid provider name | Use: "gemini", "openai", "anthropic", "ollama" |
| `ImportError: ebooklib is required` | Missing EPUB dependencies | `pip install ebooklib beautifulsoup4` |
| `FileNotFoundError` | File doesn't exist | Check file path |
| API errors | Rate limits, invalid key, etc. | Providers auto-retry 3 times with 60s delay |

### Content Safety

If an LLM provider blocks content (safety filters), the algorithm:
1. Returns `"content_flag_increment"`
2. Skips that segment
3. Continues with the next window

---

## Example Usage

### Basic Text Chunking

```python
from lumberchunker import LumberChunker

chunker = LumberChunker(provider="gemini", api_key="your-key")

text = """
Your long document goes here. It can be multiple paragraphs.

Each paragraph discusses different topics.

The LLM will identify where the content shifts and create appropriate boundaries.
"""

chunks = chunker.chunk(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk)} characters")
```

### EPUB Book Chunking

```python
from lumberchunker import LumberChunker, epub_to_chapters

# Extract chapters first
chapters = epub_to_chapters("novel.epub", provider="openai", api_key="your-key")

# Chunk each chapter
chunker = LumberChunker(provider="openai", api_key="your-key")
all_chunks = []

for chapter in chapters:
    chapter_chunks = chunker.chunk(chapter['text'])
    all_chunks.extend(chapter_chunks)
    print(f"{chapter['chapter']}: {len(chapter_chunks)} chunks")

print(f"Total: {len(all_chunks)} chunks")
```

### Using Ollama (Free/Local)

```python
from lumberchunker import LumberChunker

# No API key needed
chunker = LumberChunker(provider="ollama", model="llama3.3")
chunks = chunker.chunk_file("document.txt")
```
