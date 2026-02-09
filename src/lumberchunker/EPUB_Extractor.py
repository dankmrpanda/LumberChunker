#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import html as html_module
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
import warnings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# progress_tracker is only available in the web interface; always None in library mode
ProgressTracker = None

#


# --- Minimal EPUB helpers (no full-text extraction here) ---

def _ensure_deps() -> None:
	try:
		import ebooklib  # noqa: F401
		from ebooklib import epub  # noqa: F401
	except Exception as e:  # pragma: no cover
		raise RuntimeError(
			"Missing dependency 'ebooklib'. Install with: pip install ebooklib"
		) from e


@dataclass
class Section:
	href: str
	title: str
	level: int
	order: int
	parents: list[str]
	# Intentionally no guessed_kind; the LLM decides what to keep/drop.


def _flatten_toc(book) -> list[Section]:
	from ebooklib import epub
	import posixpath

	def _normalize_heading(t: Optional[str]) -> str:
		if not t:
			return ""
		s = str(t)
		s = html_module.unescape(s)
		s = s.strip().strip('"\'')
		s = re.sub(r"\s+", " ", s)
		# Standardize 'Chapter N: Title' when pattern appears at start
		mchap = re.match(r"^(?i)\s*chapter\s+([ivxlcdm]+|\d+)[\s:—-]*([^.!?]{0,120})", s, re.IGNORECASE)
		if mchap:
			num = mchap.group(1)
			tail = (mchap.group(2) or '').strip()
			# Trim tail to a few words and stop at punctuation
			tail = re.split(r"[.!?]", tail, 1)[0]
			words = tail.split()
			if len(words) > 8:
				tail = " ".join(words[:8])
			if tail:
				return f"Chapter {num} - {tail}".strip()
			return f"Chapter {num}"
		# If there's an early sentence terminator, keep only up to that to avoid swallowing body paragraphs
		m = re.search(r"[\.\?!]", s)
		if m and m.start() > 20:
			s = s[:m.start()].strip()
		# If still very long, cap to first 12 words
		words = s.split()
		if len(words) > 12:
			s = " ".join(words[:12]).rstrip(",;:.-—")
		# Clean leftover surrounding quotes again
		s = s.strip().strip('"\'')
		return s

	def _is_container_title(s: Optional[str]) -> bool:
		if not s:
			return False
		sl = str(s).strip().lower()
		excluded = (
			'content', 'table of contents', 'toc', 'chapter', 'prologue', 'epilogue',
			'appendix', 'foreword', 'preface', 'acknowledgement', 'acknowledgment',
			'copyright', 'about', 'maps', 'index', 'glossary', 'bibliography', 'notes',
			'dedication', 'cover', 'title page'
		)
		if any(x in sl for x in excluded):
			return False
		return re.search(r'\b(book|part|volume)\b', sl) is not None

	def _extract_first_heading(html_bytes: bytes) -> Optional[str]:
		"""Extract the first H1/H2/H3 or <title> text from an HTML document.

		Keeps this very lightweight to avoid extra dependencies.
		"""
		try:
			html = html_bytes.decode("utf-8", errors="ignore")
		except Exception:
			return None
		# Prefer h1, then h2, then h3
		for tag in ("h1", "h2", "h3"):
			m = re.search(rf"<\s*{tag}[^>]*>(.*?)<\s*/\s*{tag}\s*>", html, re.IGNORECASE | re.DOTALL)
			if m:
				raw = m.group(1)
				# Strip inner tags
				text = re.sub(r"<[^>]+>", " ", raw)
				text = html_module.unescape(text)
				text = re.sub(r"\s+", " ", text).strip()
				if text:
					return text
		# Fallback: HTML <title>
		m = re.search(r"<\s*title[^>]*>(.*?)<\s*/\s*title\s*>", html, re.IGNORECASE | re.DOTALL)
		if m:
			raw = m.group(1)
			text = re.sub(r"<[^>]+>", " ", raw)
			text = html_module.unescape(text)
			text = re.sub(r"\s+", " ", text).strip()
			if text:
				return text
		return None

	def _extract_headings(html_bytes: bytes) -> list[str]:
		"""Extract ordered candidate headings: h1-3, then p/div with chapter-like markers."""
		out: list[str] = []
		try:
			html = html_bytes.decode("utf-8", errors="ignore")
		except Exception:
			return out
		# 1) h1/h2/h3 first
		for tag in ("h1", "h2", "h3"):
			for m in re.finditer(rf"<\s*{tag}[^>]*>(.*?)<\s*/\s*{tag}\s*>", html, re.IGNORECASE | re.DOTALL):
				raw = m.group(1)
				text = re.sub(r"<[^>]+>", " ", raw)
				text = html_module.unescape(text)
				text = re.sub(r"\s+", " ", text).strip()
				text = _normalize_heading(text)
				if text:
					out.append(text)
		# 2) p/div with id/class containing 'chapter' or 'chap'
		for tag in ("p", "div"):
			for m in re.finditer(rf"<\s*{tag}[^>]*(id|class)\s*=\s*\"[^\"]*(chap|chapter)[^\"]*\"[^>]*>(.*?)<\s*/\s*{tag}\s*>", html, re.IGNORECASE | re.DOTALL):
				raw = m.group(3)
				text = re.sub(r"<[^>]+>", " ", raw)
				text = html_module.unescape(text)
				text = re.sub(r"\s+", " ", text).strip()
				text = _normalize_heading(text)
				if text:
					out.append(text)
		# 3) Any p/div whose text itself looks like a chapter marker at the start (fallback)
		for tag in ("p", "div"):
			for m in re.finditer(rf"<\s*{tag}[^>]*>(.*?)<\s*/\s*{tag}\s*>", html, re.IGNORECASE | re.DOTALL):
				raw = m.group(1)
				text = re.sub(r"<[^>]+>", " ", raw)
				text = html_module.unescape(text)
				text = re.sub(r"\s+", " ", text).strip()
				if not text:
					continue
				# Require 'chapter' to be at the start (optionally after quotes/spaces) to avoid matching random body sentences
				if re.match(r"^\s*[\"'\(\[]?\s*chapter\b", text, re.IGNORECASE) or re.fullmatch(r"(?i)(prologue|epilogue)", text) or re.fullmatch(r"(?i)([IVXLCDM]+|\d+)(?:\b|\.)", text):
					out.append(_normalize_heading(text))
		return out

	def iter_nodes(
		nodes: Iterable[Any], level: int, order_start: int, parent_titles: list[str]
	) -> tuple[list[Section], int]:
		results: list[Section] = []
		order = order_start
		for n in nodes:
			# ebooklib TOC entries can be:
			# - epub.Link(title, href)
			# - tuple(epub.Link, [children])
			# - tuple(epub.Section, [children])
			# - epub.Section(title, [children]) where section has .title and .subitems
			title = None
			href = None
			children = None
			if isinstance(n, epub.Link):
				title = getattr(n, "title", None)
				href = getattr(n, "href", None)
			elif isinstance(n, tuple) and n and isinstance(n[0], epub.Link):
				link = n[0]
				title = getattr(link, "title", None)
				href = getattr(link, "href", None)
				children = n[1] if len(n) > 1 else None
			elif isinstance(n, tuple) and n and hasattr(n[0], "title"):
				# Handle tuple(epub.Section, [children])
				section = n[0]
				title = getattr(section, "title", None)
				href = getattr(section, "href", None)  # Some sections might have hrefs
				children = n[1] if len(n) > 1 else None
			elif hasattr(n, "title") and hasattr(n, "subitems"):
				title = getattr(n, "title", None)
				href = None  # Section header may not have an href itself
				children = list(getattr(n, "subitems", []) or [])
			if href:
				# Only add leaf nodes: if this entry has children, skip it and let its deepest descendants represent content
				if not children:
					results.append(
						Section(
							href=str(href),
							title=str(title or "").strip(),
							level=level,
							order=order,
							parents=list(parent_titles),
						)
					)
					order += 1
			if children:
				# If this node has a title but no href (or even if it has one and acts as a parent),
				# include it in the ancestry for children.
				next_parents = list(parent_titles)
				if title:
					t = str(title or "").strip()
					if t:
						next_parents.append(t)
				child_list, order = iter_nodes(children, level + 1, order, next_parents)
				results.extend(child_list)
		return results, order

	toc = getattr(book, "toc", [])
	flat, _ = iter_nodes(toc, level=0, order_start=1, parent_titles=[])
	# Fallback: if TOC is empty, list documents from spine order, using first heading as title
	if not flat:
		try:
			from ebooklib import ITEM_DOCUMENT  # type: ignore
			doc_type = ITEM_DOCUMENT
		except Exception:
			doc_type = 1  # Fallback to numeric constant
		for i, item in enumerate(book.get_items_of_type(doc_type), start=1):
			href = getattr(item, "href", None)
			if not href:
				continue
			title = None
			try:
				content = item.get_content()
				title = _extract_first_heading(content)
			except Exception:
				title = None
			if not title:
				# Last resort: use item name
				title = str(getattr(item, "get_name", lambda: "")() or "").strip()
			flat.append(
				Section(
					href=str(href),
					title=str(title or "").strip(),
					level=0,
					order=i,
					parents=[],
				)
			)
	# Enrich titles when TOC provides empty/placeholder or filename-like titles by peeking into the document
	# This does not add new sections; it only improves the title string for the LLM.
	def _looks_like_filename(t: Optional[str]) -> bool:
		if not t:
			return True
		t = str(t).strip()
		if re.search(r"\.(x?html?|htm)\b", t, re.IGNORECASE):
			return True
		if ('/' in t or '\\' in t) and ' ' not in t:
			return True
		return False
	for s in flat:
		if not s.title or not s.title.strip() or _looks_like_filename(s.title):
			try:
				item = book.get_item_with_href(s.href)
			except Exception:
				item = None
			if item is not None:
				try:
					headings = _extract_headings(item.get_content())
				except Exception:
					headings = []
				# Prefer chapter-like headings; else first heading; else keep original
				cand = None
				for h in headings:
					ch = h.strip().strip('"\'')
					if re.search(r"\bchapter\b", ch, re.IGNORECASE):
						cand = ch
						break
					if re.search(r"\b(prologue|epilogue)\b", ch, re.IGNORECASE) or re.search(r"^(?:[IVXLCDM]+|\d+)(?:\b|\.)", ch):
						cand = ch
						break
				if not cand and headings:
					cand = headings[0].strip().strip('"\'')
				if cand:
					s.title = _normalize_heading(cand)

	# Try to enrich parents and discover missing sections using the actual TOC HTML if available
	def _parse_toc_html() -> tuple[dict[str, list[str]], dict[str, str]]:
		def _normalize_href(raw: str, base_href: Optional[str]) -> str:
			# Resolve relative hrefs against the TOC file path and normalize
			raw = (raw or '').strip()
			if base_href:
				base_dir = posixpath.dirname(base_href)
				joined = posixpath.normpath(posixpath.join(base_dir, raw))
			else:
				joined = posixpath.normpath(raw)
			# Remove leading './'
			if joined.startswith('./'):
				joined = joined[2:]
			return joined
		# Find a candidate TOC entry in our flat list
		candidate = None
		for sec in flat:
			t = (sec.title or '').lower()
			if 'content' in t or 'table of contents' in t or 'contents' in t or 'toc' in (sec.href or '').lower():
				candidate = sec
				break
		if not candidate:
			return {}, {}
		try:
			item = book.get_item_with_href(candidate.href)
			if not item:
				return {}, {}
			html = item.get_content().decode('utf-8', errors='ignore')
		except Exception:
			return {}, {}
		# Collect headings and links with their positions
		heading_iter = list(re.finditer(r'<\s*h([1-4])[^>]*>(.*?)<\s*/\s*h\1\s*>', html, re.IGNORECASE | re.DOTALL))
		link_iter = list(re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL))
		# Build a simple list of (pos, text, kind)
		items: list[tuple[int, str, str]] = []
		for m in heading_iter:
			text = re.sub(r'<[^>]+>', ' ', m.group(2))
			text = html_module.unescape(text)
			text = re.sub(r'\s+', ' ', text).strip()
			items.append((m.start(), text, 'heading'))
		for m in link_iter:
			href = m.group(1)
			text = re.sub(r'<[^>]+>', ' ', m.group(2))
			text = html_module.unescape(text)
			text = re.sub(r'\s+', ' ', text).strip()
			# Pack href and text together in the string, we'll split later
			items.append((m.start(), href + '\u0000' + text, 'link'))
		items.sort(key=lambda x: x[0])
		parents_by_href: dict[str, list[str]] = {}
		titles_by_href: dict[str, str] = {}
		current_parent: Optional[str] = None
		def is_container_heading(s: str) -> bool:
			if not s:
				return False
			sl = s.lower()
			# Exclude obviously non-container terms
			excluded = (
				'content', 'table of contents', 'toc', 'chapter', 'prologue', 'epilogue',
				'appendix', 'foreword', 'preface', 'acknowledgement', 'acknowledgment',
				'copyright', 'about', 'maps', 'index', 'glossary', 'bibliography', 'notes'
			)
			if any(x in sl for x in excluded):
				return False
			# Treat lines containing whole words 'book', 'part' or 'volume' as containers (not substrings like 'party')
			if re.search(r'\b(book|part|volume)\b', sl):
				return True
			return False
		for _, val, kind in items:
			if kind == 'heading':
				if is_container_heading(val):
					current_parent = val.strip()
				else:
					# Non-container heading resets nothing
					pass
			else:  # link
				href, link_text = val.split('\u0000', 1)
				href = href.split('#', 1)[0]
				href_norm = _normalize_href(href, candidate.href)
				link_text = (link_text or '').strip()
				# If link itself looks like container, start new scope and do not assign parent to it
				if is_container_heading(link_text):
					current_parent = link_text
				else:
					if current_parent:
						parents_by_href.setdefault(href_norm, []).append(current_parent)
				if link_text:
					titles_by_href[href_norm] = link_text
		return parents_by_href, titles_by_href

	# Apply inferred parents if section has no parents yet
	try:
		parents_map, titles_map = _parse_toc_html()
		if parents_map or titles_map:
			# Build a seen set of normalized hrefs using no base (canonical hrefs in flat are already manifest-relative)
			def _canon(h: str) -> str:
				# Normalize like above but without a base, and decode URL encoding
				import urllib.parse
				normalized = (h or '').split('#', 1)[0].lstrip('./')
				return urllib.parse.unquote(normalized)
			seen_hrefs = {_canon(s.href) for s in flat}
			max_order = max((s.order for s in flat), default=0)
			# First enrich existing
			for s in flat:
				href_norm = _canon(s.href)
				if not s.parents and href_norm in parents_map:
					s.parents = parents_map[href_norm]
				if (not s.title or not s.title.strip()) and href_norm in titles_map:
					t = titles_map[href_norm].strip()
					if t:
						s.title = t
			# Add missing sections discovered in TOC HTML
			for href_norm, title in titles_map.items():
				if href_norm in seen_hrefs:
					continue
				# Avoid adding container-only entries from TOC HTML
				if _is_container_title(title):
					continue
				max_order += 1
				flat.append(
					Section(
						href=href_norm,
						title=title.strip(),
						level=0,
						order=max_order,
						parents=parents_map.get(href_norm, []),
					)
				)
	except Exception:
		# Best-effort enrichment only
		pass

	# NOTE: We trust the TOC and do not supplement with additional spine documents.
	# Chapters should never be split across multiple files - we work with what the TOC provides.
	# This prevents adding chapter continuation files as separate sections.

	# Second pass: if any remaining filename-like titles, try again with broader heading candidates
	for s in flat:
		try:
			if s.title and not _looks_like_filename(s.title):
				continue
			item = book.get_item_with_href(s.href)
			if not item:
				continue
			headings = _extract_headings(item.get_content())
			cand = None
			for h in headings:
				ch = h.strip().strip('"\'')
				if re.search(r"\bchapter\b", ch, re.IGNORECASE):
					cand = ch
					break
				if re.search(r"\b(prologue|epilogue)\b", ch, re.IGNORECASE) or re.search(r"^(?:[IVXLCDM]+|\d+)(?:\b|\.)", ch):
					cand = ch
					break
			if not cand and headings:
				cand = headings[0].strip().strip('"\'')
			if cand:
				s.title = _normalize_heading(cand)
		except Exception:
			pass
	# Post-process: drop container-titled leaves and synthesize chapter numbers for filename-like titles under containers
	def _looks_like_filename2(t: Optional[str]) -> bool:
		if not t:
			return True
		s = str(t).strip()
		return bool(re.search(r"\.(x?html?|htm)\b", s, re.IGNORECASE) or (("/" in s or "\\" in s) and " " not in s))
	# Remove items whose title itself is container-like
	flat = [s for s in flat if not _is_container_title(s.title)]
	# Assign global Chapter N for filename-like items under containers when no better title is present
	# We count only true chapter-like items in sequence; prologue/epilogue are not counted.
	chapter_idx = 0
	for s in sorted(flat, key=lambda x: x.order):
		# Does this section live under a container parent (Book/Part/Volume)?
		under_container = any(_is_container_title(p) for p in (s.parents or []))
		if not under_container:
			continue
		# If it's explicitly prologue/epilogue, do not count or rename
		if re.search(r"\b(prologue|epilogue)\b", s.title or "", re.IGNORECASE):
			continue
		# If title already looks like a chapter marker, increment counter based on appearance (numeric only) else sequentially
		m = re.search(r"\bchapter\s+(\d+)\b", s.title or "", re.IGNORECASE)
		if m:
			try:
				chapter_idx = max(chapter_idx, int(m.group(1)))
				continue
			except Exception:
				pass
		# Else, if title is filename-like or empty, assign Chapter N
		if _looks_like_filename2(s.title) or not (s.title or '').strip():
			chapter_idx += 1
			s.title = f"Chapter {chapter_idx}"

	# Deduplicate by href while preserving order
	import urllib.parse
	seen = set()
	unique: list[Section] = []
	for s in flat:
		# Normalize href for deduplication: keep fragments, but remove leading ./ and decode URL encoding
		# Fragments are important for EPUBs that use anchor-based navigation within files
		key = s.href.lstrip('./')
		key_decoded = urllib.parse.unquote(key)
		# Use the decoded version as the canonical key for deduplication
		if key_decoded not in seen:
			unique.append(s)
			seen.add(key_decoded)
	return unique


def _read_sections(epub_path: str) -> list[Section]:
	_ensure_deps()
	from ebooklib import epub

	book = epub.read_epub(epub_path)
	sections = _flatten_toc(book)
	return sections
# --- Deterministic boundary detection (replaces LLM planning) ---
class KeptSection(BaseModel):
	href: str
	title: str
	full_title: str
	order: int


# --- Multi-provider LLM helpers ---
# Global provider instance for EPUB extraction (set by epub_to_chapters)
_extraction_provider = None


def _get_extraction_provider():
	"""Get the current extraction provider, or create a default OpenAI provider."""
	global _extraction_provider
	if _extraction_provider is not None:
		return _extraction_provider
	
	# Default fallback to OpenAI for backward compatibility
	try:
		from lumberchunker.providers.openai import OpenAIProvider
		return OpenAIProvider(temperature=0.0)
	except Exception:
		raise RuntimeError(
			"No LLM provider configured for EPUB extraction. "
			"Either pass a provider to epub_to_chapters() or set OPENAI_API_KEY."
		)


def _set_extraction_provider(provider):
	"""Set the provider to use for EPUB extraction."""
	global _extraction_provider
	_extraction_provider = provider


def _get_provider_usage_dict() -> dict:
	"""Return the extraction provider's usage stats as a dict, or empty dict."""
	try:
		prov = _get_extraction_provider()
		return prov.usage.to_dict()
	except Exception:
		return {}


class NarrativeCheckResult(BaseModel):
	"""Result of checking if a section is narrative content."""
	href: str
	is_narrative: bool
	reason: Optional[str] = None


NARRATIVE_CHECK_PROMPT = (
	"You are classifying if a section belongs to the main narrative of a novel or is auxiliary front/back matter.\n"
	"Decide based on the title and the first words of the content.\n"
	"Return JSON: {\"href\": string, \"is_narrative\": true|false, \"reason\": short string}.\n"
	"Auxiliary examples: license pages, copyright notices, tables of contents, forewords, acknowledgements, about-the-author, adverts, previews.\n"
	"Narrative examples: prose that starts a story scene, chapter opening, dialogue, character actions, prologue, epilogue.\n"
)


def _call_llm_json(system: str, user: str, response_model: type[BaseModel]) -> BaseModel:
	"""Call the LLM provider and parse JSON response into a Pydantic model.
	
	Works with any LumberChunker provider (Gemini, OpenAI, Anthropic, Ollama).
	"""
	provider = _get_extraction_provider()
	
	# Combine system and user prompts
	full_prompt = f"{system}\n\n{user}"
	
	# Call the provider
	try:
		response_text = provider.generate(full_prompt)
	except Exception as e:
		raise RuntimeError(f"LLM call failed: {e}")

	# Log per-call token usage
	u = provider.usage
	logger.debug(f"[TOKENS] in={u.last_input_tokens}  out={u.last_output_tokens}  "
	             f"(cumulative: {u.prompt_count} calls, {u.total_tokens} total tokens)")
	
	# Parse JSON from response
	# Try to extract JSON from the response (handle markdown code blocks)
	text = response_text.strip()
	
	# Handle markdown code blocks
	if "```json" in text:
		start = text.find("```json") + 7
		end = text.find("```", start)
		if end > start:
			text = text[start:end].strip()
	elif "```" in text:
		start = text.find("```") + 3
		end = text.find("```", start)
		if end > start:
			text = text[start:end].strip()
	
	# Find JSON object boundaries
	start_brace = text.find("{")
	end_brace = text.rfind("}")
	if start_brace != -1 and end_brace > start_brace:
		text = text[start_brace:end_brace + 1]
	
	try:
		return response_model.model_validate_json(text)
	except Exception as e:
		raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}")


def _get_first_50_tokens(text: str) -> str:
	"""Get the first 50 tokens (words) from text."""
	tokens = (text or "").split()
	return " ".join(tokens[:50])


# --- Opening sentence detection (Structured Output) ---
class OpeningSentenceResult(BaseModel):
	href: str
	opening_sentence: str
	reliable: Optional[bool] = None
	reason: Optional[str] = None


OPENING_SENTENCE_PROMPT = (
	"You are given the start of a book chapter as plain text.\n"
	"Return exactly the first complete narrative sentence as it appears in the provided TEXT, "
	"character-for-character (including punctuation, quotes, dashes, capitalization, and spaces/newlines).\n"
	"Keep in mind that the first narrative sentence should NOT include headings, roman numerals, chapter numbers, epigraphs, page headers/footers,"
	"standalone words (e.g., 'One', 'I'), or other front-matter artifacts.\n"
	"We are looking for the starting point of the narrative content.\n"
	"Output JSON: {\"href\": string, \"opening_sentence\": string, \"reliable\": boolean, \"reason\": short string}.\n"
	"Only use the provided TEXT; do not infer or rewrite."
)


def _regex_from_sentence_whitespace_flexible(sentence: str) -> str:
	"""Build a regex that matches the sentence with flexible whitespace (\\s+ for any runs)."""
	# Split on any whitespace and escape each token
	tokens = re.split(r"\s+", sentence.strip())
	tokens = [re.escape(t) for t in tokens if t]
	if not tokens:
		return ""
	return r"\s+".join(tokens)


def _normalize_with_mapping(s: str, *, keep_punct: bool) -> tuple[str, list[int]]:
	"""Normalize unicode/whitespace and build index mapping to original.

	- Lowercase
	- Normalize unicode quotes/dashes/ellipsis
	- Collapse whitespace to single spaces
	- Optionally remove punctuation (all unicode categories starting with 'P')

	Returns (normalized_string, norm_index_to_orig_index_mapping)
	"""
	import unicodedata as _ud

	def _map_char(ch: str) -> str:
		# normalize quotes
		quotes = {
			"“": '"', "”": '"', "„": '"', "‟": '"',
			"‘": "'", "’": "'", "‚": "'", "‛": "'",
		}
		if ch in quotes:
			return quotes[ch]
		# normalize dashes
		if ch in {"—", "–", "‒", "―"}:
			return "-"
		# normalize ellipsis
		if ch == "…":
			return "..."
		# zero-width & control-like
		if ch in {"\u200b", "\u200c", "\u200d", "\ufeff"}:
			return ""
		# normalize whitespace to space
		if ch.isspace():
			return " "
		return ch.lower()

	out_chars: list[str] = []
	mapping: list[int] = []
	last_was_space = False
	i = 0
	while i < len(s):
		ch = s[i]
		rep = _map_char(ch)
		# If replacement has multiple chars (like ellipsis -> ...), map each to current index
		for r in rep:
			# optionally drop punctuation
			if not keep_punct:
				cat = _ud.category(r)
				if cat and cat.startswith("P"):
					continue
			if r == " ":
				if last_was_space:
					continue
				last_was_space = True
			else:
				last_was_space = False
			out_chars.append(r)
			mapping.append(i)
		i += 1
	# trim leading/trailing spaces in normalized and fix mapping accordingly
	norm = "".join(out_chars).strip()
	if not norm:
		return "", []
	# Find start index in out_chars of norm
	# Compute leading spaces count removed
	lead = 0
	while lead < len(out_chars) and out_chars[lead] == " ":
		lead += 1
	# Compute trailing spaces count removed (not needed for mapping start)
	return norm, mapping[lead: lead + len(norm)]


def _find_sentence_start(text: str, sentence: str) -> Optional[int]:
	"""Find the start index of sentence within text using robust multi-pass matching.

	Order:
	1) Exact substring
	2) Flexible whitespace regex (case-sensitive)
	3) Flexible whitespace regex (case-insensitive)
	4) Normalized find (keep punctuation)
	5) Normalized find (drop punctuation)
	6) Difflib longest-match on normalized (keep punctuation)
	7) Difflib longest-match on punctuation-stripped
	"""
	if not text or not sentence:
		return None

	# 1) Exact match
	pos = text.find(sentence)
	if pos != -1:
		return pos

	# 2-3) Flexible whitespace regex
	pattern = _regex_from_sentence_whitespace_flexible(sentence)
	if pattern:
		m = re.search(pattern, text)
		if m:
			return m.start()
		m = re.search(pattern, text, re.IGNORECASE)
		if m:
			return m.start()

	# 4) Normalized (keep punctuation)
	norm_text, map_text = _normalize_with_mapping(text, keep_punct=True)
	norm_sent, _ = _normalize_with_mapping(sentence, keep_punct=True)
	if norm_text and norm_sent:
		idx = norm_text.find(norm_sent)
		if idx != -1 and idx < len(map_text):
			return map_text[idx]

	# 5) Normalized (drop punctuation)
	norm_text2, map_text2 = _normalize_with_mapping(text, keep_punct=False)
	norm_sent2, _ = _normalize_with_mapping(sentence, keep_punct=False)
	if norm_text2 and norm_sent2:
		idx2 = norm_text2.find(norm_sent2)
		if idx2 != -1 and idx2 < len(map_text2):
			return map_text2[idx2]

	# 6-7) Difflib longest match on normalized variants as a last resort
	try:
		from difflib import SequenceMatcher  # stdlib
	except Exception:
		SequenceMatcher = None  # type: ignore

	if SequenceMatcher is not None:
		# keep punctuation variant
		if norm_text and norm_sent:
			m = SequenceMatcher(None, norm_sent, norm_text).find_longest_match(0, len(norm_sent), 0, len(norm_text))
			# Require decent coverage of the sentence start
			if m and m.size >= max(10, int(0.7 * len(norm_sent))):
				b = m.b  # start in norm_text
				if b < len(map_text):
					return map_text[b]
		# drop punctuation variant (stricter threshold as it's looser text)
		if norm_text2 and norm_sent2:
			m2 = SequenceMatcher(None, norm_sent2, norm_text2).find_longest_match(0, len(norm_sent2), 0, len(norm_text2))
			if m2 and m2.size >= max(10, int(0.8 * len(norm_sent2))):
				b2 = m2.b
				if b2 < len(map_text2):
					return map_text2[b2]

	return None


def _detect_opening_sentence_for_record(rec: dict, model: str) -> str:
	"""Use LLM Structured Outputs to get the exact opening sentence for a record's text."""
	href = str(rec.get("href", ""))
	title = str(rec.get("full_title") or rec.get("title") or "").strip()
	text = rec.get("text") or ""
	# Provide a generous excerpt so the first sentence is complete
	excerpt = text[:3000]
	if not excerpt.strip():
		return ""
	system = OPENING_SENTENCE_PROMPT
	user = (
		f"Href: {href}\n"
		f"Title: {title}\n\n"
		f"TEXT BEGIN\n{excerpt}\nTEXT END\n"
	)
	try:
		res: OpeningSentenceResult = _call_llm_json(system, user, OpeningSentenceResult)
		s = (res.opening_sentence or "").strip("\n")
		return s
	except Exception:
		# Do not fallback heuristically; empty means no alignment performed
		return ""


def _clean_records_with_llm(records: List[dict], model: str, tracker=None) -> List[dict]:
	"""Align each record's text to start at the true opening sentence.

	Adds 'opening_sentence' to each record and trims 'text' to begin at that sentence.
	Uses progress tracker for web interface.
	"""
	out: List[dict] = []
	total_records = len(records)
	
	logger.debug("START_CLEANING=TRUE")  # Flag for web interface
	sys.stdout.flush()
	if tracker:
		tracker.set_chapter_cleaning_progress(0, total_records)
	time.sleep(0.5)  # Give web interface time to process
	
	logger.info(f"Aligning chapter starts: 0/{total_records} (0%)")
	sys.stdout.flush()
	
	for i, rec in enumerate(records):
		# Report progress before processing each chapter
		completed = i
		progress_pct = int((completed / total_records) * 100) if total_records > 0 else 0
		progress_line = f"Aligning chapter starts: {completed}/{total_records} ({progress_pct}%)"
		logger.info(progress_line)
		sys.stdout.flush()
		
		if tracker:
			tracker.set_chapter_cleaning_progress(completed, total_records)
		
		text = rec.get("text") or ""
		opening = _detect_opening_sentence_for_record(rec, model)
		
		# Add delay after LLM call to ensure UI can update
		time.sleep(0.5)  # Half second delay to ensure synchronous updates
		
		start_idx = _find_sentence_start(text, opening) if opening else None
		new_text = text
		if start_idx is not None:
			new_text = text[start_idx:]
			# Clean leading blank lines
			new_text = re.sub(r"^\s*\n+", "", new_text)
		new_rec = dict(rec)
		new_rec["opening_sentence"] = opening or ""
		new_rec["text"] = new_text
		out.append(new_rec)
		
		# Report progress after completing each chapter
		completed = i + 1
		progress_pct = int((completed / total_records) * 100)
		progress_line = f"Aligning chapter starts: {completed}/{total_records} ({progress_pct}%)"
		logger.info(progress_line)
		sys.stdout.flush()
		
		if tracker:
			tracker.set_chapter_cleaning_progress(completed, total_records)
		
		time.sleep(0.2)  # Additional delay to allow UI to update progressively
	
	logger.debug("END_CLEANING=TRUE")  # Flag for web interface
	sys.stdout.flush()
	if tracker:
		tracker.set_chapter_cleaning_progress(total_records, total_records, done=True)

	# Log cleaning usage summary
	try:
		prov = _get_extraction_provider()
		u = prov.usage
		logger.info(f"Chapter cleaning done — {total_records} chapters, "
		            f"{u.prompt_count} LLM calls, "
		            f"{u.total_input_tokens:,} in / {u.total_output_tokens:,} out / "
		            f"{u.total_tokens:,} total tokens")
	except Exception:
		pass

	return out


def _preprocess_records_collapse_newlines(records: List[dict]) -> List[dict]:
	"""Normalize newlines and trailing whitespace in each chapter's text.

	- Normalize CRLF/CR to LF.
	- Strip trailing spaces/tabs at line ends.
	- Collapse runs of 3+ newlines to a double newline (preserving paragraph boundaries).
	- Trim leading/trailing whitespace of the whole text.
	"""
	out: List[dict] = []
	for rec in records:
		new_rec = dict(rec)
		t = (new_rec.get("text") or "")
		if t:
			t = t.replace("\r\n", "\n").replace("\r", "\n")
			t = "\n".join(ln.rstrip() for ln in t.split("\n"))
			t = re.sub(r"\n{3,}", "\n\n", t)
			t = t.strip()
			new_rec["text"] = t
		out.append(new_rec)
	return out


def _extract_fragment_content(html: str, fragment_id: str) -> str:
	"""Extract content from a specific fragment/anchor to the next section."""
	if not fragment_id:
		return _html_to_text(html)
	
	# Look for the anchor with the fragment ID
	anchor_pattern = rf'<[^>]*(?:id|name)=["\']?{re.escape(fragment_id)}["\']?[^>]*>'
	match = re.search(anchor_pattern, html, re.IGNORECASE)
	
	if not match:
		# If we can't find the fragment, return the full content
		return _html_to_text(html)
	
	start_pos = match.start()
	
	# Find the next anchor/section to determine where this section ends
	# Look for subsequent anchors with similar ID patterns
	rest_html = html[start_pos:]
	
	# Try to find the next chapter/section marker
	next_patterns = [
		r'<[^>]*(?:id|name)=["\']?pgepubid\d+["\']?[^>]*>',  # Next pgepubid anchor
		r'<h[1-6][^>]*>.*?CHAPTER\s+[IVXLCDM]+',  # Next chapter heading
		r'<h[1-6][^>]*>.*?VOL\.\s+[IVXLCDM]+',  # Volume marker
	]
	
	end_pos = len(rest_html)
	for pattern in next_patterns:
		matches = list(re.finditer(pattern, rest_html, re.IGNORECASE | re.DOTALL))
		# Skip the first match if it's at position 0 (the current section)
		for match in matches:
			if match.start() > 10:  # Reduced buffer - just need to skip the current anchor
				end_pos = match.start()
				break
		if end_pos < len(rest_html):
			break
	
	# Extract the section content
	section_html = rest_html[:end_pos]
	return _html_to_text(section_html)


async def _is_valid_narrative_section(section: Section, epub_path: str, model: str) -> tuple[bool, int]:
	"""Check if a section is narrative using LLM. Returns (is_narrative, tokens_used).
	
	Note: This function assumes the section already has sufficient tokens (pre-filtered).
	"""
	# Get the text content for this section
	_ensure_deps()
	from ebooklib import epub as _epub
	import urllib.parse
	
	book = _epub.read_epub(epub_path)
	
	# Parse href to separate file path and fragment
	href_parts = (section.href or "").split('#', 1)
	href_file = href_parts[0].lstrip('./')
	fragment_id = href_parts[1] if len(href_parts) > 1 else None
	href_decoded = urllib.parse.unquote(href_file)
	
	text = ""
	try:
		# Try with decoded href first
		item = book.get_item_with_href(href_decoded)
		if item is None:
			# Fallback to original href
			item = book.get_item_with_href(href_file)
		if item is not None:
			html = item.get_content().decode("utf-8", errors="ignore")
			# Use the same fragment extraction logic as in _extract_texts_for_sections
			text = _extract_fragment_content(html, fragment_id)
	except Exception:
		text = ""
	
	# Get first 50 tokens for LLM analysis
	opening = _get_first_50_tokens(text)
	
	# If empty opening, likely auxiliary (though this should be rare after pre-filtering)
	if not opening.strip():
		return False, 0
	
	# Build full title with parents
	def _full_title(s: Section) -> str:
		parts = [pt for pt in (s.parents or []) if pt]
		if s.title:
			parts.append(s.title)
		return " — ".join(parts) if parts else (s.title or "")
	
	full_title = _full_title(section)
	
	system = NARRATIVE_CHECK_PROMPT
	user = (
		f"Href: {section.href}\n"
		f"Title: {full_title}\n"
		f"Opening text: {opening}"
	)
	
	tokens_used = _token_count(system, model) + _token_count(user, model)
	
	try:
		res_obj: NarrativeCheckResult = _call_llm_json(system, user, NarrativeCheckResult)
		# Use actual tokens from provider if available, else fall back to estimate
		try:
			prov = _get_extraction_provider()
			actual = prov.usage.last_input_tokens + prov.usage.last_output_tokens
			if actual > 0:
				tokens_used = actual
		except Exception:
			pass
		return res_obj.is_narrative, tokens_used
	except Exception:
		# Fallback: conservative approach - assume it's narrative if we can't determine
		return True, tokens_used


async def run_boundary_detection(epub_path: str, model: str, tracker=None) -> dict:
	"""Find narrative boundaries using LLM-based top-to-bottom and bottom-to-top search."""
	sections = _read_sections(epub_path)
	
	if not sections:
		return {"kept_sections": [], "dropped": [], "error": "No sections found"}
	
	# Sort sections by order
	sorted_sections = sorted(sections, key=lambda x: x.order)
	
	logger.info(f"Found {len(sorted_sections)} sections total")
	
	if tracker:
		tracker.set_top_boundary_progress(0)
	
	# Pre-filter: Remove all sections with less than 100 tokens
	logger.info("Pre-filtering short sections...")
	substantial_sections = []
	dropped_short = []
	
	for section in sorted_sections:
		# Get text content and check token count
		_ensure_deps()
		from ebooklib import epub as _epub
		import urllib.parse
		
		book = _epub.read_epub(epub_path)
		
		# Parse href to separate file path and fragment
		href_parts = (section.href or "").split('#', 1)
		href_file = href_parts[0].lstrip('./')
		fragment_id = href_parts[1] if len(href_parts) > 1 else None
		href_decoded = urllib.parse.unquote(href_file)
		
		text = ""
		try:
			# Try with decoded href first
			item = book.get_item_with_href(href_decoded)
			if item is None:
				# Fallback to original href
				item = book.get_item_with_href(href_file)
			if item is not None:
				html = item.get_content().decode("utf-8", errors="ignore")
				text = _extract_fragment_content(html, fragment_id)
		except Exception:
			text = ""
		
		text_tokens = (text or "").split()
		if len(text_tokens) >= 100:
			substantial_sections.append(section)
		else:
			dropped_short.append(section)
			logger.debug(f"  ✗ Dropped (too short): {section.title} ({len(text_tokens)} tokens)")
	
	logger.info(f"After pre-filtering: {len(substantial_sections)} substantial sections, {len(dropped_short)} short sections dropped")
	
	if not substantial_sections:
		return {"kept_sections": [], "dropped": [s.href for s in sorted_sections], "error": "No substantial sections found"}
	
	# Each side can check up to half the substantial sections (minimum 5)
	max_check_per_side = max(5, len(substantial_sections) // 2)
	
	start_index = None
	end_index = None
	total_tokens = 0
	
	logger.info(f"Searching for narrative boundaries in {len(substantial_sections)} substantial sections...")
	sys.stdout.flush()
	logger.debug(f"Will check up to {max_check_per_side} sections from each side")
	sys.stdout.flush()
	
	# Search top-to-bottom for first valid section
	# Need TWO consecutive narrative sections to confirm start boundary
	logger.debug("START_TOP=TRUE")  # Flag for web interface
	sys.stdout.flush()
	await asyncio.sleep(0.5)  # Give web interface time to process
	
	logger.info("Searching top-to-bottom for start boundary...")
	sys.stdout.flush()
	consecutive_narrative = 0
	for i, section in enumerate(substantial_sections[:max_check_per_side]):
		if tracker:
			progress = int((i / min(max_check_per_side, len(substantial_sections))) * 100)
			tracker.set_top_boundary_progress(progress)
		
		logger.info(f"  Checking section {i+1}: {section.title}")
		sys.stdout.flush()
		is_narrative, tokens_used = await _is_valid_narrative_section(section, epub_path, model)
		total_tokens += tokens_used
		
		if is_narrative:
			consecutive_narrative += 1
			logger.info(f"  ✓ Narrative: {section.title} (consecutive: {consecutive_narrative})")
			sys.stdout.flush()
			
			if consecutive_narrative >= 2:
				# Found two consecutive narrative sections, use the first one as start
				start_index = i - 1  # Use the previous section as the start
				logger.info(f"  ✓ Confirmed start boundary at section {start_index+1}: {substantial_sections[start_index].title}")
				sys.stdout.flush()
				break
		else:
			consecutive_narrative = 0  # Reset counter
			logger.info(f"  ✗ Not narrative: {section.title}")
			sys.stdout.flush()
	
	# If we only found one narrative section at the end, use it
	if start_index is None and consecutive_narrative == 1:
		start_index = min(max_check_per_side - 1, len(substantial_sections) - 1)
		logger.info(f"  ✓ Using single narrative section as start: {substantial_sections[start_index].title}")
		sys.stdout.flush()
	
	logger.debug("END_TOP=TRUE")  # Flag for web interface
	sys.stdout.flush()
	if tracker:
		tracker.set_top_boundary_progress(100, done=True)
	await asyncio.sleep(1)  # Give web interface time to update
	
	# Search bottom-to-top for last valid section
	# Need TWO consecutive narrative sections to confirm end boundary
	logger.debug("START_BOTTOM=TRUE")  # Flag for web interface
	sys.stdout.flush()
	if tracker:
		tracker.set_bottom_boundary_progress(0)
	await asyncio.sleep(0.5)  # Give web interface time to process
	
	logger.info("Searching bottom-to-top for end boundary...")
	sys.stdout.flush()
	consecutive_narrative = 0
	bottom_sections = substantial_sections[-max_check_per_side:]
	for i, section in enumerate(reversed(bottom_sections)):
		if tracker:
			progress = int((i / min(max_check_per_side, len(bottom_sections))) * 100)
			tracker.set_bottom_boundary_progress(progress)
		
		logger.info(f"  Checking section from end {i+1}: {section.title}")
		sys.stdout.flush()
		is_narrative, tokens_used = await _is_valid_narrative_section(section, epub_path, model)
		total_tokens += tokens_used
		
		if is_narrative:
			consecutive_narrative += 1
			logger.info(f"  ✓ Narrative: {section.title} (consecutive: {consecutive_narrative})")
			sys.stdout.flush()
			
			if consecutive_narrative >= 2:
				# Found two consecutive narrative sections, use the second one as end
				end_index = len(substantial_sections) - i  # Use the next section as the end
				logger.info(f"  ✓ Confirmed end boundary: {substantial_sections[end_index].title}")
				sys.stdout.flush()
				break
		else:
			consecutive_narrative = 0  # Reset counter
			logger.info(f"  ✗ Not narrative: {section.title}")
			sys.stdout.flush()
	
	# If we only found one narrative section at the end, use it
	if end_index is None and consecutive_narrative == 1:
		reverse_index = min(max_check_per_side - 1, len(substantial_sections) - 1)
		end_index = len(substantial_sections) - 1 - reverse_index
		logger.info(f"  ✓ Using single narrative section as end: {substantial_sections[end_index].title}")
		sys.stdout.flush()
	
	logger.debug("END_BOTTOM=TRUE")  # Flag for web interface
	sys.stdout.flush()
	if tracker:
		tracker.set_bottom_boundary_progress(100, done=True)
	await asyncio.sleep(1)  # Give web interface time to update
	
	# If we couldn't find boundaries, fall back to keeping everything substantial
	if start_index is None:
		logger.warning("No start boundary found, using first substantial section")
		start_index = 0
	if end_index is None:
		logger.warning("No end boundary found, using last substantial section")
		end_index = len(substantial_sections) - 1
	
	# Extract sections between boundaries (inclusive)
	kept_sections_data = substantial_sections[start_index:end_index + 1]
	
	def _full_title(s: Section) -> str:
		parts = [pt for pt in (s.parents or []) if pt]
		if s.title:
			parts.append(s.title)
		return " — ".join(parts) if parts else (s.title or "")
	
	kept_sections = [
		KeptSection(
			href=s.href,
			title=s.title,
			full_title=_full_title(s),
			order=s.order
		)
		for s in kept_sections_data
	]
	
	# Track what we dropped (both short sections and sections outside boundaries)
	dropped_hrefs = []
	# Add short sections
	dropped_hrefs.extend([s.href for s in dropped_short])
	# Add sections outside boundaries
	dropped_hrefs.extend([s.href for i, s in enumerate(substantial_sections) if i < start_index or i > end_index])
	
	# Log boundary detection usage summary
	pu = _get_provider_usage_dict()
	if pu.get("prompt_count"):
		logger.info(
			f"Boundary detection done — {pu['prompt_count']} LLM calls, "
			f"{pu.get('total_input_tokens', 0):,} in / "
			f"{pu.get('total_output_tokens', 0):,} out / "
			f"{pu.get('total_tokens', 0):,} total tokens"
		)

	return {
		"kept_sections": [ks.model_dump() for ks in kept_sections],
		"dropped": dropped_hrefs,
		"start_boundary": substantial_sections[start_index].title if start_index is not None else None,
		"end_boundary": substantial_sections[end_index].title if end_index is not None else None,
		"tokens_used": total_tokens,
		"provider_usage": _get_provider_usage_dict(),
	}


# --- Chapter text extraction helpers ---
def _html_to_text(html: str) -> str:
	"""Very lightweight HTML -> plain text.

	Minimal policy for "pure text":
	- Drop scripts/styles/comments
	- Convert <br> and common block tags to newlines
	- Strip remaining tags
	- HTML unescape
	- Do not collapse spaces or newlines (preserve original spacing as much as possible)
	"""
	# Remove scripts/styles and comments entirely
	html = re.sub(r"<\s*(script|style)[^>]*>[\s\S]*?<\s*/\s*\1\s*>", "", html, flags=re.IGNORECASE)
	html = re.sub(r"<!--([\s\S]*?)-->", "", html)
	# Map common block tags and <br> to newlines to preserve rough structure
	block_tags = [
		"h1","h2","h3","h4","h5","h6","p","div","section","article","blockquote","li","ul","ol","header","footer"
	]
	for tag in block_tags:
		html = re.sub(rf"<\s*{tag}[^>]*>", "\n", html, flags=re.IGNORECASE)
		html = re.sub(rf"<\s*/\s*{tag}\s*>", "\n", html, flags=re.IGNORECASE)
	html = re.sub(r"<\s*br\s*/?\s*>", "\n", html, flags=re.IGNORECASE)
	# Strip remaining tags (leave raw text/whitespace as-is)
	html = re.sub(r"<[^>]+>", "", html)
	# Unescape entities and normalize line endings to LF only (no whitespace collapsing)
	text = html_module.unescape(html).replace("\r\n", "\n").replace("\r", "\n")
	return text.strip()


def _extract_texts_for_sections(epub_path: str, kept_sections: List[KeptSection]) -> List[dict]:
	"""Return records with chapter metadata and full plain text for each kept section.

	Each record has: order, href, title, full_title, text, text_with_ids
	The text_with_ids field contains paragraphs prefixed with "ID X:" for tracking.
	"""
	_ensure_deps()
	try:
		from ebooklib import epub as _epub
		import urllib.parse
	except Exception as e:
		raise RuntimeError("Missing dependency 'ebooklib'. Install with: pip install ebooklib") from e

	book = _epub.read_epub(epub_path)
	records: List[dict] = []
	global_paragraph_id = 0  # Global counter for paragraph IDs across all chapters
	
	for s in sorted(kept_sections, key=lambda x: x.order):
		# Parse href to separate file path and fragment
		href_parts = (s.href or "").split('#', 1)
		href_file = href_parts[0].lstrip('./')
		fragment_id = href_parts[1] if len(href_parts) > 1 else None
		
		# URL decode the href to handle encoded characters
		href_decoded = urllib.parse.unquote(href_file)
		text = ""
		
		try:
			# Try with decoded href first
			item = book.get_item_with_href(href_decoded)
			if item is None:
				# Fallback to original href
				item = book.get_item_with_href(href_file)
			if item is None:
				# Some epubs need an alternative lookup by name
				item = getattr(book, 'get_item_with_id', lambda _id: None)(href_decoded)
			if item is None:
				item = getattr(book, 'get_item_with_id', lambda _id: None)(href_file)
				
			if item is not None:
				html = item.get_content().decode("utf-8", errors="ignore")
				# Extract content for the specific fragment
				text = _extract_fragment_content(html, fragment_id)
		except Exception:
			# Best-effort; leave text empty on failure
			text = ""
		
		# Split text into paragraphs and assign IDs
		paragraphs = _split_into_paragraphs(text)
		
		# Build text with paragraph IDs
		paragraphs_with_ids = []
		paragraph_id_mapping = []  # Store (id, paragraph) tuples
		for para in paragraphs:
			if para.strip():  # Only assign IDs to non-empty paragraphs
				paragraphs_with_ids.append(f"ID {global_paragraph_id}: {para}")
				paragraph_id_mapping.append((global_paragraph_id, para))
				global_paragraph_id += 1
		
		text_with_ids = "\n\n".join(paragraphs_with_ids)
			
		records.append({
			"order": s.order,
			"href": s.href,
			"title": s.title,
			"full_title": getattr(s, 'full_title', s.title),
			"text": text,  # Original text without IDs
			"text_with_ids": text_with_ids,  # Text with paragraph IDs
			"paragraph_id_mapping": paragraph_id_mapping,  # List of (id, paragraph) tuples
		})
	return records


def _split_into_paragraphs(text: str) -> List[str]:
	"""Split text into paragraphs based on double newlines or multiple newlines.
	
	Args:
		text: The text to split.
		
	Returns:
		List of paragraph strings.
	"""
	if not text:
		return []
	
	# Split on double newlines or multiple consecutive newlines
	paragraphs = re.split(r'\n\s*\n+', text.strip())
	
	# Filter out empty paragraphs and strip whitespace
	paragraphs = [p.strip() for p in paragraphs if p.strip()]
	
	return paragraphs


# --- Token counting helper (kept for estimation reporting) ---
def _token_count(text: str, model: str) -> int:
	try:
		import tiktoken  # type: ignore
	except Exception:
		return 0
	try:
		enc = tiktoken.encoding_for_model(model)
	except Exception:
		enc = tiktoken.get_encoding("cl100k_base")
	return len(enc.encode(text or ""))





def main(argv: Optional[List[str]] = None, job_id: Optional[str] = None) -> int:
	parser = argparse.ArgumentParser(description="EPUB -> narrative chapters extraction with LLM boundary detection")
	parser.add_argument("epub", help="Path to the .epub file")
	parser.add_argument(
		"--model",
		default=os.getenv("EPUB_AGENT_MODEL", "gpt-4o-mini"),
		help="LLM model to use (default: gpt-4o-mini)",
	)
	parser.add_argument(
		"--save",
		default=None,
		help="Path to save chapters dataframe. If omitted, a default .xlsx will be written to test_out/<name>.xlsx.",
	)
	parser.add_argument(
		"--job-id",
		default=job_id,
		help="Job ID for progress tracking",
	)

	args = parser.parse_args(argv)
	epub_path = args.epub
	job_id = args.job_id or job_id
	
	# Initialize progress tracker if available
	tracker = None
	if ProgressTracker and job_id:
		tracker = ProgressTracker(job_id)
		tracker.update_stage("initializing")
		tracker.update_overall_progress(0)
	
	if not os.path.exists(epub_path):
		logger.error(f"File not found: {epub_path}")
		if tracker:
			tracker.set_error(f"File not found: {epub_path}")
		return 2

	try:
		# Silence noisy warnings from ebooklib
		warnings.filterwarnings("ignore", category=UserWarning, module=r"ebooklib.*")
		warnings.filterwarnings("ignore", category=FutureWarning, module=r"ebooklib.*")
		# Load .env before anything else so OPENAI_API_KEY and others are available
		try:
			from dotenv import load_dotenv  # type: ignore
			load_dotenv(override=False)
		except Exception:
			pass
		# Quick dependency check
		_ensure_deps()
	except RuntimeError as e:
		logger.error(str(e))
		return 3

	# Run the boundary detection
	try:
		if tracker:
			tracker.update_stage("boundary_detection")
			tracker.update_overall_progress(10)
		parsed = asyncio.run(run_boundary_detection(epub_path, args.model, tracker))
	except Exception as e:
		logger.error(f"Boundary detection failed: {e}")
		if tracker:
			tracker.set_error(f"Boundary detection failed: {e}")
		return 4

	# Collect results for final reporting
	first_step_kept_titles: list[str] = []
	first_step_dropped_hrefs: list[str] = []
	if isinstance(parsed, dict):
		if "kept_sections" in parsed and isinstance(parsed["kept_sections"], list):
			first_step_kept_titles = [
				(it.get("full_title") or it.get("title") if isinstance(it, dict) else str(it))
				for it in parsed["kept_sections"]
			]
		if "dropped" in parsed and isinstance(parsed["dropped"], list):
			first_step_dropped_hrefs = [str(t) for t in parsed["dropped"]]
		
		# Print boundary detection results
		if "start_boundary" in parsed and parsed["start_boundary"]:
			logger.info(f"Start boundary: {parsed['start_boundary']}")
		if "end_boundary" in parsed and parsed["end_boundary"]:
			logger.info(f"End boundary: {parsed['end_boundary']}")

	# Build KeptSection list from parsed for extraction
	kept_objs: List[KeptSection] = []
	if isinstance(parsed, dict) and isinstance(parsed.get("kept_sections"), list):
		for it in parsed["kept_sections"]:
			if isinstance(it, dict):
				kept_objs.append(KeptSection(**{
					"href": it.get("href", ""),
					"title": it.get("title", ""),
					"full_title": it.get("full_title") or it.get("title", ""),
					"order": int(it.get("order", 0) or 0),
				}))
	try:
		import pandas as pd  # type: ignore
	except Exception:
		logger.error("pandas is required to save the chapters dataframe. Install with: pip install pandas")
		return 5
	
	records = _extract_texts_for_sections(epub_path, kept_objs)
	# Preprocess: collapse excessive line breaks to a single newline
	records = _preprocess_records_collapse_newlines(records)
	# Clean chapter texts: detect the exact opening sentence and trim prefixes
	records = _clean_records_with_llm(records, args.model, tracker)
	# Build simplified dataframe with only Chapter and Chunk columns
	simplified_records = [
		{
			"Chapter": record.get("full_title") or record.get("title", ""),
			"Chunk": record.get("text", "")
		}
		for record in records
	]
	df = pd.DataFrame.from_records(simplified_records)
	# Determine output path; default to test_out/<basename>.xlsx
	if args.save:
		out_path = args.save
	else:
		base = os.path.splitext(os.path.basename(epub_path))[0]
		out_dir = os.path.join(os.path.dirname(__file__), "test_out")
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"{base}.xlsx")
	ext = os.path.splitext(out_path)[1].lower() or ".xlsx"
	try:
		# Always prefer .xlsx; if user asked for another extension, handle a few common cases
		if ext == ".xlsx":
			try:
				# openpyxl is the usual writer; if missing, instruct the user
				import openpyxl  # type: ignore  # noqa: F401
			except Exception:
				logger.warning("openpyxl not found; attempting to write .xlsx may fail. Install with: pip install openpyxl")
			df.to_excel(out_path, index=False)
		elif ext == ".csv":
			df.to_csv(out_path, index=False)
		elif ext == ".jsonl":
			with open(out_path, "w", encoding="utf-8") as f:
				for rec in records:
					f.write(json.dumps(rec, ensure_ascii=False) + "\n")
		elif ext == ".json":
			df.to_json(out_path, orient="records", force_ascii=False, indent=2)
		else:
			# Unknown extension: write xlsx by default
			try:
				import openpyxl  # type: ignore  # noqa: F401
			except Exception:
				pass
			df.to_excel(out_path if out_path.endswith('.xlsx') else out_path + '.xlsx', index=False)
		logger.info(f"Saved chapters dataframe to: {out_path}")
		if tracker:
			tracker.set_completed()
		# Final printed output: final kept list and dropped section names
		logger.info("Final kept sections:")
		for i, row in enumerate(df.itertuples(index=False), start=1):
			logger.info(f"{i:02d}. {getattr(row, 'Chapter')}")

		# Map first-step dropped hrefs to titles for interpretability
		dropped_names: list[str] = []
		try:
			sections_all = _read_sections(epub_path)
			href_to_title = {s.href: (" — ".join([*s.parents, s.title]) if s.title else s.title) for s in sections_all}
			for h in first_step_dropped_hrefs:
				name = href_to_title.get(h)
				if name:
					dropped_names.append(name)
		except Exception:
			pass
		# if dropped_names:
		# 	print("\nDropped sections (by title):")
		# 	for name in dropped_names:
		# 		print(f"- {name}")

		# Token usage summary
		boundary_tokens = parsed.get("tokens_used", 0) if isinstance(parsed, dict) else 0
		pu = parsed.get("provider_usage", {}) if isinstance(parsed, dict) else {}
		if pu.get("prompt_count"):
			logger.info(
				f"LLM usage: {pu['prompt_count']} prompts, "
				f"{pu.get('total_input_tokens', 0):,} input tokens, "
				f"{pu.get('total_output_tokens', 0):,} output tokens, "
				f"{pu.get('total_tokens', 0):,} total tokens"
			)
		else:
			logger.info(f"Estimated tokens — boundary detection: {boundary_tokens}")
	except Exception as e:
		logger.error(f"Failed to save chapters dataframe: {e}")
		return 6

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

