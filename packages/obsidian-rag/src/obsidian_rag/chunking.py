"""Chinese-optimized text chunking for Obsidian RAG.

Key design:
- Splits on Chinese punctuation (。！？；、：) and English punctuation (.!?:;,)
- Uses paragraph splitting for additional granularity
- Merges small chunks up to a target size
- Preserves metadata (path, chunk index) for each output chunk
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from obsidian_rag.config import MIN_CHUNK_CHARS, MAX_CHUNK_CHARS, TARGET_CHUNK_CHARS

# ── Chinese punctuation split pattern ─────────────────────────────────────────
#
# Primary split on full-width Chinese punctuation:
#   。 ！？；：
# Also split on ASCII equivalents to keep logic unified:
#   . ! ? ; :
# Trailing whitespace is stripped from each segment.

_CHUNK_SPLIT_RE = re.compile(
    r"(?<=[。！？；：.!?:;,])\s*",
    re.UNICODE,
)

# Paragraph separator (double newline or single newline)
_PARAGRAPH_RE = re.compile(r"\n\s*\n", re.UNICODE)


@dataclass
class ChunkOptions:
    """Configurable chunking parameters."""
    min_chars: int = MIN_CHUNK_CHARS
    max_chars: int = MAX_CHUNK_CHARS
    target_chars: int = TARGET_CHUNK_CHARS


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    id: str          # MD5 of "path-index"
    path: str        # Relative path in vault
    index: int       # Position within the note
    text: str


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _split_sentences(text: str) -> list[str]:
    """Split text on punctuation boundaries, return non-empty segments."""
    parts = _CHUNK_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text on paragraph boundaries."""
    parts = _PARAGRAPH_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, options: Optional[ChunkOptions] = None) -> list[str]:
    """Split raw body text into a list of text chunks.

    Algorithm:
    1. Split into paragraphs (double newline).
    2. For each paragraph:
       - If it fits in max_chars, treat as a single segment.
       - Otherwise, split on punctuation boundaries.
    3. Merge short segments up to target_chars.

    Returns a flat list of chunk strings (raw text, not Chunk objects).
    """
    if options is None:
        options = ChunkOptions()

    min_c = options.min_chars
    max_c = options.max_chars
    target_c = options.target_chars

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    if len(text) <= max_c:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> list[str]:
        """Merge accumulated segments into chunks of up to target_chars."""
        merged: list[str] = []
        buf = ""
        for seg in current:
            if len(seg) >= target_c:
                if buf:
                    merged.append(buf)
                    buf = ""
                merged.append(seg)
            elif len(buf) + len(seg) + 2 <= target_c:
                buf = (buf + "\n\n" + seg).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = seg
        if buf:
            merged.append(buf)
        return merged

    for paragraph in _split_paragraphs(text):
        if not paragraph:
            continue

        if len(paragraph) <= max_c:
            current.append(paragraph)
        else:
            # Flush any accumulated short segments before long paragraph
            if current:
                chunks.extend(flush())
                current = []

            # Split long paragraph on punctuation
            sentences = _split_sentences(paragraph)
            for sent in sentences:
                if not sent:
                    continue
                if len(sent) <= max_c:
                    current.append(sent)
                else:
                    # Very long sentence — chunk it directly
                    if current:
                        chunks.extend(flush())
                        current = []
                    # Split long sentence into fixed-size pieces
                    for i in range(0, len(sent), max_c):
                        sub = sent[i : i + max_c]
                        if len(sub) >= min_c:
                            chunks.append(sub)

    if current:
        chunks.extend(flush())

    # Filter out chunks below minimum size
    return [c for c in chunks if len(c) >= min_c]


def build_chunks(
    relative_path: str,
    body: str,
    options: Optional[ChunkOptions] = None,
) -> list[Chunk]:
    """Build a list of Chunk objects from a note's body text.

    Args:
        relative_path: Vault-relative path (e.g. "Daily Notes/2024-01-01.md")
        body: The note body (frontmatter already stripped)
        options: Chunking parameters

    Returns:
        List of Chunk objects with id, path, index, text.
    """
    if options is None:
        options = ChunkOptions()

    texts = chunk_text(body, options)
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                id=_md5(f"{relative_path}-{i}"),
                path=relative_path,
                index=i,
                text=text,
            )
        )
    return chunks
