"""Chunking strategies for the retrieval experiments.

Two strategies are compared head-to-head by the eval harness:

- ``fixed``: uniform sliding windows of N words with overlap, ignoring
  document structure entirely.
- ``structure``: split on SEC Item headings (Item 1, Item 1A, ...) first,
  then size-cap each section with the same sliding window. Every chunk
  keeps the heading of the section it came from.

Each chunk records its source document and section so results can be
cited and so the eval can tell *where* a hit came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ITEM_HEADING = re.compile(r"^Item\s+\d+[A-Z]?\.\s+.*$", re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    doc_id: str  # e.g. "AAPL"
    section: str  # heading line the chunk belongs to, or "" if none
    index: int  # position within the document's chunk list


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    if not words:
        return []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def fixed_chunks(
    text: str, doc_id: str, chunk_words: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Uniform sliding-window chunks; section is tracked but not used to split."""
    words = text.split()
    section_by_pos: list[str] = []
    current = ""
    for line in text.splitlines():
        if ITEM_HEADING.match(line.strip()):
            current = line.strip()
        section_by_pos.append(current)
    # Map a word offset back to the section in force at that point.
    line_word_starts: list[int] = []
    total = 0
    for line in text.splitlines():
        line_word_starts.append(total)
        total += len(line.split())

    def section_at(word_pos: int) -> str:
        sec = ""
        for start, s in zip(line_word_starts, section_by_pos):
            if start <= word_pos:
                sec = s
            else:
                break
        return sec

    chunks = []
    pos = 0
    step = chunk_words - overlap
    for i, window in enumerate(_windows(words, chunk_words, overlap)):
        chunks.append(
            Chunk(
                text=" ".join(window),
                doc_id=doc_id,
                section=section_at(pos),
                index=i,
            )
        )
        pos += step
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a filing into (heading, body) pairs on Item heading lines."""
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        if ITEM_HEADING.match(line.strip()):
            sections.append((heading, body))
            heading = line.strip()
            body = []
        else:
            body.append(line)
    sections.append((heading, body))
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def structure_chunks(
    text: str, doc_id: str, max_words: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Split on Item headings first, then size-cap each section.

    The heading is prepended to every chunk from that section so a chunk
    is never separated from the section it belongs to.
    """
    chunks: list[Chunk] = []
    idx = 0
    for heading, body in split_sections(text):
        words = body.split()
        if not words:
            continue
        if len(words) <= max_words:
            windows = [words]
        else:
            windows = _windows(words, max_words, overlap)
        for window in windows:
            piece = " ".join(window)
            full = f"{heading}\n{piece}" if heading else piece
            chunks.append(Chunk(text=full, doc_id=doc_id, section=heading, index=idx))
            idx += 1
    return chunks


CHUNKERS = {
    "fixed": fixed_chunks,
    "structure": structure_chunks,
}
