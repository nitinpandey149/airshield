"""Chunking: split a document into retrievable passages.

Chunks are built from whole paragraphs and only split mid-paragraph when a single
paragraph exceeds the target size. Overlap keeps a sentence that straddles a
boundary retrievable from both sides. Every chunk inherits the parent document's
metadata, which is what makes citations possible.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from airshield_core.rag.documents import Document

#: Target chunk size in characters. Small enough for precise retrieval, large
#: enough to carry a complete idea.
DEFAULT_CHUNK_CHARS = 900
#: Characters of overlap between consecutive chunks.
DEFAULT_OVERLAP_CHARS = 150
#: Chunks shorter than this are dropped as noise (headers, stray fragments).
MIN_CHUNK_CHARS = 80

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with full provenance."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict

    @property
    def citation_key(self) -> str:
        return self.doc_id


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping paragraph boundaries intact."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(paragraph: str, size: int, overlap: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries, then on characters."""
    sentences = _SENTENCE_END.split(paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            # A single sentence longer than the target: fall back to characters.
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(sentence), size - overlap):
                pieces.append(sentence[start : start + size])
            continue
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the tail of each piece to the next, for boundary continuity."""
    if len(pieces) <= 1 or overlap <= 0:
        return pieces
    out = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        tail = previous[-overlap:]
        # Trim to a word boundary so the overlap does not start mid-word.
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1 :]
        out.append(f"{tail} {current}".strip() if tail else current)
    return out


def chunk_document(
    document: Document,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split one document into overlapping, metadata-carrying chunks."""
    paragraphs = _split_paragraphs(document.text)

    # Pack paragraphs greedily up to the target size.
    packed: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_hard_split(paragraph, chunk_chars, overlap_chars))
            continue
        if len(current) + len(paragraph) + 2 <= chunk_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                packed.append(current)
            current = paragraph
    if current:
        packed.append(current)

    packed = _apply_overlap(packed, overlap_chars)

    metadata = document.metadata()
    chunks: list[Chunk] = []
    for ordinal, text in enumerate(packed):
        if len(text) < min_chars:
            continue
        digest = hashlib.sha1(f"{document.doc_id}:{ordinal}:{text}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=digest,
                doc_id=document.doc_id,
                text=text,
                ordinal=ordinal,
                metadata=metadata,
            )
        )
    return chunks
