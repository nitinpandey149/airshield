"""Semantic retrieval over the knowledge index.

A small, honest retrieval layer: embed the question, run an exact cosine search,
drop anything below a relevance floor, and return the passages with their source
metadata attached. If nothing clears the floor the result is empty, and the
assistant is expected to say it does not know rather than to answer from memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from airshield_core.rag.embeddings import Embedder
from airshield_core.rag.vectorstore import SearchHit, VectorStore, VectorStoreError

#: Default number of passages handed to the language model. Small enough to stay
#: focused, large enough to cover a question that spans two documents.
DEFAULT_TOP_K = 4

#: Minimum cosine similarity for a passage to count as relevant. Tuned so an
#: unrelated question ("what is the capital of France") retrieves nothing.
DEFAULT_MIN_SCORE = 0.22

#: Queries shorter than this cannot be judged relevant; retrieval is skipped.
MIN_QUERY_CHARS = 3

#: Words that carry no retrieval signal on their own.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did doing have has had having
    i me my we our you your it its of to in on at for with about into over after
    and or but if then than so what which who whom this that these those there
    how why when where can could should would may might will just please tell
    """.split()
)


@dataclass
class RetrievedChunk:
    """A retrieved passage plus the provenance needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: str
    source: str
    url: str
    category: str
    document_type: str
    publication_date: str | None
    licence: str
    ordinal: int

    def citation(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "category": self.category,
            "document_type": self.document_type,
            "publication_date": self.publication_date,
            "licence": self.licence,
        }


class RetrievalError(RuntimeError):
    """Raised when the index cannot be used."""


class Retriever:
    """Embeds queries and searches the vector index."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    @classmethod
    def from_index(
        cls,
        index_dir,
        *,
        embedder: Embedder | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> "Retriever":
        store = VectorStore.load(index_dir)
        if embedder is None:
            # Rebuild the embedder that produced the index, so query and index
            # vectors live in the same space.
            from airshield_core.rag.embeddings import build_embedder

            preferred = None
            if store.embedder_name.startswith("hashing-fallback"):
                preferred = "hashing"
            elif store.embedder_name.startswith("model2vec:"):
                preferred = store.embedder_name.split(":", 1)[1]
            embedder = build_embedder(preferred)
        if embedder.dimension != store.dimension:
            raise RetrievalError(
                f"embedder produces {embedder.dimension}-d vectors but the index is "
                f"{store.dimension}-d; rebuild the index with "
                "`python knowledge/build_index.py`"
            )
        return cls(store, embedder, top_k=top_k, min_score=min_score)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _is_meaningful(query: str) -> bool:
        """Reject queries with no content words, e.g. "???" or "the a"."""
        if len(query.strip()) < MIN_QUERY_CHARS:
            return False
        words = re.findall(r"[a-z0-9]+", query.lower())
        return any(word not in _STOPWORDS and len(word) > 1 for word in words)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return relevant passages, best first. Empty when nothing is relevant."""
        if not query or not self._is_meaningful(query):
            return []
        if self.store.is_empty:
            return []

        vector = self.embedder.encode([query])[0]
        hits: list[SearchHit] = self.store.search(
            vector,
            top_k=top_k or self.top_k,
            min_score=min_score if min_score is not None else self.min_score,
            category=category,
        )
        return [
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=hit.text,
                score=hit.score,
                title=hit.metadata.get("title", ""),
                source=hit.metadata.get("source", ""),
                url=hit.metadata.get("url", ""),
                category=hit.metadata.get("category", ""),
                document_type=hit.metadata.get("document_type", ""),
                publication_date=hit.metadata.get("publication_date"),
                licence=hit.metadata.get("licence", ""),
                ordinal=hit.ordinal,
            )
            for hit in hits
        ]

    def citations(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Unique citations, in the order the passages were ranked.

        Only documents that were actually retrieved appear, so the UI can never
        show a source that did not contribute.
        """
        seen: set[str] = set()
        out: list[dict] = []
        for chunk in chunks:
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            out.append(chunk.citation())
        return out

    def stats(self) -> dict:
        return {
            **self.store.stats(),
            "top_k": self.top_k,
            "min_score": self.min_score,
        }
