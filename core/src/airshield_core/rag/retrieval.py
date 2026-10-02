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

#: Weight of the lexical overlap term in the hybrid score. Semantic similarity
#: dominates; the lexical term breaks ties in favour of a passage that actually
#: contains the words asked about. Without it, a question naming a specific
#: standard ("India AQI") can rank a generically-similar document ("US AQI")
#: above the one that answers it.
LEXICAL_WEIGHT = 0.25

#: Extra score for a document whose jurisdiction matches the one the question is
#: about. The Indian and US AQI scales share category names but not breakpoints,
#: so answering an Indian question from a US document is a correctness bug, not
#: just a ranking miss. This is a preference, not a filter: a jurisdiction match
#: is boosted, never required.
JURISDICTION_BONUS = 0.12

#: Words that indicate the question is about the Indian regime.
_INDIA_CUES = frozenset(
    """india indian cpcb ncap delhi mumbai bengaluru bangalore chennai kolkata
    hyderabad satisfactory severe grap""".split()
)

#: Minimum semantic similarity required of a *semantic* index. The hashing
#: fallback embedder produces scores on a lower scale, so the caller's
#: ``min_score`` is used as-is there; this floor only raises the bar when the
#: index is genuinely semantic. 0.38 is set just above the top score an
#: unrelated question reaches on this corpus ("capital of France" peaks at
#: 0.27), so off-topic questions retrieve nothing.
SEMANTIC_SCORE_FLOOR = 0.38

#: Most chunks any single document may contribute to one result set. A long PDF
#: otherwise floods the list and crowds out the other sources.
MAX_CHUNKS_PER_DOCUMENT = 2


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
    jurisdiction: str = "Global"

    def citation(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "category": self.category,
            "document_type": self.document_type,
            "jurisdiction": self.jurisdiction,
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

    @staticmethod
    def _content_words(query: str) -> set[str]:
        """Query words that carry retrieval signal."""
        return {
            word
            for word in re.findall(r"[a-z0-9]+", query.lower())
            if word not in _STOPWORDS and len(word) > 1
        }

    @staticmethod
    def _lexical_overlap(words: set[str], text: str) -> float:
        """Fraction of the query's content words present in ``text``."""
        if not words:
            return 0.0
        present = set(re.findall(r"[a-z0-9]+", text.lower()))
        return len(words & present) / len(words)

    @classmethod
    def _jurisdiction_of(cls, words: set[str]) -> str | None:
        """The regime a question is about, or ``None`` when it does not say."""
        return "India" if words & _INDIA_CUES else None

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
        # Over-fetch, then re-rank: the vector search gives recall, the lexical
        # and jurisdiction terms below give precision on domain words.
        limit = top_k or self.top_k
        candidates = max(limit * 4, 20)
        hits: list[SearchHit] = self.store.search(
            vector, top_k=candidates, min_score=0.0, category=category
        )

        # A semantic index scores on a different scale to the hashing fallback,
        # so the honesty floor is raised only for the former. The caller's
        # explicit min_score always applies.
        floor = min_score if min_score is not None else self.min_score
        if self.store.embedder_is_semantic:
            floor = max(floor, SEMANTIC_SCORE_FLOOR)

        words = self._content_words(query)
        wanted_jurisdiction = self._jurisdiction_of(words)

        ranked: list[tuple[float, SearchHit]] = []
        for hit in hits:
            if hit.score < floor:
                continue
            blended = hit.score + LEXICAL_WEIGHT * self._lexical_overlap(words, hit.text)
            if (
                wanted_jurisdiction is not None
                and hit.metadata.get("jurisdiction") == wanted_jurisdiction
            ):
                blended += JURISDICTION_BONUS
            ranked.append((blended, hit))
        ranked.sort(key=lambda pair: -pair[0])

        # Keep at most MAX_CHUNKS_PER_DOCUMENT passages from any one document so
        # a long report cannot crowd out the rest of the corpus.
        per_document: dict[str, int] = {}
        selected: list[tuple[float, SearchHit]] = []
        for blended, hit in ranked:
            used = per_document.get(hit.doc_id, 0)
            if used >= MAX_CHUNKS_PER_DOCUMENT:
                continue
            per_document[hit.doc_id] = used + 1
            selected.append((blended, hit))
            if len(selected) >= limit:
                break

        return [
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=hit.text,
                score=blended,
                title=hit.metadata.get("title", ""),
                source=hit.metadata.get("source", ""),
                url=hit.metadata.get("url", ""),
                category=hit.metadata.get("category", ""),
                document_type=hit.metadata.get("document_type", ""),
                publication_date=hit.metadata.get("publication_date"),
                licence=hit.metadata.get("licence", ""),
                ordinal=hit.ordinal,
                jurisdiction=hit.metadata.get("jurisdiction", "Global"),
            )
            for blended, hit in selected
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
