"""A small, dependency-light vector store.

Numpy-backed cosine search over a persisted ``.npz`` index plus a JSON sidecar
holding the chunk text and metadata. This is deliberately simple: the corpus is
tens of documents, so an exact brute-force search is both fast and correct, and
there is no service to run.

The persisted format is intentionally plain, and the interface is narrow
(``add`` / ``search`` / ``save`` / ``load``), so it can be swapped for Chroma,
FAISS, OpenSearch or a managed vector database later without touching the
retriever or the assistant.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

from airshield_core.rag.chunking import Chunk

INDEX_FILENAME = "index.npz"
META_FILENAME = "index_meta.json"


class VectorStoreError(RuntimeError):
    """Raised when an index is missing, corrupt or dimensionally inconsistent."""


@dataclass
class SearchHit:
    """One search result with its score and full provenance."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict
    ordinal: int


class VectorStore:
    """Exact cosine-similarity store over normalised vectors."""

    def __init__(self, dimension: int | None = None):
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._dimension = dimension
        self.embedder_name: str = ""
        self.embedder_is_semantic: bool = False
        self.built_at: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def doc_ids(self) -> list[str]:
        return sorted({chunk.doc_id for chunk in self._chunks})

    # ------------------------------------------------------------------- add
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise VectorStoreError("vectors must be a 2-D array")

        if self._vectors is None:
            self._dimension = int(matrix.shape[1])
            self._vectors = matrix
        else:
            if matrix.shape[1] != self._dimension:
                raise VectorStoreError(
                    f"dimension mismatch: index is {self._dimension}-d, "
                    f"got {matrix.shape[1]}-d"
                )
            self._vectors = np.vstack([self._vectors, matrix])
        self._chunks.extend(chunks)

    # ---------------------------------------------------------------- search
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        *,
        min_score: float = 0.0,
        category: str | None = None,
    ) -> list[SearchHit]:
        """Return the ``top_k`` most similar chunks above ``min_score``.

        ``min_score`` is the honesty guard: a query with no genuinely relevant
        passage returns nothing rather than the least-bad match.
        """
        if self._vectors is None or self.is_empty:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._dimension:
            raise VectorStoreError(
                f"query is {query.shape[0]}-d but the index is {self._dimension}-d"
            )
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        scores = self._vectors @ query

        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for index in order:
            chunk = self._chunks[int(index)]
            if category is not None and chunk.metadata.get("category") != category:
                continue
            score = float(scores[int(index)])
            if score < min_score:
                # Sorted descending, so nothing further can qualify.
                break
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                    ordinal=chunk.ordinal,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | pathlib.Path) -> pathlib.Path:
        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            raise VectorStoreError("refusing to save an empty index")

        np.savez_compressed(target / INDEX_FILENAME, vectors=self._vectors)
        meta = {
            "dimension": self._dimension,
            "size": self.size,
            "embedder_name": self.embedder_name,
            "embedder_is_semantic": self.embedder_is_semantic,
            "built_at": self.built_at,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "ordinal": c.ordinal,
                    "metadata": c.metadata,
                }
                for c in self._chunks
            ],
        }
        (target / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | pathlib.Path) -> "VectorStore":
        target = pathlib.Path(directory)
        index_path = target / INDEX_FILENAME
        meta_path = target / META_FILENAME
        if not index_path.is_file() or not meta_path.is_file():
            raise VectorStoreError(
                f"no vector index at {target}. Build it with "
                "`python knowledge/build_index.py`."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dimension=meta.get("dimension"))
        store.embedder_name = meta.get("embedder_name", "")
        store.embedder_is_semantic = bool(meta.get("embedder_is_semantic", False))
        store.built_at = meta.get("built_at")

        vectors = np.load(index_path)["vectors"].astype(np.float32)
        store._vectors = vectors
        store._chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                ordinal=item["ordinal"],
                metadata=item["metadata"],
            )
            for item in meta["chunks"]
        ]
        if len(store._chunks) != vectors.shape[0]:
            raise VectorStoreError(
                f"index is inconsistent: {vectors.shape[0]} vectors but "
                f"{len(store._chunks)} chunks"
            )
        return store

    def stats(self) -> dict:
        return {
            "chunks": self.size,
            "documents": len(self.doc_ids()),
            "dimension": self._dimension,
            "embedder": self.embedder_name,
            "semantic": self.embedder_is_semantic,
            "built_at": self.built_at,
        }
