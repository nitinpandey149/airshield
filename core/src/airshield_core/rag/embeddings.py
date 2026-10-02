"""Embedding backends for the retrieval index.

The default backend is a real static sentence-embedding model (``model2vec``,
``minishlab/potion-base-8M``), which runs on CPU with no GPU and no PyTorch. It
produces genuine semantic vectors, so "why is running worse than walking" can
match a passage that never uses the word "running".

A deterministic hashing embedder is available as an explicit fallback for
environments where the model cannot be downloaded. It is a *lexical* bag-of-words
projection, not a semantic model, and it says so: the index records which backend
produced it, and the API surfaces that, so a degraded index is never presented as
a full-quality one.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

#: Default model. 8M parameters, 256 dimensions, fast on CPU.
DEFAULT_MODEL = "minishlab/potion-base-8M"


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into unit-norm vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_semantic(self) -> bool: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Model2VecEmbedder:
    """Real semantic embeddings via a static distilled model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from model2vec import StaticModel

        self._model = StaticModel.from_pretrained(model_name)
        self._name = model_name
        probe = self._model.encode(["dimension probe"])
        self._dimension = int(np.asarray(probe).shape[1])

    @property
    def name(self) -> str:
        return f"model2vec:{self._name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_semantic(self) -> bool:
        return True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = np.asarray(self._model.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return _normalise_rows(vectors)


class HashingEmbedder:
    """Deterministic lexical fallback. NOT a semantic model.

    Hashes word unigrams and bigrams into a fixed-width vector. It matches on
    shared vocabulary only, so it is noticeably weaker than the semantic
    backend. ``is_semantic`` is False and the name says so, which is how callers
    (and the UI) can tell the difference.
    """

    def __init__(self, dimension: int = 512):
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hashing-fallback:{self._dimension}d"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_semantic(self) -> bool:
        return False

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        features = list(words)
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        return features

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self._dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
        if len(texts) == 0:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return _normalise_rows(matrix)


def build_embedder(
    prefer: str | None = None, *, allow_fallback: bool = True
) -> Embedder:
    """Build the best available embedder.

    ``prefer`` may be ``"semantic"``, ``"hashing"`` or a specific model2vec model
    name. When the semantic model is unavailable and ``allow_fallback`` is set,
    the lexical fallback is returned so the assistant still works — but it
    identifies itself, so no caller can mistake it for the real thing.
    """
    if prefer == "hashing":
        return HashingEmbedder()

    model_name = prefer if prefer and prefer != "semantic" else DEFAULT_MODEL
    try:
        return Model2VecEmbedder(model_name)
    except Exception:
        if not allow_fallback:
            raise
        return HashingEmbedder()
