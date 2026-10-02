"""Retrieval-augmented knowledge layer for AirShield Pulse.

This package answers *knowledge* questions ("What is PM2.5?") by retrieving from
the curated corpus in ``knowledge/``. It is deliberately separate from the
numerical pipeline: no PM2.5 forecast, AQI value, exposure score or route
ranking is ever produced here. Those come from the XGBoost model and the
deterministic engines in :mod:`airshield_core.exposure` and
:mod:`airshield_core.windows`.

See ``docs/rag.md`` for the reasoning behind that boundary.
"""

from airshield_core.rag.answering import Answer, GroundedAnswerer
from airshield_core.rag.chunking import Chunk, chunk_document
from airshield_core.rag.context import ForecastContext
from airshield_core.rag.documents import Document, KnowledgeBase, load_knowledge_base
from airshield_core.rag.embeddings import Embedder, build_embedder
from airshield_core.rag.retrieval import RetrievedChunk, Retriever
from airshield_core.rag.vectorstore import VectorStore

__all__ = [
    "Answer",
    "GroundedAnswerer",
    "Chunk",
    "chunk_document",
    "ForecastContext",
    "Document",
    "KnowledgeBase",
    "load_knowledge_base",
    "Embedder",
    "build_embedder",
    "RetrievedChunk",
    "Retriever",
    "VectorStore",
]
