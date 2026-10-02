"""Knowledge-base documents and their provenance.

A :class:`Document` is a verbatim extract of a real, citable source. The
metadata travels with the text all the way through chunking, embedding and
retrieval, so any answer the assistant gives can point back to the exact source
it came from. A chunk that cannot be traced to a document is not returned.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Literal

Category = Literal[
    "air_quality",
    "pollutants",
    "exposure",
    "outdoor_activity",
    "AQI",
    "environmental_guidance",
]

DocumentType = Literal["guidance", "technical", "standard", "research"]

#: Repo-root-relative default location of the corpus.
DEFAULT_KNOWLEDGE_DIR = "knowledge"


@dataclass(frozen=True)
class Document:
    """One source document, with the provenance needed to cite it."""

    doc_id: str
    title: str
    source: str
    url: str
    category: str
    document_type: str
    text: str
    jurisdiction: str = "Global"
    """Which air-quality regime a document describes: ``India``, ``United
    States`` or ``Global``. Used to prefer the standard the user is in, since
    the two national AQI scales are not interchangeable."""
    publication_date: str | None = None
    licence: str = ""
    retrieved_at: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    def metadata(self) -> dict:
        """Metadata attached to every chunk derived from this document."""
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
            "retrieved_at": self.retrieved_at,
        }

    def citation(self) -> dict:
        """The compact citation shape returned by the API."""
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


@dataclass
class KnowledgeBase:
    """The loaded corpus, plus the manifest that produced it."""

    documents: list[Document] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(d.char_count for d in self.documents)

    def by_id(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def stats(self) -> dict:
        by_category: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for document in self.documents:
            by_category[document.category] = by_category.get(document.category, 0) + 1
            by_source[document.source] = by_source.get(document.source, 0) + 1
        return {
            "documents": len(self.documents),
            "characters": self.total_chars,
            "by_category": by_category,
            "by_source": by_source,
            "sources": sorted({d.source for d in self.documents}),
        }


class KnowledgeBaseError(RuntimeError):
    """Raised when the corpus is missing or unusable."""


def _knowledge_root(path: str | pathlib.Path | None) -> pathlib.Path:
    if path is None:
        # core/src/airshield_core/rag/documents.py -> repo root is parents[4]
        return pathlib.Path(__file__).resolve().parents[4] / DEFAULT_KNOWLEDGE_DIR
    candidate = pathlib.Path(path)
    if candidate.is_absolute():
        return candidate
    return pathlib.Path(__file__).resolve().parents[4] / candidate


def load_knowledge_base(path: str | pathlib.Path | None = None) -> KnowledgeBase:
    """Load the manifest and the extracted source text.

    Raises :class:`KnowledgeBaseError` if the manifest is missing, so a caller
    never silently proceeds with an empty corpus and then answers from memory.
    """
    root = _knowledge_root(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise KnowledgeBaseError(
            f"knowledge base manifest not found at {manifest_path}. "
            "Run `python knowledge/build_index.py` to build it."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources_dir = root / "sources"
    fetch_report_path = root / "fetch_report.json"
    retrieved_at: str | None = None
    if fetch_report_path.is_file():
        try:
            retrieved_at = json.loads(fetch_report_path.read_text(encoding="utf-8")).get(
                "fetched_at"
            )
        except (json.JSONDecodeError, OSError):
            retrieved_at = None

    documents: list[Document] = []
    missing: list[str] = []
    for entry in manifest.get("documents", []):
        text_path = sources_dir / f"{entry['id']}.txt"
        if not text_path.is_file():
            missing.append(entry["id"])
            continue
        documents.append(
            Document(
                doc_id=entry["id"],
                title=entry["title"],
                source=entry["source"],
                url=entry["url"],
                category=entry.get("category", "environmental_guidance"),
                document_type=entry.get("document_type", "guidance"),
                text=text_path.read_text(encoding="utf-8"),
                jurisdiction=entry.get("jurisdiction", "Global"),
                publication_date=entry.get("publication_date"),
                licence=entry.get("licence", ""),
                retrieved_at=retrieved_at,
            )
        )

    if not documents:
        raise KnowledgeBaseError(
            f"no extracted source text found under {sources_dir}. "
            "Run `python knowledge/build_index.py`."
        )

    return KnowledgeBase(documents=documents, manifest=manifest)
