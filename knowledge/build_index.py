"""Build the AirShield retrieval index.

    Source text (knowledge/sources/*.txt)
        -> load with provenance
        -> chunk (overlapping, paragraph-aware)
        -> embed (semantic model, or the labelled lexical fallback)
        -> persist vector store (knowledge/index/)

Reproducible: run it twice on the same corpus and you get the same chunks. It
refuses to build from an empty corpus rather than shipping an index that would
make the assistant look like it knows things it does not.

    python knowledge/build_index.py
    python knowledge/build_index.py --embedder hashing   # lexical fallback
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "core" / "src"))

from airshield_core.rag.chunking import chunk_document  # noqa: E402
from airshield_core.rag.documents import KnowledgeBaseError, load_knowledge_base  # noqa: E402
from airshield_core.rag.embeddings import build_embedder  # noqa: E402
from airshield_core.rag.vectorstore import VectorStore  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-dir",
        default=str(REPO_ROOT / "knowledge"),
        help="directory containing manifest.json and sources/",
    )
    parser.add_argument(
        "--index-dir",
        default=str(REPO_ROOT / "knowledge" / "index"),
        help="where to write the vector index",
    )
    parser.add_argument(
        "--embedder",
        default="semantic",
        help="'semantic' (default), 'hashing' (lexical fallback), or a model name",
    )
    parser.add_argument(
        "--chunk-chars", type=int, default=900, help="target chunk size in characters"
    )
    parser.add_argument(
        "--overlap-chars", type=int, default=150, help="overlap between chunks"
    )
    args = parser.parse_args(argv)

    try:
        kb = load_knowledge_base(args.knowledge_dir)
    except KnowledgeBaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"loaded {len(kb.documents)} documents ({kb.total_chars:,} characters)")
    print(f"  sources: {', '.join(kb.stats()['sources'])}")

    chunks = []
    for document in kb.documents:
        produced = chunk_document(
            document, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars
        )
        chunks.extend(produced)
        print(f"  {document.doc_id:42s} {len(produced):3d} chunks")

    if not chunks:
        print("ERROR: no chunks were produced; refusing to build an empty index", file=sys.stderr)
        return 1

    print(f"\nembedding {len(chunks)} chunks with '{args.embedder}' ...")
    embedder = build_embedder(args.embedder)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    print(f"  embedder: {embedder.name} ({embedder.dimension}-d, semantic={embedder.is_semantic})")
    if not embedder.is_semantic:
        print("  NOTE: the lexical fallback matches vocabulary, not meaning. Retrieval")
        print("        quality is lower. Install `model2vec` for the semantic backend.")

    store = VectorStore()
    store.add(chunks, vectors)
    store.embedder_name = embedder.name
    store.embedder_is_semantic = embedder.is_semantic
    store.built_at = datetime.now(timezone.utc).isoformat()

    target = store.save(args.index_dir)
    print(f"\nwrote index to {target}")
    print(f"  {store.size} chunks, {len(store.doc_ids())} documents, {store.dimension}-d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
