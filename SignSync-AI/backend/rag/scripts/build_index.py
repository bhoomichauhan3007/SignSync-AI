"""Build the vector index from the knowledge base folder.

Usage (run from the backend folder):
    python -m rag.scripts.build_index
    python -m rag.scripts.build_index --embedder hashing --vector-store numpy   # offline, no model download
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ..config import RagConfig
from ..embeddings.embedder import create_embedder
from ..ingestion.chunking import chunk_documents
from ..ingestion.loaders import iter_documents
from ..vector_store.store import VectorStore


def main(argv: list[str] | None = None) -> int:
    cfg = RagConfig.from_env()
    parser = argparse.ArgumentParser(description="Build the SignSync AI knowledge-base index.")
    parser.add_argument("--kb", type=Path, default=cfg.knowledge_base_dir, help="knowledge base folder")
    parser.add_argument("--index", type=Path, default=cfg.index_dir, help="output index folder")
    parser.add_argument("--embedder", default=cfg.embedder, help="auto | sentence-transformers | hashing")
    parser.add_argument("--model", default=cfg.embedding_model, help="sentence-transformers model name")
    parser.add_argument("--vector-store", default=cfg.vector_store, help="auto | faiss | numpy")
    parser.add_argument("--chunk-chars", type=int, default=cfg.chunk_chars)
    parser.add_argument("--overlap-chars", type=int, default=cfg.chunk_overlap_chars)
    args = parser.parse_args(argv)

    print(f"Reading documents from: {args.kb}")
    try:
        documents = list(iter_documents(args.kb))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not documents:
        print("ERROR: no supported documents (.pdf, .txt, .md) found. Add files to knowledge_base/verified, "
              "knowledge_base/sample or knowledge_base/unverified.")
        return 1

    chunks = chunk_documents(documents, args.chunk_chars, args.overlap_chars)
    if not chunks:
        print("ERROR: documents contained no usable text (scanned PDFs need OCR first).")
        return 1

    per_status = Counter(c.status for c in chunks)
    per_source = Counter(c.source for c in chunks)
    print(f"Documents: {len({d.source for d in documents})} file(s), {len(documents)} page/section unit(s)")
    print(f"Chunks: {len(chunks)} total | by status: {dict(per_status)}")
    for source, count in sorted(per_source.items()):
        print(f"  - {source}: {count} chunk(s)")

    embedder = create_embedder(args.embedder, args.model)
    print(f"Embedder: {embedder.kind} ({embedder.name}), dimension {embedder.dim}")
    store = VectorStore.build(embedder, [c.to_dict() for c in chunks], backend=args.vector_store)
    store.save(args.index)
    print(f"Vector store: {store.index.backend}, {store.chunk_count} vectors")
    print(f"Index saved to: {args.index}")
    if embedder.kind == "hashing":
        print("NOTE: hashing embedder matches shared words only. Install sentence-transformers for semantic search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
