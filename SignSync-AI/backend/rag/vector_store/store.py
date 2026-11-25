"""Vector index (FAISS or exact numpy search) plus chunk metadata stored on disk.

Index folder layout (backend/rag/index/ by default):
    manifest.json   embedder spec, backend name, dimension, chunk count, timestamp
    chunks.json     one record per chunk: id, text, source, title, page, status
    vectors.npy     numpy backend only
    index.faiss     FAISS backend only

Why two backends?
- FAISS IndexFlatIP is the course-standard exact inner-product index. Use it when faiss-cpu is installed.
- FlatNumpyIndex does the same exact search with a matrix multiply. It gives the same ranking
  and needs no extra package, so the project still runs in a fresh environment.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..embeddings.embedder import Embedder, embedder_from_spec

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.json"
NUMPY_VECTORS_FILE = "vectors.npy"
FAISS_FILE = "index.faiss"


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


class FlatNumpyIndex:
    backend = "numpy"

    def __init__(self, dim: int, vectors: np.ndarray | None = None):
        self.dim = dim
        self.vectors = vectors if vectors is not None else np.zeros((0, dim), dtype=np.float32)

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n_queries = query.shape[0]
        k = min(k, self.count)
        if k == 0:
            return np.zeros((n_queries, 0), np.float32), np.zeros((n_queries, 0), np.int64)
        similarities = query @ self.vectors.T
        order = np.argsort(-similarities, axis=1)[:, :k]
        scores = np.take_along_axis(similarities, order, axis=1)
        return scores, order

    def save(self, directory: Path) -> None:
        np.save(directory / NUMPY_VECTORS_FILE, self.vectors)

    @classmethod
    def load(cls, directory: Path, dim: int) -> "FlatNumpyIndex":
        vectors = np.load(directory / NUMPY_VECTORS_FILE).astype(np.float32)
        return cls(dim, vectors)


class FaissFlatIndex:
    backend = "faiss"

    def __init__(self, dim: int, index=None):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index = index if index is not None else faiss.IndexFlatIP(dim)

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.count)
        if k == 0:
            return np.zeros((query.shape[0], 0), np.float32), np.zeros((query.shape[0], 0), np.int64)
        scores, ids = self.index.search(np.ascontiguousarray(query, dtype=np.float32), k)
        return scores, ids

    def save(self, directory: Path) -> None:
        self._faiss.write_index(self.index, str(directory / FAISS_FILE))

    @classmethod
    def load(cls, directory: Path, dim: int) -> "FaissFlatIndex":
        import faiss

        return cls(dim, faiss.read_index(str(directory / FAISS_FILE)))


def create_index(backend: str, dim: int):
    backend = (backend or "auto").lower()
    if backend == "auto":
        backend = "faiss" if faiss_available() else "numpy"
    if backend == "faiss":
        return FaissFlatIndex(dim)
    if backend == "numpy":
        return FlatNumpyIndex(dim)
    raise ValueError(f"Unknown vector store '{backend}'. Use auto, faiss or numpy.")


def embedding_text(chunk: dict) -> str:
    """Text that gets embedded. The title is included so the document name also matches queries."""
    return f"{chunk['title']}. {chunk['text']}"


class VectorStore:
    def __init__(self, embedder: Embedder, index, chunks: list[dict]):
        if index.count != len(chunks):
            raise ValueError(f"Index has {index.count} vectors but {len(chunks)} chunk records")
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, embedder: Embedder, chunks: list[dict], backend: str = "auto") -> "VectorStore":
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        texts = [embedding_text(c) for c in chunks]
        embedder.fit(texts)               # learns IDF for the hashing embedder; no-op otherwise
        vectors = embedder.encode(texts)
        index = create_index(backend, embedder.dim)
        index.add(vectors)
        return cls(embedder, index, [dict(c) for c in chunks])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[float, dict]]:
        scores, ids = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if int(idx) < 0:
                continue
            results.append((float(score), self.chunks[int(idx)]))
        return results

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.index.save(directory)
        (directory / CHUNKS_FILE).write_text(json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest = {
            "format_version": FORMAT_VERSION,
            "backend": self.index.backend,
            "dim": self.index.dim,
            "chunk_count": self.chunk_count,
            "embedder": self.embedder.spec(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No index found in {directory}. Run: python -m rag.scripts.build_index")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Index format version mismatch. Rebuild the index with build_index.")
        dim = int(manifest["dim"])
        if manifest["backend"] == "faiss":
            if not faiss_available():
                raise RuntimeError("This index was built with FAISS. Install faiss-cpu or rebuild with --vector-store numpy.")
            index = FaissFlatIndex.load(directory, dim)
        else:
            index = FlatNumpyIndex.load(directory, dim)
        chunks = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        embedder = embedder_from_spec(manifest["embedder"])
        return cls(embedder, index, chunks)
