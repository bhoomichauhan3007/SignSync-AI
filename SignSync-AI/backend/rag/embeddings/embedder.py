"""Text embedding back-ends with one shared interface.

- SentenceTransformerEmbedder (default target): all-MiniLM-L6-v2 from the sentence-transformers
  package. It captures meaning, so "how do I start learning" can match "beginner lessons".
  Needs `pip install sentence-transformers` (pulls in PyTorch) and downloads the model once.

- HashingEmbedder: dependency-free TF-IDF-style vectoriser. It is LEXICAL: it matches words
  and word pairs, not meaning. It exists so ingestion, indexing, retrieval and the whole API
  can run offline without a model download. It is a development fallback, not a semantic model.

Both return L2-normalised float32 vectors, so the dot product of two vectors equals their
cosine similarity.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HASH_DIM = 4096


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    kind: str = ""
    name: str = ""
    dim: int = 0

    def fit(self, texts: list[str]) -> None:
        """Learn corpus statistics. No-op for pretrained models."""

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        """Restore what fit() learned, so queries are embedded the same way as documents."""

    def spec(self) -> dict:
        return {"kind": self.kind, "model": self.name, "dim": self.dim, "state": self.state()}

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return an (n, dim) float32 array with L2-normalised rows."""


class HashingEmbedder(Embedder):
    kind = "hashing"

    STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
        "of", "to", "in", "on", "at", "for", "with", "and", "or", "but", "if", "it", "its",
        "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "my",
        "your", "can", "how", "what", "which", "who", "whom", "why", "when", "where",
    })
    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = DEFAULT_HASH_DIM):
        self.dim = int(dim)
        self.name = f"hashing-{self.dim}"
        self.idf = np.ones(self.dim, dtype=np.float32)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        for token in cls._TOKEN.findall(text.lower()):
            if token in cls.STOPWORDS:
                continue
            # crude plural folding: "greetings" -> "greeting", "signs" -> "sign"
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return tokens

    @classmethod
    def features(cls, text: str) -> list[str]:
        unigrams = cls.tokenize(text)
        bigrams = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
        return unigrams + bigrams

    def _bucket(self, feature: str) -> int:
        # sha1 (not Python's hash()) so the bucket for a word is identical across runs
        digest = hashlib.sha1(feature.encode("utf-8")).digest()[:4]
        return int.from_bytes(digest, "big") % self.dim

    def _term_counts(self, text: str) -> dict[int, int]:
        counts: dict[int, int] = {}
        for feature in self.features(text):
            bucket = self._bucket(feature)
            counts[bucket] = counts.get(bucket, 0) + 1
        return counts

    def fit(self, texts: list[str]) -> None:
        n = len(texts)
        document_frequency = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            buckets = list(self._term_counts(text).keys())
            if buckets:
                document_frequency[buckets] += 1
        self.idf = (np.log((1.0 + n) / (1.0 + document_frequency)) + 1.0).astype(np.float32)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for bucket, count in self._term_counts(text).items():
                matrix[row, bucket] = (1.0 + math.log(count)) * self.idf[bucket]
        return _normalize_rows(matrix)

    def state(self) -> dict:
        return {"idf": self.idf.tolist()}

    def load_state(self, state: dict) -> None:
        if "idf" in state:
            self.idf = np.asarray(state["idf"], dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    kind = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_ST_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy: heavy optional dependency

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _normalize_rows(np.asarray(vectors, dtype=np.float32))


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def create_embedder(choice: str = "auto", model_name: str = DEFAULT_ST_MODEL,
                    hashing_dim: int = DEFAULT_HASH_DIM) -> Embedder:
    choice = (choice or "auto").lower()
    if choice == "auto":
        choice = "sentence-transformers" if sentence_transformers_available() else "hashing"
    if choice == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if choice == "hashing":
        return HashingEmbedder(hashing_dim)
    raise ValueError(f"Unknown embedder '{choice}'. Use auto, sentence-transformers or hashing.")


def embedder_from_spec(spec: dict) -> Embedder:
    """Rebuild the exact embedder that was used to build an index (stored in manifest.json)."""
    kind = spec.get("kind")
    if kind == "hashing":
        embedder = HashingEmbedder(int(spec["dim"]))
        embedder.load_state(spec.get("state", {}))
        return embedder
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(spec["model"])
    raise ValueError(f"Index was built with unknown embedder kind '{kind}'.")
