"""Semantic retrieval: question -> embedding -> nearest chunks -> filtered, de-duplicated results.

Steps:
1. Embed the question with the SAME embedder used for the documents.
2. Ask the index for k * candidate_multiplier nearest chunks (over-fetch so dedupe still leaves k).
3. Drop chunks whose score is below min_score (relevance threshold). Better to say
   "I don't know" than to pass weak matches to the LLM.
4. Remove duplicates: identical text, or overlapping neighbour chunks that repeat the same passage.
5. Return the top k with source information.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..vector_store.store import VectorStore

# Thresholds depend on the embedder. Cosine scores from a sentence model are high for
# related text; hashing scores are lower because they only count shared words.
# Override with RAG_MIN_SCORE after checking real scores (see query_cli --show-scores).
DEFAULT_MIN_SCORE = {"sentence-transformers": 0.35, "hashing": 0.10}
FALLBACK_MIN_SCORE = 0.20
CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    source: str
    page: int | None
    status: str
    score: float

    def to_source(self) -> dict:
        return {
            "title": self.title,
            "page": self.page,
            "score": round(self.score, 3),
            "status": self.status,
            "source": self.source,
        }


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the first (highest-scoring) copy of each passage.

    Chunks overlap by design, so a neighbour can contain the start of a chunk already kept.
    We compare the first 80 normalised characters of each text against the kept texts.
    """
    kept: list[RetrievedChunk] = []
    kept_norm: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        norm = _normalise(chunk.text)
        if not norm or norm in seen:
            continue
        head = norm[:80]
        if any(head in previous or previous[:80] in norm for previous in kept_norm):
            continue
        seen.add(norm)
        kept.append(chunk)
        kept_norm.append(norm)
    return kept


class Retriever:
    def __init__(self, store: VectorStore, top_k: int = 4, min_score: float | None = None):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.store = store
        self.top_k = top_k
        if min_score is None:
            min_score = DEFAULT_MIN_SCORE.get(store.embedder.kind, FALLBACK_MIN_SCORE)
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[RetrievedChunk]:
        query = (query or "").strip()
        k = top_k if top_k is not None else self.top_k
        threshold = self.min_score if min_score is None else min_score
        if not query or self.store.chunk_count == 0:
            return []
        query_vector = self.store.embedder.encode([query])
        hits = self.store.search(query_vector, k * CANDIDATE_MULTIPLIER)
        candidates = [
            RetrievedChunk(
                chunk_id=chunk["id"],
                text=chunk["text"],
                title=chunk["title"],
                source=chunk["source"],
                page=chunk.get("page"),
                status=chunk.get("status", "unverified"),
                score=score,
            )
            for score, chunk in hits
            if score >= threshold
        ]
        return deduplicate(candidates)[:k]
