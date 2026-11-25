"""Tests the SEMANTIC SEARCH wiring (sentence-transformers + FAISS) and the GEMINI answer path.

The real packages are not installed in the build environment, so small stand-ins are used
ONLY when the real package is missing. This checks our code (building, saving, loading, searching,
answering). It does NOT show that the real all-MiniLM-L6-v2 model gives good answers: run the
real packages on your computer (see docs/SETUP_SEMANTIC_GEMINI_WHISPER.md) and check with query_cli.
"""
import importlib.util
import json
import pickle
import re
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np

from ..assistant import answer_question
from ..config import RagConfig
from ..embeddings.embedder import SentenceTransformerEmbedder, create_embedder, embedder_from_spec
from ..generation.llm import GeminiProvider
from ..ingestion.chunking import chunk_documents
from ..ingestion.loaders import iter_documents
from ..retrieval.retriever import Retriever
from ..vector_store.store import VectorStore, faiss_available
from ._helpers import make_kb

DIM = 64


def _bucket_vector(text: str) -> np.ndarray:
    """Stand-in 'sentence model': bag of hashed words, normalised. Not semantic - plumbing only."""
    vec = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int.from_bytes(word.encode()[:4].ljust(4, b"_"), "big") % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class _FakeSentenceTransformer:
    def __init__(self, name):
        self.name = name

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.stack([_bucket_vector(t) for t in texts])


class _FakeFaissFlatIP:
    def __init__(self, dim):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self):
        return int(self._vectors.shape[0])

    def add(self, vectors):
        self._vectors = np.vstack([self._vectors, np.asarray(vectors, dtype=np.float32)])

    def search(self, query, k):
        sims = query @ self._vectors.T
        order = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, order, axis=1), order


def _fake_write_index(index, path):
    Path(path).write_bytes(pickle.dumps({"dim": index.dim, "vectors": index._vectors}))


def _fake_read_index(path):
    data = pickle.loads(Path(path).read_bytes())
    idx = _FakeFaissFlatIP(data["dim"])
    idx._vectors = data["vectors"]
    return idx


_injected = []


def setUpModule():
    if importlib.util.find_spec("sentence_transformers") is None:
        st = types.ModuleType("sentence_transformers")
        st.SentenceTransformer = _FakeSentenceTransformer
        sys.modules["sentence_transformers"] = st
        _injected.append("sentence_transformers")
    if importlib.util.find_spec("faiss") is None:
        fa = types.ModuleType("faiss")
        fa.IndexFlatIP = _FakeFaissFlatIP
        fa.write_index = _fake_write_index
        fa.read_index = _fake_read_index
        sys.modules["faiss"] = fa
        _injected.append("faiss")


def tearDownModule():
    for name in _injected:
        sys.modules.pop(name, None)


class SemanticPathTests(unittest.TestCase):
    def setUp(self):
        kb = make_kb({
            "verified/beginners.md": "# Learning ISL\n\nBeginners should start with the manual alphabet and daily practice.",
            "sample/emergency.md": "# Emergency\n\nFor an emergency in India call 112 and ask for help.",
        })
        self.index_dir = Path(tempfile.mkdtemp())
        embedder = create_embedder("sentence-transformers", "fake/minilm")
        chunks = chunk_documents(list(iter_documents(kb)))
        store = VectorStore.build(embedder, [c.to_dict() for c in chunks], backend="faiss")
        store.save(self.index_dir)

    def test_semantic_index_saves_embedder_spec_and_faiss_backend(self):
        manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedder"]["kind"], "sentence-transformers")
        self.assertEqual(manifest["backend"], "faiss")
        self.assertEqual(manifest["dim"], DIM)

    def test_semantic_index_reloads_and_retrieves_the_right_passage(self):
        loaded = VectorStore.load(self.index_dir)
        self.assertIsInstance(loaded.embedder, SentenceTransformerEmbedder)
        hits = Retriever(loaded, top_k=2, min_score=0.05).retrieve("what number do I call in an emergency in India")
        self.assertTrue(hits)
        self.assertEqual(hits[0].title, "Emergency")

    def test_retrieval_returns_source_fields(self):
        loaded = VectorStore.load(self.index_dir)
        hits = Retriever(loaded, top_k=1, min_score=0.05).retrieve("emergency help 112")
        self.assertEqual(set(hits[0].to_source()), {"title", "page", "score", "status", "source"})


class GeminiAnswerTests(unittest.TestCase):
    def test_gemini_answer_is_grounded_and_cites_sources(self):
        kb = make_kb({"verified/beginners.md": "# Learning ISL\n\nBeginners should start with the manual alphabet."})
        from ..embeddings.embedder import HashingEmbedder
        store = VectorStore.build(HashingEmbedder(), [c.to_dict() for c in chunk_documents(list(iter_documents(kb)))], backend="numpy")
        retriever = Retriever(store, top_k=2, min_score=0.05)
        cfg = RagConfig.from_env()
        reply = {"candidates": [{"content": {"parts": [{"text": "Start with the manual alphabet [S1]."}]}}]}
        response = mock.MagicMock()
        response.read.return_value = json.dumps(reply).encode()
        response.__enter__.return_value = response
        provider = GeminiProvider("test-key", "gemini-2.5-flash")
        with mock.patch("urllib.request.urlopen", return_value=response):
            result = answer_question("How do beginners start?", retriever, provider, cfg)
        self.assertEqual(result["mode"], "llm")
        self.assertIn("[S1]", result["answer"])
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["title"], "Learning ISL")


if __name__ == "__main__":
    unittest.main()
