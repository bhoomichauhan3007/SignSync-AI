import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..embeddings.embedder import HashingEmbedder
from ..ingestion.chunking import chunk_documents
from ..ingestion.loaders import iter_documents
from ..retrieval.retriever import RetrievedChunk, Retriever, deduplicate
from ..vector_store.store import FlatNumpyIndex, VectorStore, faiss_available
from ._helpers import make_index, make_kb


def _chunk(text, cid="c", score=0.5):
    return RetrievedChunk(chunk_id=cid, text=text, title="T", source="s.md", page=None,
                          status="sample", score=score)


class FlatNumpyIndexTests(unittest.TestCase):
    def test_returns_best_match_first_and_respects_k(self):
        index = FlatNumpyIndex(3)
        index.add(np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32))
        scores, ids = index.search(np.array([[1, 0, 0]], dtype=np.float32), k=2)
        self.assertEqual(list(ids[0]), [0, 2])
        self.assertGreater(scores[0][0], scores[0][1])

    def test_empty_index_returns_nothing(self):
        index = FlatNumpyIndex(3)
        scores, ids = index.search(np.ones((1, 3), dtype=np.float32), k=5)
        self.assertEqual(scores.shape, (1, 0))


class VectorStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip_keeps_metadata_and_ranking(self):
        kb = make_kb({"sample/a.md": "# Greetings\n\nHello and thank you are common greetings in ISL.",
                      "sample/b.md": "# Emergency\n\nIn an emergency call for help and say the emergency phrase."})
        store, directory = make_index(kb)
        loaded = VectorStore.load(directory)
        self.assertEqual(loaded.chunk_count, store.chunk_count)
        self.assertEqual(loaded.chunks[0]["title"], "Greetings")
        query = loaded.embedder.encode(["emergency help"])
        best = loaded.search(query, 1)[0][1]
        self.assertEqual(best["title"], "Emergency")

    def test_load_missing_index_gives_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            VectorStore.load(Path(tempfile.mkdtemp()))
        self.assertIn("build_index", str(ctx.exception))

    def test_build_refuses_zero_chunks(self):
        with self.assertRaises(ValueError):
            VectorStore.build(HashingEmbedder(), [], backend="numpy")

    @unittest.skipUnless(faiss_available(), "faiss-cpu not installed here; run in your local environment")
    def test_faiss_backend_round_trip(self):
        kb = make_kb({"sample/a.md": "Hello thank you greetings in sign language practice."})
        chunks = chunk_documents(list(iter_documents(kb)))
        store = VectorStore.build(HashingEmbedder(), [c.to_dict() for c in chunks], backend="faiss")
        out = Path(tempfile.mkdtemp())
        store.save(out)
        loaded = VectorStore.load(out)
        self.assertEqual(loaded.index.backend, "faiss")
        self.assertEqual(loaded.chunk_count, store.chunk_count)


class DeduplicateTests(unittest.TestCase):
    def test_identical_text_is_kept_once(self):
        result = deduplicate([_chunk("Same text here.", "a", 0.9), _chunk("same   text here.", "b", 0.8)])
        self.assertEqual([c.chunk_id for c in result], ["a"])

    def test_overlapping_neighbour_is_removed(self):
        first = "Alpha sign advice. " + ("filler words about practice routines " * 6)
        neighbour = first[-200:] + " plus a new sentence that is unique."
        kept = deduplicate([_chunk(first, "a", 0.9), _chunk(neighbour, "b", 0.8)])
        self.assertEqual(len(kept), 1)


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.kb = make_kb({
            "verified/greetings.md": "# Greetings\n\nBasic ISL greetings include hello, good morning and thank you.",
            "sample/emergency.md": "# Emergency\n\nFor emergency help use the phrase help and call 112 in India.",
            "sample/alphabet.md": "# Alphabet\n\nThe manual alphabet is practised with one handshape per letter.",
        })
        self.store, _ = make_index(self.kb)

    def test_returns_sources_with_title_page_score_status(self):
        hits = Retriever(self.store, top_k=2, min_score=0.05).retrieve("emergency help phrase")
        self.assertTrue(hits)
        source = hits[0].to_source()
        self.assertEqual(set(source), {"title", "page", "score", "status", "source"})
        self.assertEqual(source["title"], "Emergency")

    def test_top_k_is_respected(self):
        hits = Retriever(self.store, top_k=1, min_score=0.0).retrieve("hello sign alphabet emergency")
        self.assertEqual(len(hits), 1)

    def test_threshold_filters_weak_matches(self):
        self.assertEqual(Retriever(self.store, min_score=0.99).retrieve("emergency help"), [])

    def test_unrelated_question_returns_nothing_with_default_threshold(self):
        self.assertEqual(Retriever(self.store).retrieve("capital city of france population"), [])

    def test_empty_query_and_empty_store(self):
        self.assertEqual(Retriever(self.store).retrieve("   "), [])
        empty = VectorStore(HashingEmbedder(), FlatNumpyIndex(4096), [])
        self.assertEqual(Retriever(empty).retrieve("hello"), [])


if __name__ == "__main__":
    unittest.main()
