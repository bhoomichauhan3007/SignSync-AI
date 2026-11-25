import unittest

import numpy as np

from ..embeddings.embedder import (
    HashingEmbedder,
    create_embedder,
    embedder_from_spec,
    sentence_transformers_available,
)


class HashingEmbedderTests(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dim=2048)
        self.corpus = [
            "Learn the manual alphabet first and practise daily.",
            "Emergency phrases help you call for help quickly.",
            "Greetings such as hello and thank you are basic signs.",
        ]
        self.embedder.fit(self.corpus)

    def test_output_shape_and_unit_length(self):
        vectors = self.embedder.encode(self.corpus)
        self.assertEqual(vectors.shape, (3, 2048))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_batch_and_single_encoding_match(self):
        batch = self.embedder.encode(self.corpus)
        single = np.vstack([self.embedder.encode([t]) for t in self.corpus])
        np.testing.assert_allclose(batch, single, atol=1e-6)

    def test_related_text_scores_higher_than_unrelated(self):
        docs = self.embedder.encode(self.corpus)
        query = self.embedder.encode(["how do I call for help in an emergency"])[0]
        scores = docs @ query
        self.assertEqual(int(np.argmax(scores)), 1)

    def test_same_word_always_maps_to_same_bucket(self):
        self.assertEqual(self.embedder._bucket("greeting"), HashingEmbedder(2048)._bucket("greeting"))

    def test_plural_folding_and_stopwords(self):
        self.assertIn("greeting", HashingEmbedder.tokenize("Greetings"))
        self.assertNotIn("the", HashingEmbedder.tokenize("the sign"))

    def test_empty_text_gives_zero_vector_not_nan(self):
        vector = self.embedder.encode(["the of is"])[0]
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertEqual(float(np.linalg.norm(vector)), 0.0)

    def test_state_round_trip_restores_idf(self):
        spec = self.embedder.spec()
        restored = embedder_from_spec(spec)
        np.testing.assert_allclose(restored.encode(["hello sign"]), self.embedder.encode(["hello sign"]))

    def test_create_embedder_hashing_and_unknown_choice(self):
        self.assertEqual(create_embedder("hashing", hashing_dim=512).dim, 512)
        with self.assertRaises(ValueError):
            create_embedder("nonsense")


@unittest.skipUnless(sentence_transformers_available(),
                     "sentence-transformers not installed here; run this test in your local environment")
class SentenceTransformerTests(unittest.TestCase):
    def test_all_minilm_produces_384_dim_normalised_vectors(self):
        embedder = create_embedder("sentence-transformers")
        vectors = embedder.encode(["Hello, how are you?", "Thank you very much."])
        self.assertEqual(vectors.shape, (2, embedder.dim))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
