import unittest

import numpy as np

from ml import sign_model
from ml.sign_model import REPO_ROOT, to_feature_vector

DATASET = REPO_ROOT / "data" / "exports" / "alphabet_dataset.npz"


class SignModelTests(unittest.TestCase):
    def test_flat_and_point_inputs_give_same_vector(self):
        points = [[float(i), float(i) + 0.5, 0.25] for i in range(21)]
        flat = [v for p in points for v in p]
        np.testing.assert_allclose(to_feature_vector(points), to_feature_vector(flat))
        dict_points = [{"x": p[0], "y": p[1], "z": p[2]} for p in points]
        np.testing.assert_allclose(to_feature_vector(dict_points), to_feature_vector(flat))

    def test_wrong_shape_and_nan_are_rejected(self):
        with self.assertRaises(ValueError):
            to_feature_vector([[0, 0, 0]] * 5)
        with self.assertRaises(ValueError):
            to_feature_vector([[float("nan"), 0, 0]] * 21)

    def test_word_sequence_must_be_30_frames_of_1662(self):
        from ml.sign_model import to_word_sequence
        good = [[0.0] * 1662 for _ in range(30)]
        self.assertEqual(to_word_sequence(good).shape, (1, 30, 1662))
        flat = [0.0] * (30 * 1662)
        self.assertEqual(to_word_sequence(flat).shape, (1, 30, 1662))
        with self.assertRaises(ValueError):
            to_word_sequence([[0.0] * 1662 for _ in range(10)])
        with self.assertRaises(ValueError):
            to_word_sequence([[0.0] * 100 for _ in range(30)])
        with self.assertRaises(ValueError):
            to_word_sequence([[float("nan")] * 1662 for _ in range(30)])

    def test_real_model_predicts_a_letter_with_probabilities(self):
        if not sign_model.DEFAULT_MODEL.exists():
            self.skipTest("alphabet_mlp.joblib missing")
        if not DATASET.exists():
            self.skipTest(f"training data missing: {DATASET} (extract the project so that data/exports exists)")
        data = np.load(DATASET, allow_pickle=True)
        result = sign_model.predict(data["X"][0].tolist(), "static")
        self.assertRegex(result["label"], r"^[A-Z]$")
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(len(result["top_k"]), 3)

    def test_measured_accuracy_is_recorded_not_assumed(self):
        if not DATASET.exists() or not sign_model.DEFAULT_MODEL.exists():
            self.skipTest("dataset or model missing")
        from ml.evaluate import evaluate
        result = evaluate()
        self.assertEqual(result["samples"], 4633)
        self.assertGreater(result["top1_accuracy"], 0.8)  # measured value is 0.9046 in docs/TESTING_REPORT.md


class NlpPipelineTests(unittest.TestCase):
    def test_sign_sequence_uses_vocabulary_and_fingerspelling(self):
        from nlp.pipeline import text_to_sign_sequence
        items = text_to_sign_sequence("hello water")
        labels = [i["label"] for i in items]
        self.assertIn("HELLO", labels)
        self.assertIn("WATER", labels)
        hello = next(i for i in items if i["label"] == "HELLO")
        self.assertTrue(hello["animation_id"].endswith(".mp4"))

    def test_sentence_without_api_key_uses_rule_fallback(self):
        from nlp.pipeline import signs_to_sentence
        sentence = signs_to_sentence(["WATER", "WANT"])
        self.assertTrue(sentence.endswith("."))
        self.assertIn("water", sentence.lower())

    def test_speech_to_text_fails_cleanly_without_whisper(self):
        try:
            import whisper  # noqa: F401
            self.skipTest("whisper installed; this test checks the missing-package case")
        except ImportError:
            pass
        from nlp.pipeline import speech_to_text
        with self.assertRaises(ImportError):
            speech_to_text(b"RIFF....fake", "en")


if __name__ == "__main__":
    unittest.main()


class WordAssetsTests(unittest.TestCase):
    """Checks Member 2's word files without needing TensorFlow."""

    def test_labels_match_the_vocabulary_exactly(self):
        import json
        from ml.sign_model import WORD_ARTIFACTS
        labels = json.loads((WORD_ARTIFACTS / "label_encoder.json").read_text(encoding="utf-8"))
        names = [labels[str(i)] for i in range(len(labels))]
        self.assertEqual(len(names), 40)
        self.assertEqual(names, sorted(names))
        vocab = json.loads((REPO_ROOT / "nlp_translation" / "sign_vocab.json").read_text(encoding="utf-8"))
        words = {w.lower() for w in vocab if len(w) > 1}
        self.assertEqual(set(names), words)

    def test_normalisation_has_one_value_per_feature(self):
        import numpy as np
        from ml.sign_model import WORD_ARTIFACTS
        with np.load(WORD_ARTIFACTS / "normalization.npz") as stats:
            self.assertEqual(stats["mean"].shape, (1662,))
            self.assertEqual(stats["std"].shape, (1662,))
            self.assertTrue(float(stats["std"].min()) > 0)

    def test_word_model_file_exists(self):
        from ml.sign_model import word_model_path
        self.assertTrue(word_model_path().exists())


class SignFeedbackTests(unittest.TestCase):
    """Member 5 hook cv.feedback.compare_with_reference (letters use real training examples)."""

    def _example(self, letter, split="Testing"):
        import numpy as np
        data = np.load(DATASET, allow_pickle=True)
        idx = np.where((np.asarray(data["y"]).astype(str) == letter) & (np.asarray(data["splits"]).astype(str) == split))[0]
        return data["X"][idx[0]].tolist()

    def test_correct_letter_scores_high_and_wrong_letter_lower(self):
        from cv.feedback import compare_with_reference
        if not DATASET.exists():
            self.skipTest("alphabet dataset missing")
        good = compare_with_reference(self._example("A"), "A")
        bad = compare_with_reference(self._example("A"), "Q")
        self.assertGreater(good["score"], bad["score"])
        self.assertGreaterEqual(good["score"], 70)
        self.assertIn(good["feedback"]["overall"], {"ok", "warn", "fail"})

    def test_words_get_the_documented_fixed_score(self):
        from cv.feedback import compare_with_reference
        out = compare_with_reference([[0.0, 0.0, 0.0]] * 21, "hello")
        self.assertEqual(out["score"], 70.0)
        self.assertIn("word model", out["hint"])

    def test_missing_hand_is_reported_not_crashed(self):
        from cv.feedback import compare_with_reference
        out = compare_with_reference([0.0] * 5, "A")
        self.assertEqual(out["score"], 0.0)
        self.assertIn("hand", out["hint"].lower())
