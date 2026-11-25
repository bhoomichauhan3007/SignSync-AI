"""Tests the REAL backend adapter (app/services/ai_adapter.py) without FastAPI.

The adapter only needs `settings.ai_mode`, so a stub `app.config` is installed before import.
"""
import sys
import types
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

_config = types.ModuleType("app.config")


class _Settings:
    ai_mode = "real"


_config.settings = _Settings()
sys.modules["app.config"] = _config

from app.services import ai_adapter  # noqa: E402


class AdapterFallbackTests(unittest.TestCase):
    def setUp(self):
        ai_adapter.reset_cache()

    def test_status_reports_real_hooks_when_modules_import(self):
        status = ai_adapter.status()
        self.assertEqual(status["ai_mode"], "real")
        self.assertEqual(status["hooks"]["predict_sign"], "real")
        self.assertEqual(status["hooks"]["signs_to_sentence"], "real")
        self.assertEqual(status["hooks"]["ask_assistant"], "real")
        self.assertEqual(status["hooks"]["compare_with_reference"], "real")  # cv.feedback now exists (Member 5 hook)

    def test_static_sign_uses_real_model(self):
        from ml.evaluate import DATASET
        import numpy as np
        if not DATASET.exists():
            self.skipTest("dataset missing")
        X = np.load(DATASET, allow_pickle=True)["X"][0].tolist()
        out = ai_adapter.predict_sign(X, "static")
        self.assertEqual(out["source"], "real")
        self.assertRegex(out["label"], r"^[A-Z]$")

    def test_dynamic_sign_falls_back_to_mock_instead_of_failing(self):
        out = ai_adapter.predict_sign([[0.0, 0.0, 0.0]] * 21, "dynamic")
        self.assertEqual(out["source"], "mock")
        self.assertIn("label", out)

    def test_speech_failure_falls_back_to_mock(self):
        out = ai_adapter.speech_to_text(b"fake audio", "en")
        self.assertEqual(out["source"], "mock")

    def test_tutor_answers_through_adapter_with_real_source_flag(self):
        import os
        os.environ.setdefault("RAG_LLM_PROVIDER", "extractive")
        out = ai_adapter.ask_assistant("How can I learn Indian Sign Language?", [])
        self.assertEqual(out["source"], "real")
        self.assertIn("answer", out)
        self.assertIsInstance(out["sources"], list)

    def test_text_to_sign_sequence_real_output_shape(self):
        out = ai_adapter.text_to_sign_sequence("hello water", known_labels=["HELLO", "WATER"])
        self.assertEqual(out["source"], "real")
        self.assertTrue(all({"label", "animation_id"} <= set(item) for item in out["sequence"]))


if __name__ == "__main__":
    unittest.main()
