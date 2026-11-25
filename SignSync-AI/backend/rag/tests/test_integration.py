import os
import unittest
from pathlib import Path

from ..assistant import reset_state
from ..integration.nlp_bridge import intent_label, should_ask_tutor, tutor_reply_for_result
from ..scripts.evaluate import evaluate
from ..retrieval.retriever import Retriever
from ._helpers import make_index, make_kb


class NlpBridgeTests(unittest.TestCase):
    def test_intent_forms(self):
        self.assertEqual(intent_label({"intent": "Question", "entities": []}), "question")
        self.assertEqual(intent_label("emergency"), "emergency")
        self.assertEqual(intent_label(None), "")

    def test_only_question_and_emergency_go_to_tutor(self):
        self.assertTrue(should_ask_tutor({"intent": "question"}))
        self.assertTrue(should_ask_tutor({"intent": "emergency"}))
        for label in ["greeting", "request", "thanks", "farewell", "statement"]:
            self.assertFalse(should_ask_tutor({"intent": label}), label)

    def test_non_question_returns_none_without_calling_tutor(self):
        result = {"words": ["HELLO"], "sentence": "Hello.", "intent": {"intent": "greeting"}}
        self.assertIsNone(tutor_reply_for_result(result))

    def test_question_is_answered_by_tutor(self):
        os.environ["RAG_LLM_PROVIDER"] = "extractive"
        os.environ["RAG_MIN_SCORE"] = "0.05"
        kb = make_kb({"sample/info.md": "# Info\n\nThe emergency number in India is 112 for urgent help."})
        _, index_dir = make_index(kb)
        os.environ["RAG_INDEX_DIR"] = str(index_dir)
        reset_state()
        try:
            result = {"words": ["WATER"], "sentence": "What is the emergency number in India?",
                      "intent": {"intent": "question", "entities": []}}
            reply = tutor_reply_for_result(result)
            self.assertIsNotNone(reply)
            self.assertIn("112", reply["answer"])
        finally:
            reset_state()
            os.environ.pop("RAG_INDEX_DIR", None)


class EvaluationTests(unittest.TestCase):
    def test_metrics_computed_from_run(self):
        kb = make_kb({"sample/info.md": "# Info\n\nIndian Sign Language uses handshape and movement."})
        store, _ = make_index(kb)
        retriever = Retriever(store, top_k=2, min_score=0.05)
        questions = [
            {"question": "What is Indian Sign Language?", "expect_source": "sample/info.md"},
            {"question": "Who won the world cup?", "expect_source": None},
        ]
        report = evaluate(questions, retriever)
        self.assertEqual(report["answerable"], 1)
        self.assertEqual(report["source_hit_rate"], 1.0)
        self.assertEqual(report["refusal_accuracy"], 1.0)
        self.assertEqual(len(report["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
