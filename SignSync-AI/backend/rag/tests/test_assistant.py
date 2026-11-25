import inspect
import json
import os
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest import mock

from ..assistant import EMPTY_QUESTION_TEXT, answer_question, ask, reset_state
from ..config import RagConfig
from ..generation.llm import (
    GeminiProvider,
    LLMConfigError,
    LLMError,
    LLMProvider,
    OllamaProvider,
    build_provider,
)
from ..generation.prompts import REFUSAL_TEXT, SYSTEM_PROMPT
from ..retrieval.retriever import Retriever
from ._helpers import make_index, make_kb

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.reply


def _config(**overrides):
    base = dict(
        knowledge_base_dir=KB_DIR, index_dir=Path("."), embedder="hashing", embedding_model="",
        vector_store="numpy", top_k=4, min_score=None, chunk_chars=900, chunk_overlap_chars=150,
        max_context_chars=6000, llm_provider="extractive", gemini_api_key=None,
        gemini_model="gemini-2.5-flash", ollama_url="http://localhost:11434", ollama_model="llama3.2",
        llm_timeout_seconds=5.0,
    )
    base.update(overrides)
    return RagConfig(**base)


class _Base(unittest.TestCase):
    def setUp(self):
        self.kb = make_kb({
            "verified/course.md": ("# ISL Course\n\nBeginners should learn the manual alphabet first. "
                                   "Basic greetings include hello and thank you."),
            "sample/emergency.md": ("# Emergency Phrases\n\nIn an emergency, use the phrase help and "
                                    "call 112, the national emergency number in India."),
        })
        self.store, _ = make_index(self.kb)
        self.retriever = Retriever(self.store, top_k=3, min_score=0.05)
        self.cfg = _config()


class ExtractiveModeTests(_Base):
    def test_no_provider_returns_passages_with_sources(self):
        result = answer_question("How do beginners learn the manual alphabet?", self.retriever, None, self.cfg)
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("no AI summary", result["answer"])
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["title"], "ISL Course")

    def test_unverified_sample_material_gets_a_provisional_note(self):
        result = answer_question("emergency phrase help 112", self.retriever, None, self.cfg)
        self.assertIn("not yet checked by a qualified ISL teacher", result["answer"])

    def test_verified_material_does_not_get_the_provisional_note(self):
        result = answer_question("beginners manual alphabet", self.retriever, None, self.cfg)
        self.assertNotIn("provisional", result["answer"])

    def test_no_ai_answer_shows_the_matching_sentence_not_the_whole_chunk(self):
        from ..generation.prompts import format_extractive
        from ..retrieval.retriever import RetrievedChunk
        text = ("## How to start learning ISL. Practise in front of a mirror every day. "
                "Learn the handshapes first, then add location and movement. Eat breakfast before class.")
        chunk = RetrievedChunk("c", text, "ISL basics", "x.md", None, "unverified", 0.5)
        answer = format_extractive([chunk], "How do I start learning ISL?")
        self.assertIn("handshapes", answer)
        self.assertNotIn("breakfast", answer)

    def test_emergency_question_includes_safety_number(self):
        result = answer_question("What do I do in an emergency?", self.retriever, None, self.cfg)
        self.assertIn("112", result["answer"])

    def test_off_topic_question_is_refused_without_calling_llm(self):
        provider = FakeProvider(reply="should not be used")
        result = answer_question("What is the capital of France?", self.retriever, provider, self.cfg)
        self.assertEqual(result["mode"], "refused")
        self.assertEqual(result["answer"], REFUSAL_TEXT)
        self.assertEqual(result["sources"], [])
        self.assertEqual(provider.calls, [])

    def test_sign_video_question_gets_limitation_note(self):
        result = answer_question("What does this sign mean? Dragon fruit", self.retriever, None, self.cfg)
        self.assertIn("cannot see videos", result["answer"])  # note is added even when passages match
        refused = answer_question("What does this sign mean? Quantum chromodynamics", self.retriever, None, self.cfg)
        self.assertEqual(refused["mode"], "refused")
        self.assertIn("cannot see videos", refused["answer"])

    def test_empty_question(self):
        result = answer_question("   ", self.retriever, None, self.cfg)
        self.assertEqual(result["answer"], EMPTY_QUESTION_TEXT)
        self.assertEqual(result["mode"], "refused")


class LlmModeTests(_Base):
    def test_llm_receives_grounded_prompt_and_strict_rules(self):
        provider = FakeProvider(reply="Beginners start with the manual alphabet [S1].")
        result = answer_question("How should beginners start?", self.retriever, provider, self.cfg)
        self.assertEqual(result["mode"], "llm")
        self.assertEqual(provider.calls[0][0], SYSTEM_PROMPT)
        user_prompt = provider.calls[0][1]
        self.assertIn("[S1]", user_prompt)
        self.assertIn("manual alphabet", user_prompt)
        self.assertIn("How should beginners start?", user_prompt)
        self.assertIn("Beginners start with the manual alphabet [S1].", result["answer"])

    def test_llm_failure_falls_back_to_passages_with_reason(self):
        provider = FakeProvider(error=LLMError("The Gemini free-tier quota or rate limit has been reached."))
        result = answer_question("How should beginners start?", self.retriever, provider, self.cfg)
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("quota", result["answer"])
        self.assertIn("Showing the matching passages instead", result["answer"])
        self.assertTrue(result["sources"])

    def test_missing_key_notice_is_shown_in_extractive_fallback(self):
        result = answer_question("beginners manual alphabet", self.retriever, None, self.cfg,
                                 notice="The AI answer service is not configured (GEMINI_API_KEY is not set).")
        self.assertIn("GEMINI_API_KEY", result["answer"])

    def test_history_is_passed_to_the_llm(self):
        provider = FakeProvider(reply="Yes [S1].")
        history = [{"role": "user", "content": "I am a beginner."}, {"role": "assistant", "content": "Welcome."}]
        answer_question("Then what should I learn?", self.retriever, provider, self.cfg, history=history)
        self.assertIn("User: I am a beginner.", provider.calls[0][1])


class GeminiProviderTests(unittest.TestCase):
    def test_missing_key_raises_config_error_without_network(self):
        with self.assertRaises(LLMConfigError):
            GeminiProvider(None, "gemini-2.5-flash")

    def _fake_response(self, payload):
        response = mock.MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.__enter__.return_value = response
        return response

    def test_parses_successful_response(self):
        payload = {"candidates": [{"content": {"parts": [{"text": "Learn the alphabet first."}]}}]}
        with mock.patch("urllib.request.urlopen", return_value=self._fake_response(payload)) as urlopen:
            text = GeminiProvider("test-key", "gemini-2.5-flash").generate("sys", "user")
        self.assertEqual(text, "Learn the alphabet first.")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.headers["X-goog-api-key"], "test-key")  # key sent as header, not in URL
        self.assertNotIn("test-key", request.full_url)

    def test_quota_error_gives_friendly_message_without_key(self):
        error = urllib.error.HTTPError("url", 429, "Too Many Requests", {}, BytesIO(b""))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(LLMError) as ctx:
                GeminiProvider("secret-key-123", "gemini-2.5-flash").generate("s", "u")
        self.assertIn("quota", str(ctx.exception))
        self.assertNotIn("secret-key-123", str(ctx.exception))

    def test_blocked_response_is_reported(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=self._fake_response({"promptFeedback": {"blockReason": "SAFETY"}})):
            with self.assertRaises(LLMError) as ctx:
                GeminiProvider("k", "m").generate("s", "u")
        self.assertIn("SAFETY", str(ctx.exception))

    def test_network_failure_is_reported(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(LLMError):
                GeminiProvider("k", "m").generate("s", "u")

    def test_auto_uses_gemini_only_when_key_is_present(self):
        self.assertIsNone(build_provider(_config(llm_provider="auto", gemini_api_key=None)))
        self.assertIsInstance(build_provider(_config(llm_provider="auto", gemini_api_key="k")), GeminiProvider)

    def test_build_provider_selection(self):
        self.assertIsNone(build_provider(_config(llm_provider="extractive")))
        self.assertIsInstance(build_provider(_config(llm_provider="ollama")), OllamaProvider)
        with self.assertRaises(LLMConfigError):
            build_provider(_config(llm_provider="gemini", gemini_api_key=None))
        with self.assertRaises(LLMConfigError):
            build_provider(_config(llm_provider="unknown"))


class OllamaProviderTests(unittest.TestCase):
    def test_connection_failure_message(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(LLMError) as ctx:
                OllamaProvider("http://localhost:11434", "llama3.2").generate("s", "u")
        self.assertIn("ollama serve", str(ctx.exception))

    def test_parses_chat_response(self):
        response = mock.MagicMock()
        response.read.return_value = json.dumps({"message": {"content": "Hello [S1]."}}).encode()
        response.__enter__.return_value = response
        with mock.patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(OllamaProvider("http://x", "m").generate("s", "u"), "Hello [S1].")


class ContractAndEntryPointTests(unittest.TestCase):
    """Checks the exact interface the backend adapter relies on."""

    def setUp(self):
        self._old_env = dict(os.environ)
        reset_state()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)
        reset_state()

    def test_ask_signature_matches_adapter_call(self):
        params = list(inspect.signature(ask).parameters)
        self.assertEqual(params, ["question", "history"])

    def test_missing_index_returns_clear_message_not_exception(self):
        empty_dir = Path(make_kb({}))
        os.environ["RAG_INDEX_DIR"] = str(empty_dir / "no_index_here")
        os.environ["RAG_LLM_PROVIDER"] = "extractive"
        result = ask("How do I learn ISL?", [])
        self.assertEqual(result["sources"], [])
        self.assertIn("build_index", result["answer"])
        self.assertEqual(result["mode"], "unavailable")

    def test_end_to_end_with_real_sample_knowledge_base(self):
        """Build the real knowledge_base/ into a temp index and ask through the public entry point."""
        from ..ingestion.chunking import chunk_documents
        from ..ingestion.loaders import iter_documents
        from ..embeddings.embedder import HashingEmbedder
        from ..vector_store.store import VectorStore

        chunks = chunk_documents(list(iter_documents(KB_DIR)))
        store = VectorStore.build(HashingEmbedder(), [c.to_dict() for c in chunks], backend="numpy")
        index_dir = Path(make_kb({}))
        store.save(index_dir)
        os.environ["RAG_INDEX_DIR"] = str(index_dir)
        os.environ["RAG_LLM_PROVIDER"] = "extractive"
        os.environ["RAG_MIN_SCORE"] = "0.05"
        os.environ["GEMINI_API_KEY"] = ""  # ensure no key is picked up by accident
        reset_state()

        result = ask("What is the national emergency number in India?", [])
        self.assertEqual(set(result), {"answer", "sources", "mode"})
        self.assertTrue(result["sources"])
        self.assertEqual(set(result["sources"][0]), {"title", "page", "score", "status", "source"})
        self.assertIn("112", result["answer"])
        self.assertEqual(result["sources"][0]["status"], "sample")

        refusal = ask("Who won the cricket world cup in 1983?", [])
        self.assertEqual(refusal["mode"], "refused")
        self.assertEqual(refusal["sources"], [])


if __name__ == "__main__":
    unittest.main()
