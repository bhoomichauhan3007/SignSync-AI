"""SignSync AI Tutor: the function the backend calls.

Contract (see backend/app/services/ai_adapter.py and app/schemas.py AssistantResponse):

    ask(question: str, history: list[dict] | None = None) -> dict
    returns {"answer": str, "sources": [{"title", "page", "score", "status", "source"}], "mode": str}

`mode` is "llm", "extractive", "refused" or "unavailable". The backend's response schema only
keeps answer and sources; mode is for logs and tests.

Flow: retrieve passages -> if none pass the threshold, refuse (no LLM call)
      -> build a grounded prompt -> call the LLM -> on any LLM failure, fall back to showing
         the retrieved passages, so the student always gets the verified text.
"""
from __future__ import annotations

import logging
import re
import threading

from .config import RagConfig
from .generation.llm import LLMConfigError, LLMError, LLMProvider, build_provider
from .generation.prompts import (
    EMERGENCY_NOTE,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    UNVERIFIED_NOTE,
    VIDEO_LIMITATION_NOTE,
    build_user_prompt,
    format_extractive,
)
from .retrieval.retriever import Retriever, RetrievedChunk
from .vector_store.store import VectorStore

logger = logging.getLogger("signsync.rag")

NOT_BUILT_TEXT = ("The ISL knowledge base has not been indexed yet. In the backend folder run "
                  "`python -m rag.scripts.build_index`, then ask again.")
EMPTY_QUESTION_TEXT = "Please type a question about Indian Sign Language."
EMERGENCY_WORDS = ("emergency", "ambulance", "police", "fire", "bleeding", "injured", "accident", "hospital", "danger")
_lock = threading.Lock()
_state: dict = {"loaded": False, "retriever": None, "provider": None, "cfg": None,
                "error": None, "notice": None}


def _is_emergency(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in EMERGENCY_WORDS)


def _mentions_video(question: str) -> bool:
    lowered = question.lower()
    return any(phrase in lowered for phrase in ("this sign", "that sign", "this gesture", "my sign"))


def _join(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _result(answer: str, sources: list[RetrievedChunk], mode: str) -> dict:
    return {
        "answer": answer,
        "sources": [source.to_source() for source in sources],
        "mode": mode,
    }


def answer_question(question: str, retriever: Retriever, provider: LLMProvider | None,
                    cfg: RagConfig, history: list[dict] | None = None,
                    notice: str | None = None) -> dict:
    """Pure function (no global state) so tests can inject a fake retriever and provider."""
    question = (question or "").strip()
    if not question:
        return _result(EMPTY_QUESTION_TEXT, [], "refused")

    prefix: list[str] = []
    if _is_emergency(question):
        prefix.append(EMERGENCY_NOTE)
    if _mentions_video(question):
        prefix.append(VIDEO_LIMITATION_NOTE)
    passages = retriever.retrieve(question)

    if not passages:
        return _result(_join(prefix + [REFUSAL_TEXT]), [], "refused")

    mode = "extractive"
    sources = passages
    body: str | None = None

    if provider is not None:
        user_prompt, included = build_user_prompt(question, passages, history, cfg.max_context_chars)
        try:
            body = provider.generate(SYSTEM_PROMPT, user_prompt).strip()
            mode = "llm"
            sources = included
            if "[S" not in body:
                logger.warning("LLM answer has no [S#] citation; check the prompt or model.")
        except LLMError as exc:
            logger.warning("LLM unavailable, falling back to passages: %s", exc)
            notice = str(exc)

    if body is None:
        body = format_extractive(passages, question)
        if notice:
            body = f"{notice} Showing the matching passages instead.\n\n{body}"

    # Gemini answers: the app shows the status of each source as a label, so no extra note is added.
    # Passage-only answers have no writer, so the note is added here.
    if mode == "extractive" and not any(source.status == "verified" for source in sources):
        body = _join([body, UNVERIFIED_NOTE])

    return _result(_join(prefix + [body]), sources, mode)


def _ensure_loaded() -> None:
    with _lock:
        if _state["loaded"]:
            return
        cfg = RagConfig.from_env()
        _state["cfg"] = cfg
        try:
            store = VectorStore.load(cfg.index_dir)
            _state["retriever"] = Retriever(store, top_k=cfg.top_k, min_score=cfg.min_score)
        except FileNotFoundError:
            _state["error"] = NOT_BUILT_TEXT
        except Exception as exc:  # corrupt or mismatched index: report, do not crash the API
            logger.exception("Could not load the RAG index")
            _state["error"] = (f"The knowledge base index could not be loaded ({exc}). "
                               "Rebuild it with `python -m rag.scripts.build_index`.")
        try:
            _state["provider"] = build_provider(cfg)
        except LLMConfigError as exc:
            logger.warning("LLM not configured, using extractive mode: %s", exc)
            _state["provider"] = None
            _state["notice"] = f"The AI answer service is not configured: {exc}"
        _state["loaded"] = True


def get_retriever():
    """(retriever, error_message). Used by the CLI for retrieval-only testing."""
    _ensure_loaded()
    return _state["retriever"], _state["error"]


def reset_state() -> None:
    """Forget loaded components so the next call re-reads the environment and index (tests, reloads)."""
    with _lock:
        _state.update({"loaded": False, "retriever": None, "provider": None, "cfg": None,
                       "error": None, "notice": None})


def ask(question: str, history: list[dict] | None = None) -> dict:
    """Entry point used by app/services/ai_adapter.py (HOOKS['ask_assistant'] -> rag.assistant.ask)."""
    _ensure_loaded()
    if _state["error"]:
        return _result(_state["error"], [], "unavailable")
    return answer_question(
        question,
        _state["retriever"],
        _state["provider"],
        _state["cfg"],
        history,
        notice=_state["notice"],
    )
