"""Connects Member 3's NLP output to the tutor.

Member 3's `signs_to_speech_result(words)` returns:
    {"words": [...], "sentence": "...", "intent": {"intent": "question", "entities": [...]}, "audio_path": ...}
Their intent labels are: greeting, request, question, emergency, thanks, farewell, statement.

When the sentence is a question or an emergency, the tutor answers it. Other intents are left to
Member 3's own text-to-sign and speech path. This module imports nothing from Member 3, so it runs
without their dependencies (Whisper, gTTS, Anthropic).
"""
from __future__ import annotations

from ..assistant import ask

TUTOR_INTENTS = frozenset({"question", "emergency"})


def intent_label(intent) -> str:
    """Accept either Member 3's dict form or a plain string."""
    if isinstance(intent, dict):
        return str(intent.get("intent", "")).strip().lower()
    return str(intent or "").strip().lower()


def should_ask_tutor(intent) -> bool:
    return intent_label(intent) in TUTOR_INTENTS


def tutor_reply_for_result(result: dict, history: list[dict] | None = None) -> dict | None:
    """Return the tutor's answer for a Member 3 result, or None when the tutor should not answer."""
    sentence = (result or {}).get("sentence", "")
    if not sentence or not should_ask_tutor(result.get("intent")):
        return None
    return ask(sentence, history or [])
