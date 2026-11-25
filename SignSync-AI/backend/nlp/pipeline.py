"""Member 3 adapter: the functions the backend calls (HOOKS nlp.pipeline.*).

Uses Member 3's Gemini adapter (nlp_translation/member3_adapter.py) when it can be imported.
If Gemini or its packages are not available, it uses the rule-based code in isl_nlp.py, so the
backend still works without an API key.
Speech recognition needs openai-whisper; speech output needs gTTS and internet. Without them the
call raises and the backend's adapter returns its mock response (logged).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

NLP_SRC = Path(__file__).resolve().parents[2] / "nlp_translation"
if str(NLP_SRC) not in sys.path:
    sys.path.insert(0, str(NLP_SRC))

import isl_nlp  # Member 3's rule-based code, unchanged

logger = logging.getLogger("signsync.nlp")
try:
    import member3_adapter as _m3   # Member 3's Gemini-based adapter
except Exception as exc:            # missing google-genai / dotenv, or any import problem
    logger.warning("Member 3 Gemini adapter not available, using rule-based NLP: %s", exc)
    _m3 = None

VOCAB_PATH = Path(os.getenv("ISL_SIGN_VOCAB", NLP_SRC / "sign_vocab.json"))


def _vocab() -> dict:
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def signs_to_sentence(words: list[str]) -> str:
    words = [str(w) for w in words]
    if _m3 is not None:
        return _m3.signs_to_sentence(words)
    return isl_nlp.words_to_sentence(words)


def speech_to_text(audio_bytes: bytes, language: str = "en") -> str:
    if _m3 is not None:
        return _m3.speech_to_text(audio_bytes, language)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(audio_bytes)
        path = handle.name
    try:
        return isl_nlp.speech_to_text(path, language)
    finally:
        Path(path).unlink(missing_ok=True)


def text_to_sign_sequence(text: str) -> list[dict]:
    """Gloss plus clip per item. animation_id is the clip path from sign_vocab.json."""
    if _m3 is not None:
        return _m3.text_to_sign_sequence(text)
    result = isl_nlp.text_to_signs(text, _vocab())
    return [{"label": clip["gloss"], "animation_id": clip["clip"]} for clip in result["clips"]]


def text_to_speech(text: str, language: str = "en") -> bytes:
    if _m3 is not None:
        return _m3.text_to_speech(text, language)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        path = handle.name
    try:
        isl_nlp.text_to_audio_file(text, path, lang=language)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)
