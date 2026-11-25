"""
AI adapter: the ONLY place where the backend touches the other members' AI code.

Every function first tries to call the teammate's real implementation (when AI_MODE=real
and the module is importable). If that is not available it falls back to a deterministic
MOCK so the whole backend, frontend and tests work from day one.

To integrate a teammate's code, edit the HOOKS table below (module path + function name)
and drop their package next to the `app/` folder (or add it to PYTHONPATH).
See INTEGRATION_CONTRACT.md for the expected signatures.

FULL-PROJECT NOTE: a real hook that raises when called (e.g. Whisper not installed) is logged
and the mock answer is returned, so the request does not fail with 500.
"""
import hashlib
import importlib
import json
import logging
import re
from typing import Any, Callable, Iterable, Optional

from ..config import settings

logger = logging.getLogger("signsync.ai")

# name -> (module path, function name)   <-- edit these when teammates deliver
HOOKS: dict[str, tuple[str, str]] = {
    "predict_sign": ("ml.sign_model", "predict"),  # Member 2
    "compare_with_reference": ("cv.feedback", "compare_with_reference"),  # Member 1 / 2
    "signs_to_sentence": ("nlp.pipeline", "signs_to_sentence"),  # Member 3
    "speech_to_text": ("nlp.pipeline", "speech_to_text"),  # Member 3
    "text_to_sign_sequence": ("nlp.pipeline", "text_to_sign_sequence"),  # Member 3
    "text_to_speech": ("nlp.pipeline", "text_to_speech"),  # Member 3
    "ask_assistant": ("rag.assistant", "ask"),  # Member 4
}

_cache: dict[str, Optional[Callable]] = {}


def _resolve(name: str) -> Optional[Callable]:
    if settings.ai_mode != "real":
        return None
    if name in _cache:
        return _cache[name]
    module_path, attr = HOOKS[name]
    fn: Optional[Callable] = None
    try:
        fn = getattr(importlib.import_module(module_path), attr)
        logger.info("AI hook '%s' -> %s.%s (real)", name, module_path, attr)
    except (ImportError, AttributeError) as exc:
        logger.warning("AI hook '%s' unavailable (%s). Using mock.", name, exc)
    _cache[name] = fn
    return fn


def status() -> dict:
    """Which hooks are real and which are still mocked (useful during integration)."""
    return {
        "ai_mode": settings.ai_mode,
        "hooks": {name: ("real" if _resolve(name) else "mock") for name in HOOKS},
    }


def reset_cache() -> None:
    _cache.clear()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _h(obj: Any) -> int:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


_MOCK_LABELS = ["HELLO", "THANK YOU", "HELP", "WATER", "FOOD", "YES", "NO", "SORRY", "PLEASE", "NAME"]
_STOPWORDS = {"I", "IS", "ARE", "AM", "THE", "A", "AN", "TO", "OF", "BE"}


# ----------------------------------------------------------------------------
# 1. Sign prediction  (Member 2)
# ----------------------------------------------------------------------------
def predict_sign(landmarks: list, mode: str = "static", mock_hint: Optional[str] = None) -> dict:
    """Return {"label": str, "confidence": float 0..1, "top_k": [...], "source": str}."""
    fn = _resolve("predict_sign")
    if fn:
        try:
            out = fn(landmarks, mode)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                out = {"label": out[0], "confidence": out[1]}
            out = dict(out)
            out.setdefault("top_k", [])
            out["source"] = "real"
            return out
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")

    h = _h(landmarks)
    if mock_hint and (h % 10) < 8:
        label = mock_hint.upper()
    else:
        label = _MOCK_LABELS[h % len(_MOCK_LABELS)]
    confidence = 0.62 + ((h >> 8) % 37) / 100  # 0.62 .. 0.98
    others = [x for x in _MOCK_LABELS if x != label][:2]
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "top_k": [
            {"label": label, "confidence": round(confidence, 4)},
            {"label": others[0], "confidence": round((1 - confidence) * 0.6, 4)},
            {"label": others[1], "confidence": round((1 - confidence) * 0.4, 4)},
        ],
        "source": "mock",
    }


# ----------------------------------------------------------------------------
# 2. Compare attempt with reference sign (Member 1 / 2)
# ----------------------------------------------------------------------------
_HINTS = {
    "hand_position": "Try adjusting where your hand is relative to your body.",
    "hand_shape": "Check your finger positions and hand shape.",
    "movement": "Try repeating the movement more clearly and smoothly.",
    "orientation": "Rotate your palm to match the reference orientation.",
}


def compare_with_reference(landmarks: list, label: str, reference: Optional[list] = None) -> dict:
    """Return {"score": 0..100, "feedback": {aspect: ok|warn|fail}, "hint": str|None, "source": str}."""
    fn = _resolve("compare_with_reference")
    if fn:
        try:
            out = dict(fn(landmarks, label, reference))
            out.setdefault("feedback", {})
            out.setdefault("hint", None)
            out["source"] = "real"
            return out
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")

    h = _h([landmarks, label])
    score = 55 + (h % 45)  # 55 .. 99
    feedback: dict[str, str] = {}
    for i, aspect in enumerate(("hand_position", "hand_shape", "movement", "orientation")):
        v = (h >> (i * 5)) % 100
        feedback[aspect] = "ok" if (score >= 90 or v >= 35) else ("warn" if v >= 15 else "fail")
    hint = next((_HINTS[a] for a, s in feedback.items() if s != "ok"), None)
    return {"score": float(score), "feedback": feedback, "hint": hint, "source": "mock"}


# ----------------------------------------------------------------------------
# 3. NLP / speech (Member 3)
# ----------------------------------------------------------------------------
def signs_to_sentence(words: list[str]) -> dict:
    fn = _resolve("signs_to_sentence")
    if fn:
        try:
            out = fn(words)
            sentence = out["sentence"] if isinstance(out, dict) else str(out)
            return {"sentence": sentence, "source": "real"}
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")
    sentence = " ".join(w.lower() for w in words).strip()
    sentence = (sentence[:1].upper() + sentence[1:]) + ("" if sentence.endswith(("?", ".", "!")) else ".")
    return {"sentence": sentence, "source": "mock"}


def speech_to_text(audio_bytes: bytes, language: str = "en") -> dict:
    fn = _resolve("speech_to_text")
    if fn:
        try:
            out = fn(audio_bytes, language)
            return {"text": out["text"] if isinstance(out, dict) else str(out), "source": "real"}
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")
    return {"text": "hello how are you", "source": "mock"}


def text_to_sign_sequence(text: str, known_labels: Optional[Iterable[str]] = None) -> dict:
    """Return {"sequence": [{"label","animation_id"}], "source"}."""
    fn = _resolve("text_to_sign_sequence")
    if fn:
        try:
            raw = fn(text)
            items = []
            for it in raw:
                if isinstance(it, str):
                    items.append({"label": it.upper(), "animation_id": _slug(it)})
                else:
                    label = str(it["label"]).upper()
                    items.append({"label": label, "animation_id": it.get("animation_id", _slug(label))})
            return {"sequence": items, "source": "real"}
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")

    known = {k.upper() for k in (known_labels or [])}
    words = re.findall(r"[A-Za-z0-9']+", text.upper())
    items, i = [], 0
    while i < len(words):
        matched = False
        for n in (3, 2):
            phrase = " ".join(words[i : i + n])
            if len(words[i : i + n]) == n and phrase in known:
                items.append({"label": phrase, "animation_id": _slug(phrase)})
                i += n
                matched = True
                break
        if matched:
            continue
        if words[i] not in _STOPWORDS:
            items.append({"label": words[i], "animation_id": _slug(words[i])})
        i += 1
    if not items and words:  # everything was a stopword
        items = [{"label": w, "animation_id": _slug(w)} for w in words]
    return {"sequence": items, "source": "mock"}


def text_to_speech(text: str, language: str = "en") -> dict:
    """Return {"audio": bytes|None, "content_type": str, "source": str}."""
    fn = _resolve("text_to_speech")
    if fn:
        try:
            audio = fn(text, language)
            return {"audio": audio, "content_type": "audio/mpeg", "source": "real"}
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")
    return {"audio": None, "content_type": "audio/wav", "source": "mock"}


# ----------------------------------------------------------------------------
# 4. RAG assistant (Member 4)
# ----------------------------------------------------------------------------
def ask_assistant(question: str, history: Optional[list[dict]] = None) -> dict:
    fn = _resolve("ask_assistant")
    if fn:
        try:
            out = fn(question, history or [])
            if isinstance(out, str):
                out = {"answer": out, "sources": []}
            out = dict(out)
            out.setdefault("sources", [])
            out["source"] = "real"
            return out
        except Exception:  # real hook failed at call time: log, then use the mock
            logger.exception("AI hook call failed; falling back to mock")

    q = question.lower()
    if "learn" in q or "start" in q:
        answer = "Start with the alphabet and numbers, then greetings, then everyday words. Practise a few signs daily."
    elif "difference" in q:
        answer = "Compare handshape, orientation, location and movement of the two signs - these four parameters distinguish most signs."
    else:
        answer = "This is a placeholder answer. The RAG assistant will answer from the ISL knowledge base once it is connected."
    return {"answer": answer, "sources": [{"title": "mock-knowledge-base"}], "source": "mock"}
