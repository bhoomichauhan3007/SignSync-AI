
"""
Member 3: Gemini NLP adapter.
Keeps the existing backend function names and output formats.
Does not modify isl_nlp.py or backend files.
"""

import tempfile
from pathlib import Path

import isl_nlp
from gemini_nlp import words_to_sentence as gemini_sentence
from gemini_nlp import text_to_gloss as gemini_gloss


def _vocab():
    """Load the existing sign vocabulary."""
    vocab_path = Path(__file__).resolve().parent / "sign_vocab.json"

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Sign vocabulary not found: {vocab_path}"
        )

    import json
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def signs_to_sentence(words: list[str]) -> str:
    """Convert recognized ISL words into a natural sentence."""
    return gemini_sentence([str(w) for w in words])


def speech_to_text(audio_bytes: bytes, language: str = "en") -> str:
    """Convert uploaded speech audio into text using Whisper."""
    with tempfile.NamedTemporaryFile(
        suffix=".m4a", delete=False
    ) as handle:
        handle.write(audio_bytes)
        path = handle.name

    try:
        return isl_nlp.speech_to_text(path, language)
    finally:
        Path(path).unlink(missing_ok=True)


def text_to_sign_sequence(text: str) -> list[dict]:
    """Convert text into ISL glosses and their corresponding clips."""
    vocab = _vocab()

    gloss = gemini_gloss(text, vocab)
    clips = isl_nlp.gloss_to_clips(gloss, vocab)

    return [
        {
            "label": clip["gloss"],
            "animation_id": clip["clip"]
        }
        for clip in clips
    ]


def text_to_speech(text: str, language: str = "en") -> bytes:
    """Convert text into speech audio using gTTS."""
    with tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False
    ) as handle:
        path = handle.name

    try:
        isl_nlp.text_to_audio_file(text, path, lang=language)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)

# Member 2 → Member 3 integration

def transformer_prediction_to_sentence(predicted_word: str) -> str:
    """
    Receives a recognized word from Member 2's Transformer
    and converts it into a natural sentence.
    """
    if not predicted_word:
        return ""

    return signs_to_sentence([predicted_word])