from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import Sign, User
from ..schemas import (
    SentenceRequest,
    SentenceResponse,
    SignSequenceResponse,
    TextToSignRequest,
    TextToSpeechRequest,
)
from ..services import ai_adapter as ai
from ..services.conversation import log_message

router = APIRouter(prefix="/translate", tags=["translate"])


def _build_sequence(db: Session, text: str) -> dict:
    known = set(db.scalars(select(Sign.label)).all())
    out = ai.text_to_sign_sequence(text, known)
    sequence = []
    for item in out["sequence"]:
        label = item["label"]
        is_known = label in known
        sequence.append(
            {
                "label": label,
                "animation_id": item["animation_id"],
                "known": is_known,
                # unknown words fall back to fingerspelling (letters)
                "fingerspell": [] if is_known else [c for c in label if c.isalpha()],
            }
        )
    return {"text": text, "sequence": sequence, "source": out["source"]}


@router.post("/signs-to-sentence", response_model=SentenceResponse)
def signs_to_sentence(
    payload: SentenceRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    out = ai.signs_to_sentence(payload.words)
    log_message(
        db, user, payload.conversation_id, "sign_to_text", out["sentence"],
        {"words": payload.words, "source": out["source"]},
    )
    db.commit()
    return out


@router.post("/text-to-sign", response_model=SignSequenceResponse)
def text_to_sign(
    payload: TextToSignRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = _build_sequence(db, payload.text)
    log_message(
        db, user, payload.conversation_id, "text_to_sign", payload.text,
        {"labels": [s["label"] for s in result["sequence"]], "source": result["source"]},
    )
    db.commit()
    return result


@router.post("/speech-to-sign", response_model=SignSequenceResponse)
def speech_to_sign(
    audio: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    language: str = Form(default="en"),
    conversation_id: Optional[int] = Form(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Speech (audio file) -> text -> sign sequence. For testing you may send `text` instead of `audio`."""
    if audio is None and not text:
        raise HTTPException(status_code=400, detail="Provide either an audio file or text")
    if audio is not None:
        data = audio.file.read()
        if len(data) > settings.max_audio_bytes:
            raise HTTPException(status_code=413, detail="Audio file too large")
        text = ai.speech_to_text(data, language)["text"]
    result = _build_sequence(db, text)
    log_message(
        db, user, conversation_id, "speech_to_sign", text,
        {"labels": [s["label"] for s in result["sequence"]], "source": result["source"]},
    )
    db.commit()
    return result


@router.post("/text-to-speech")
def text_to_speech(payload: TextToSpeechRequest, user: User = Depends(get_current_user)):
    """Returns audio bytes when the TTS module is connected; otherwise JSON telling the
    frontend to use the browser's built-in speechSynthesis."""
    out = ai.text_to_speech(payload.text, payload.language)
    if out["audio"]:
        return Response(content=out["audio"], media_type=out["content_type"])
    return JSONResponse(
        {
            "text": payload.text,
            "audio_available": False,
            "note": "No server-side TTS yet - use window.speechSynthesis in the browser.",
            "source": out["source"],
        }
    )
