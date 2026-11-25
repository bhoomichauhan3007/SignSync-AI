from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Sign
from . import ai_adapter as ai


def run_prediction(
    db: Session, landmarks: list, mode: str = "static", mock_hint: Optional[str] = None
) -> dict:
    """Predict a sign, apply the confidence threshold and detect emergency signs."""
    raw = ai.predict_sign(landmarks, mode, mock_hint=mock_hint)
    label = str(raw["label"]).upper()
    confidence = float(raw["confidence"])
    low = confidence < settings.confidence_threshold

    sign = db.scalar(select(Sign).where(func.upper(Sign.label) == label))
    is_emergency = bool(sign and sign.is_emergency) and not low
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "low_confidence": low,
        "message": "I'm not confident about this sign. Please repeat." if low else None,
        "top_k": raw.get("top_k", []),
        "is_emergency": is_emergency,
        "emergency_phrase": sign.emergency_phrase if (is_emergency and sign) else None,
        "source": raw.get("source", "mock"),
    }
