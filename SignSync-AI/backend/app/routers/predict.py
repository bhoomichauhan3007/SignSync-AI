from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import PredictRequest, PredictResponse
from ..services.conversation import log_message
from ..services.inference import run_prediction

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/sign", response_model=PredictResponse)
def predict_sign(
    payload: PredictRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sign -> label + confidence. Handles low-confidence and emergency signs."""
    result = run_prediction(db, payload.landmarks, payload.mode)
    if payload.conversation_id is not None and not result["low_confidence"]:
        log_message(
            db,
            user,
            payload.conversation_id,
            "sign_to_text",
            result["label"],
            {"confidence": result["confidence"], "source": result["source"]},
        )
        db.commit()
    return result
