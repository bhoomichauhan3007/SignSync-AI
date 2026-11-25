from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import Attempt, Sign, User
from ..schemas import AttemptOut, EvaluateRequest, EvaluateResponse
from ..services import ai_adapter as ai
from ..services import gamification as gm
from ..services.inference import run_prediction
from ..services.progress import update_progress_for_sign

router = APIRouter(prefix="/learn", tags=["learn"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    payload: EvaluateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Learner performs a sign -> AI checks it, gives feedback; XP, streak, badges and progress update."""
    sign = db.get(Sign, payload.sign_id)
    if sign is None:
        raise HTTPException(status_code=404, detail="Sign not found")

    mode = "dynamic" if sign.is_dynamic else "static"
    pred = run_prediction(db, payload.landmarks, mode, mock_hint=sign.label)
    comp = ai.compare_with_reference(payload.landmarks, sign.label, sign.reference_landmarks)

    label_ok = pred["label"].upper() == sign.label.upper()
    is_correct = label_ok and not pred["low_confidence"]
    score = float(comp["score"])
    if not label_ok:
        score = min(score, 40.0)

    attempt = Attempt(
        user_id=user.id,
        sign_id=sign.id,
        predicted_label=pred["label"],
        confidence=pred["confidence"],
        score=score,
        is_correct=is_correct,
        feedback=comp["feedback"],
    )
    db.add(attempt)
    db.flush()

    g = gm.get_or_create(db, user.id)
    gm.touch_streak(g)
    xp_gained = gm.xp_for_attempt(is_correct, score)
    g.xp = (g.xp or 0) + xp_gained
    update_progress_for_sign(db, user.id, sign.id, score)
    new_badges = gm.award_new_badges(db, user.id, g)
    db.commit()

    return EvaluateResponse(
        sign_id=sign.id,
        label=sign.label,
        predicted_label=pred["label"],
        confidence=pred["confidence"],
        low_confidence=pred["low_confidence"],
        is_correct=is_correct,
        score=score,
        feedback=comp["feedback"],
        hint=comp.get("hint"),
        xp_gained=xp_gained,
        xp_total=g.xp,
        level=gm.level_for(g.xp),
        streak=g.streak,
        new_badges=new_badges,
        source=comp.get("source", "mock"),
    )


@router.get("/attempts", response_model=list[AttemptOut])
def my_attempts(
    limit: int = Query(20, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Attempt)
        .where(Attempt.user_id == user.id)
        .order_by(Attempt.id.desc())
        .limit(limit)
    )
    return db.scalars(stmt).all()
