import random

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import Lesson, QuizResult, Sign, User
from ..schemas import QuizOut, QuizQuestion, QuizResultOut, QuizSubmit
from ..services import gamification as gm

router = APIRouter(prefix="/quiz", tags=["quiz"])

XP_PER_CORRECT_ANSWER = 5


@router.get("/{lesson_id}", response_model=QuizOut)
def get_quiz(lesson_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Multiple-choice quiz: 'Which sign is this?' for each sign of the lesson."""
    lesson = db.get(Lesson, lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lesson not found")
    if not lesson.signs:
        raise HTTPException(status_code=400, detail="Lesson has no signs")

    all_labels = list(db.scalars(select(Sign.label)).all())
    questions = []
    for sign in lesson.signs:
        pool = [l for l in all_labels if l != sign.label]
        options = random.sample(pool, min(3, len(pool))) + [sign.label]
        random.shuffle(options)
        questions.append(
            QuizQuestion(
                sign_id=sign.id,
                prompt="Which sign is this?",
                description=sign.description,
                reference_media_url=sign.reference_media_url,
                options=options,
            )
        )
    random.shuffle(questions)
    return QuizOut(lesson_id=lesson.id, questions=questions)


@router.post("/{lesson_id}/submit", response_model=QuizResultOut)
def submit_quiz(
    lesson_id: int,
    payload: QuizSubmit,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lesson = db.get(Lesson, lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lesson not found")
    by_id = {s.id: s for s in lesson.signs}

    results, score = [], 0
    for ans in payload.answers:
        sign = by_id.get(ans.sign_id)
        if sign is None:
            raise HTTPException(status_code=400, detail=f"Sign {ans.sign_id} is not in this lesson")
        correct = ans.chosen_label.strip().upper() == sign.label.upper()
        score += int(correct)
        results.append({"sign_id": sign.id, "correct": correct, "correct_label": sign.label})

    xp = score * XP_PER_CORRECT_ANSWER
    g = gm.get_or_create(db, user.id)
    gm.touch_streak(g)
    g.xp = (g.xp or 0) + xp
    db.add(QuizResult(user_id=user.id, lesson_id=lesson.id, score=score, total=len(payload.answers)))
    db.flush()
    gm.award_new_badges(db, user.id, g)
    db.commit()
    return QuizResultOut(
        lesson_id=lesson.id, score=score, total=len(payload.answers), xp_gained=xp, results=results
    )
