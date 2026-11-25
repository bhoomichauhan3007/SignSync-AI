from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Attempt, Lesson, Progress, Sign, lesson_signs


def update_progress_for_sign(db: Session, user_id: int, sign_id: int, score: float) -> None:
    """Recalculate progress for every lesson that contains this sign. Call after flushing the attempt."""
    lessons = db.scalars(
        select(Lesson).join(lesson_signs, lesson_signs.c.lesson_id == Lesson.id).where(
            lesson_signs.c.sign_id == sign_id
        )
    ).all()
    for lesson in lessons:
        sign_ids = [s.id for s in lesson.signs]
        if not sign_ids:
            continue
        done = db.scalar(
            select(func.count(func.distinct(Attempt.sign_id))).where(
                Attempt.user_id == user_id,
                Attempt.is_correct.is_(True),
                Attempt.sign_id.in_(sign_ids),
            )
        ) or 0
        prog = db.scalar(
            select(Progress).where(Progress.user_id == user_id, Progress.lesson_id == lesson.id)
        )
        if prog is None:
            prog = Progress(
                user_id=user_id,
                lesson_id=lesson.id,
                completion_pct=0.0,
                best_score=0.0,
                attempts_count=0,
            )
            db.add(prog)
        prog.completion_pct = round(done / len(sign_ids) * 100, 1)
        prog.best_score = max(prog.best_score or 0.0, float(score))
        prog.attempts_count = (prog.attempts_count or 0) + 1
