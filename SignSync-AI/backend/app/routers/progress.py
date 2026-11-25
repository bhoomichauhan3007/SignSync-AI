from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import Attempt, Gamification, Lesson, Progress, Sign, User, lesson_signs
from ..schemas import (
    CategoryProgressOut,
    GamificationOut,
    LeaderboardEntry,
    LessonProgressOut,
    ProgressOut,
    RecommendationOut,
    WeakSign,
)
from ..services import gamification as gm

router = APIRouter(prefix="/progress", tags=["progress"])
leaderboard_router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("/me", response_model=ProgressOut)
def my_progress(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    g = gm.get_or_create(db, user.id)
    db.commit()

    prog_rows = {p.lesson_id: p for p in db.scalars(select(Progress).where(Progress.user_id == user.id))}
    lessons = []
    for lesson in db.scalars(select(Lesson).order_by(Lesson.order_index, Lesson.id)):
        p = prog_rows.get(lesson.id)
        lessons.append(
            LessonProgressOut(
                lesson_id=lesson.id,
                title=lesson.title,
                completion_pct=p.completion_pct if p else 0.0,
                best_score=p.best_score if p else 0.0,
                attempts_count=p.attempts_count if p else 0,
            )
        )

    totals = dict(db.execute(select(Sign.category, func.count(Sign.id)).group_by(Sign.category)).all())
    mastered = dict(
        db.execute(
            select(Sign.category, func.count(func.distinct(Attempt.sign_id)))
            .join(Attempt, Attempt.sign_id == Sign.id)
            .where(Attempt.user_id == user.id, Attempt.is_correct.is_(True))
            .group_by(Sign.category)
        ).all()
    )
    categories = [
        CategoryProgressOut(
            category=cat,
            total=total,
            mastered=mastered.get(cat, 0),
            percent=round(mastered.get(cat, 0) / total * 100, 1) if total else 0.0,
        )
        for cat, total in sorted(totals.items())
    ]

    return ProgressOut(
        gamification=GamificationOut(
            xp=g.xp,
            level=gm.level_for(g.xp),
            streak=g.streak,
            best_streak=g.best_streak,
            badges=list(g.badges or []),
            last_active_date=g.last_active_date,
        ),
        lessons=lessons,
        categories=categories,
    )


@router.get("/recommendations", response_model=RecommendationOut)
def recommendations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Rule-based personalisation: weakest signs + dynamic-vs-static comparison."""
    rows = db.execute(
        select(Attempt.sign_id, func.avg(Attempt.score), func.count(Attempt.id))
        .where(Attempt.user_id == user.id)
        .group_by(Attempt.sign_id)
        .order_by(func.avg(Attempt.score).asc())
        .limit(5)
    ).all()

    if not rows:
        first = db.scalar(select(Lesson.id).order_by(Lesson.order_index, Lesson.id).limit(1))
        return RecommendationOut(
            message="You have not practised yet. Start with the first lesson!",
            weak_signs=[],
            suggested_lesson_id=first,
        )

    labels = {s.id: s.label for s in db.scalars(select(Sign).where(Sign.id.in_([r[0] for r in rows])))}
    weak = [
        WeakSign(sign_id=r[0], label=labels.get(r[0], "?"), avg_score=round(float(r[1]), 1), attempts=r[2])
        for r in rows
    ]

    split = dict(
        db.execute(
            select(Sign.is_dynamic, func.avg(Attempt.score))
            .join(Attempt, Attempt.sign_id == Sign.id)
            .where(Attempt.user_id == user.id)
            .group_by(Sign.is_dynamic)
        ).all()
    )
    message = "Practise these signs again to improve your accuracy."
    if True in split and False in split and split[True] < split[False] - 5:
        message = "You are struggling with dynamic (moving) signs. Practise these signs again."

    suggested = db.scalar(
        select(lesson_signs.c.lesson_id).where(lesson_signs.c.sign_id == weak[0].sign_id).limit(1)
    )
    return RecommendationOut(message=message, weak_signs=weak, suggested_lesson_id=suggested)


@leaderboard_router.get("", response_model=list[LeaderboardEntry])
def leaderboard(
    limit: int = Query(10, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(User.name, Gamification.xp, Gamification.streak)
        .join(Gamification, Gamification.user_id == User.id)
        .order_by(Gamification.xp.desc(), User.id)
        .limit(limit)
    ).all()
    return [
        LeaderboardEntry(rank=i + 1, name=r[0], xp=r[1], level=gm.level_for(r[1]), streak=r[2])
        for i, r in enumerate(rows)
    ]
