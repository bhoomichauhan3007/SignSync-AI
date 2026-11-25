from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Attempt, Gamification


def get_or_create(db: Session, user_id: int) -> Gamification:
    g = db.get(Gamification, user_id)
    if g is None:
        g = Gamification(user_id=user_id, xp=0, streak=0, best_streak=0, badges=[])
        db.add(g)
        db.flush()
    return g


def level_for(xp: int) -> int:
    return xp // 100 + 1


def touch_streak(g: Gamification, today: Optional[date] = None) -> None:
    today = today or datetime.now(timezone.utc).date()
    if g.last_active_date == today:
        return
    if g.last_active_date == today - timedelta(days=1):
        g.streak = (g.streak or 0) + 1
    else:
        g.streak = 1
    g.best_streak = max(g.best_streak or 0, g.streak)
    g.last_active_date = today


def xp_for_attempt(is_correct: bool, score: float) -> int:
    if not is_correct:
        return 0
    return 10 + (5 if score >= 90 else 0)


def award_new_badges(db: Session, user_id: int, g: Gamification) -> list[str]:
    correct_count = db.scalar(
        select(func.count())
        .select_from(Attempt)
        .where(Attempt.user_id == user_id, Attempt.is_correct.is_(True))
    ) or 0
    candidates = {
        "first_sign": correct_count >= 1,
        "ten_signs": correct_count >= 10,
        "streak_3": (g.streak or 0) >= 3,
        "streak_7": (g.streak or 0) >= 7,
        "xp_100": (g.xp or 0) >= 100,
        "xp_500": (g.xp or 0) >= 500,
    }
    existing = list(g.badges or [])
    new = [b for b, ok in candidates.items() if ok and b not in existing]
    if new:
        g.badges = existing + new  # reassign so SQLAlchemy detects the JSON change
    return new
