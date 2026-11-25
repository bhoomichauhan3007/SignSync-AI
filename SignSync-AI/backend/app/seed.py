"""
Seed data + CSV import.

  python -m app.seed                 # create tables and seed placeholder signs/lessons/admin
  python -m app.seed --csv data/signs.csv   # import the real label set from the ML dataset

CSV columns: label,category,level,is_dynamic,description,reference_media_url,is_emergency,emergency_phrase
The seeded labels are PLACEHOLDERS - replace them with the real ISL label set from Member 2.
"""
import argparse
import csv
import string

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, SessionLocal, engine
from .models import Gamification, Lesson, Sign, User
from .security import hash_password

WORDS = ["HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", "WATER", "FOOD", "NAME", "HELP"]
PHRASES = ["POLICE", "FIRE", "HOSPITAL", "DANGER", "CALL FAMILY"]
EMERGENCY = {
    "HELP": "I need help.",
    "POLICE": "Please call the police.",
    "FIRE": "There is a fire.",
    "HOSPITAL": "I need to go to the hospital.",
    "DANGER": "I am in danger.",
    "CALL FAMILY": "Please call my family.",
}

LESSONS = [
    ("Alphabet A-M", "Learn the first half of the alphabet.", 1, list("ABCDEFGHIJKLM")),
    ("Alphabet N-Z", "Learn the second half of the alphabet.", 1, list("NOPQRSTUVWXYZ")),
    ("Numbers 0-9", "Learn to sign numbers.", 1, [str(i) for i in range(10)]),
    ("Greetings", "Everyday polite expressions.", 3, ["HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO"]),
    ("Daily Words", "Common words for daily life.", 3, ["WATER", "FOOD", "NAME"]),
    ("Emergency Phrases", "Signs that can save a life.", 4, ["HELP", "POLICE", "FIRE", "HOSPITAL", "DANGER", "CALL FAMILY"]),
]


def _sign(label: str, category: str, level: int, dynamic: bool) -> Sign:
    return Sign(
        label=label,
        category=category,
        level=level,
        is_dynamic=dynamic,
        description=f"Sign for '{label}'.",
        is_emergency=label in EMERGENCY,
        emergency_phrase=EMERGENCY.get(label),
    )


def seed_signs_and_lessons(db: Session) -> None:
    if db.scalar(select(func.count(Sign.id))):
        return
    signs: dict[str, Sign] = {}
    for ch in string.ascii_uppercase:
        signs[ch] = _sign(ch, "alphabet", 1, False)
    for d in range(10):
        signs[str(d)] = _sign(str(d), "number", 1, False)
    for w in WORDS:
        signs[w] = _sign(w, "word", 3, True)
    for p in PHRASES:
        signs[p] = _sign(p, "phrase", 4, True)
    db.add_all(signs.values())
    db.flush()

    for i, (title, desc, level, labels) in enumerate(LESSONS, start=1):
        lesson = Lesson(title=title, description=desc, level=level, order_index=i)
        lesson.signs = [signs[l] for l in labels]
        db.add(lesson)
    db.commit()


def ensure_admin(db: Session) -> None:
    email = settings.admin_email.lower()
    if db.scalar(select(User).where(User.email == email)):
        return
    admin = User(name="Admin", email=email, password_hash=hash_password(settings.admin_password), role="admin")
    db.add(admin)
    db.flush()
    db.add(Gamification(user_id=admin.id, xp=0, streak=0, best_streak=0, badges=[]))
    db.commit()


def seed_database(db: Session) -> None:
    seed_signs_and_lessons(db)
    ensure_admin(db)


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def import_signs_csv(db: Session, path: str) -> tuple[int, int]:
    created = updated = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().upper()
            sign = db.scalar(select(Sign).where(Sign.label == label))
            if sign is None:
                sign = Sign(label=label)
                db.add(sign)
                created += 1
            else:
                updated += 1
            sign.category = (row.get("category") or "word").strip().lower()
            sign.level = int(row.get("level") or 1)
            sign.is_dynamic = _truthy(row.get("is_dynamic", ""))
            sign.description = row.get("description") or None
            sign.reference_media_url = row.get("reference_media_url") or None
            sign.is_emergency = _truthy(row.get("is_emergency", ""))
            sign.emergency_phrase = row.get("emergency_phrase") or None
    db.commit()
    return created, updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Import/update signs from a CSV file")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        seed_database(session)
        if args.csv:
            c, u = import_signs_csv(session, args.csv)
            print(f"Imported signs: {c} created, {u} updated")
        else:
            print("Seed complete.")
