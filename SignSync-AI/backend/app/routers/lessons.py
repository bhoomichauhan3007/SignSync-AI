from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..database import get_db
from ..deps import require_admin
from ..models import Lesson, Sign
from ..schemas import LessonCreate, LessonDetail, LessonOut, LessonSignsUpdate, SignOut

router = APIRouter(prefix="/lessons", tags=["lessons"])


def _out(lesson: Lesson) -> LessonOut:
    return LessonOut(
        id=lesson.id,
        title=lesson.title,
        description=lesson.description,
        level=lesson.level,
        order_index=lesson.order_index,
        sign_count=len(lesson.signs),
    )


def _detail(lesson: Lesson) -> LessonDetail:
    base = _out(lesson)
    return LessonDetail(**base.model_dump(), signs=[SignOut.model_validate(s) for s in lesson.signs])


@router.get("", response_model=list[LessonOut])
def list_lessons(db: Session = Depends(get_db)):
    lessons = db.scalars(
        select(Lesson).options(selectinload(Lesson.signs)).order_by(Lesson.order_index, Lesson.id)
    ).all()
    return [_out(l) for l in lessons]


@router.get("/{lesson_id}", response_model=LessonDetail)
def get_lesson(lesson_id: int, db: Session = Depends(get_db)):
    lesson = db.get(Lesson, lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return _detail(lesson)


@router.post("", response_model=LessonDetail, status_code=201, dependencies=[Depends(require_admin)])
def create_lesson(payload: LessonCreate, db: Session = Depends(get_db)):
    lesson = Lesson(
        title=payload.title,
        description=payload.description,
        level=payload.level,
        order_index=payload.order_index,
    )
    if payload.sign_ids:
        lesson.signs = list(db.scalars(select(Sign).where(Sign.id.in_(payload.sign_ids))).all())
    db.add(lesson)
    db.commit()
    db.refresh(lesson)
    return _detail(lesson)


@router.put("/{lesson_id}/signs", response_model=LessonDetail, dependencies=[Depends(require_admin)])
def set_lesson_signs(lesson_id: int, payload: LessonSignsUpdate, db: Session = Depends(get_db)):
    lesson = db.get(Lesson, lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail="Lesson not found")
    lesson.signs = list(db.scalars(select(Sign).where(Sign.id.in_(payload.sign_ids))).all())
    db.commit()
    db.refresh(lesson)
    return _detail(lesson)
