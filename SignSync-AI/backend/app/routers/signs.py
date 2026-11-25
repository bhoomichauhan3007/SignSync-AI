from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import require_admin
from ..models import Sign
from ..schemas import SignCreate, SignOut

router = APIRouter(prefix="/signs", tags=["signs"])


def _apply(sign: Sign, data: SignCreate) -> None:
    sign.label = data.label.strip().upper()
    sign.category = data.category.strip().lower()
    sign.level = data.level
    sign.is_dynamic = data.is_dynamic
    sign.description = data.description
    sign.reference_media_url = data.reference_media_url
    sign.reference_landmarks = data.reference_landmarks
    sign.is_emergency = data.is_emergency
    sign.emergency_phrase = data.emergency_phrase


@router.get("", response_model=list[SignOut])
def list_signs(
    category: Optional[str] = None,
    level: Optional[int] = None,
    q: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Sign).order_by(Sign.id)
    if category:
        stmt = stmt.where(Sign.category == category.lower())
    if level is not None:
        stmt = stmt.where(Sign.level == level)
    if q:
        stmt = stmt.where(Sign.label.ilike(f"%{q}%"))
    return db.scalars(stmt.offset(offset).limit(limit)).all()


@router.post("", response_model=SignOut, status_code=201, dependencies=[Depends(require_admin)])
def create_sign(payload: SignCreate, db: Session = Depends(get_db)):
    label = payload.label.strip().upper()
    if db.scalar(select(Sign).where(Sign.label == label)):
        raise HTTPException(status_code=409, detail="Sign already exists")
    sign = Sign()
    _apply(sign, payload)
    db.add(sign)
    db.commit()
    db.refresh(sign)
    return sign


@router.post("/bulk", response_model=dict, dependencies=[Depends(require_admin)])
def bulk_upsert(payload: list[SignCreate], db: Session = Depends(get_db)):
    """Create or update many signs at once (e.g. the label set from the ML dataset)."""
    created = updated = 0
    for item in payload:
        label = item.label.strip().upper()
        sign = db.scalar(select(Sign).where(Sign.label == label))
        if sign is None:
            sign = Sign()
            db.add(sign)
            created += 1
        else:
            updated += 1
        _apply(sign, item)
    db.commit()
    return {"created": created, "updated": updated}


@router.get("/{sign_id}", response_model=SignOut)
def get_sign(sign_id: int, db: Session = Depends(get_db)):
    sign = db.get(Sign, sign_id)
    if sign is None:
        raise HTTPException(status_code=404, detail="Sign not found")
    return sign


@router.put("/{sign_id}", response_model=SignOut, dependencies=[Depends(require_admin)])
def update_sign(sign_id: int, payload: SignCreate, db: Session = Depends(get_db)):
    sign = db.get(Sign, sign_id)
    if sign is None:
        raise HTTPException(status_code=404, detail="Sign not found")
    _apply(sign, payload)
    db.commit()
    db.refresh(sign)
    return sign


@router.delete("/{sign_id}", status_code=204, dependencies=[Depends(require_admin)])
def delete_sign(sign_id: int, db: Session = Depends(get_db)):
    sign = db.get(Sign, sign_id)
    if sign is None:
        raise HTTPException(status_code=404, detail="Sign not found")
    db.delete(sign)
    db.commit()
