from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import User, UserProfile
from ..schemas import ProfileOut, UserOut, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserOut)
def read_me(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserOut)
def update_me(payload: UserUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if payload.name is not None:
        user.name = payload.name.strip()
    db.commit()
    db.refresh(user)
    return user


@router.get("/me/profile", response_model=ProfileOut | None)
def read_my_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.get(UserProfile, user.id)
    if profile is None:
        return None
    return ProfileOut(
        who=profile.who, age_group=profile.age_group, experience=profile.experience,
        goal=profile.goal, problem=profile.problem, created_at=profile.created_at,
    )
