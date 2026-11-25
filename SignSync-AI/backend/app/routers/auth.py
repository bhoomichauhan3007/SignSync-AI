from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Gamification, User, UserProfile
from ..ratelimit import RateLimiter
from ..schemas import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserOut
from ..security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])
login_limiter = RateLimiter(settings.login_rate_limit, 60)


def _issue_tokens(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=UserOut.model_validate(user),
    )


def _authenticate(db: Session, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email.lower()))
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return user


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(name=payload.name.strip(), email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.flush()
    db.add(Gamification(user_id=user.id, xp=0, streak=0, best_streak=0, badges=[]))
    if payload.profile is not None:
        details = payload.profile
        db.add(UserProfile(
            user_id=user.id, who=details.who, age_group=details.age_group,
            experience=details.experience, goal=details.goal,
            problem=(details.problem or "").strip() or None, consent=details.consent,
        ))
    db.commit()
    db.refresh(user)
    return _issue_tokens(user)


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(login_limiter)])
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    return _issue_tokens(_authenticate(db, payload.email, payload.password))


@router.post("/token", response_model=TokenResponse, dependencies=[Depends(login_limiter)])
def token_form(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Form-based login used by the Swagger 'Authorize' button (username = email)."""
    return _issue_tokens(_authenticate(db, form.username, form.password))


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    sub = decode_token(payload.refresh_token, "refresh")
    if sub is None:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    user = db.get(User, int(sub))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return _issue_tokens(user)
