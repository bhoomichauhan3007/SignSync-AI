from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ---------- Auth / users ----------
WHO_OPTIONS = {"learner", "parent", "teacher", "friend", "professional", "other"}
AGE_OPTIONS = {"under_18", "18_25", "26_40", "41_60", "over_60"}
EXPERIENCE_OPTIONS = {"never", "few_signs", "basics", "regular"}
GOAL_OPTIONS = {"fun_culture", "school_college", "communicate", "work_service", "other"}


class ProfileIn(BaseModel):
    who: str
    age_group: str
    experience: str
    goal: str
    problem: str = Field(default="", max_length=1000)
    consent: bool

    @field_validator("who")
    @classmethod
    def _who(cls, v):
        if v not in WHO_OPTIONS:
            raise ValueError("choose one of the listed options")
        return v

    @field_validator("age_group")
    @classmethod
    def _age(cls, v):
        if v not in AGE_OPTIONS:
            raise ValueError("choose one of the listed options")
        return v

    @field_validator("experience")
    @classmethod
    def _experience(cls, v):
        if v not in EXPERIENCE_OPTIONS:
            raise ValueError("choose one of the listed options")
        return v

    @field_validator("goal")
    @classmethod
    def _goal(cls, v):
        if v not in GOAL_OPTIONS:
            raise ValueError("choose one of the listed options")
        return v

    @field_validator("consent")
    @classmethod
    def _consent(cls, v):
        if v is not True:
            raise ValueError("please accept the privacy notice to create an account")
        return v


class ProfileOut(BaseModel):
    who: str
    age_group: str
    experience: str
    goal: str
    problem: Optional[str] = None
    created_at: datetime


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    profile: Optional[ProfileIn] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    role: str
    created_at: datetime


class UserUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------- Signs / lessons ----------
class SignOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    category: str
    level: int
    is_dynamic: bool
    description: Optional[str] = None
    reference_media_url: Optional[str] = None
    is_emergency: bool = False
    emergency_phrase: Optional[str] = None


class SignCreate(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=30)
    level: int = 1
    is_dynamic: bool = False
    description: Optional[str] = None
    reference_media_url: Optional[str] = None
    reference_landmarks: Optional[list] = None
    is_emergency: bool = False
    emergency_phrase: Optional[str] = None


class LessonOut(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    level: int
    order_index: int
    sign_count: int


class LessonDetail(LessonOut):
    signs: list[SignOut]


class LessonCreate(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    description: Optional[str] = None
    level: int = 1
    order_index: int = 0
    sign_ids: list[int] = []


class LessonSignsUpdate(BaseModel):
    sign_ids: list[int]


# ---------- Prediction ----------
class PredictRequest(BaseModel):
    # One frame: list of [x, y, z] points. Dynamic sign: list of frames.
    # Exact shape is agreed with the CV / ML members (see INTEGRATION_CONTRACT.md).
    landmarks: list[Any] = Field(min_length=1)
    mode: str = Field(default="static", pattern="^(static|dynamic)$")
    conversation_id: Optional[int] = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    low_confidence: bool
    message: Optional[str] = None
    top_k: list[dict] = []
    is_emergency: bool = False
    emergency_phrase: Optional[str] = None
    source: str = "mock"


# ---------- Translation ----------
class SentenceRequest(BaseModel):
    words: list[str] = Field(min_length=1)
    conversation_id: Optional[int] = None


class SentenceResponse(BaseModel):
    sentence: str
    source: str


class TextToSignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    conversation_id: Optional[int] = None


class SequenceItem(BaseModel):
    label: str
    animation_id: str
    known: bool
    fingerspell: list[str] = []


class SignSequenceResponse(BaseModel):
    text: str
    sequence: list[SequenceItem]
    source: str


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: str = "en"


# ---------- Learning ----------
class EvaluateRequest(BaseModel):
    sign_id: int
    landmarks: list[Any] = Field(min_length=1)


class EvaluateResponse(BaseModel):
    sign_id: int
    label: str
    predicted_label: str
    confidence: float
    low_confidence: bool
    is_correct: bool
    score: float
    feedback: dict
    hint: Optional[str] = None
    xp_gained: int
    xp_total: int
    level: int
    streak: int
    new_badges: list[str] = []
    source: str


class AttemptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sign_id: int
    predicted_label: str
    confidence: float
    score: float
    is_correct: bool
    feedback: Optional[dict] = None
    created_at: datetime


# ---------- Quiz ----------
class QuizQuestion(BaseModel):
    sign_id: int
    prompt: str
    description: Optional[str] = None
    reference_media_url: Optional[str] = None
    options: list[str]


class QuizOut(BaseModel):
    lesson_id: int
    questions: list[QuizQuestion]


class QuizAnswer(BaseModel):
    sign_id: int
    chosen_label: str


class QuizSubmit(BaseModel):
    answers: list[QuizAnswer] = Field(min_length=1)


class QuizResultOut(BaseModel):
    lesson_id: int
    score: int
    total: int
    xp_gained: int
    results: list[dict]


# ---------- Progress ----------
class GamificationOut(BaseModel):
    xp: int
    level: int
    streak: int
    best_streak: int
    badges: list[str]
    last_active_date: Optional[date] = None


class LessonProgressOut(BaseModel):
    lesson_id: int
    title: str
    completion_pct: float
    best_score: float
    attempts_count: int


class CategoryProgressOut(BaseModel):
    category: str
    total: int
    mastered: int
    percent: float


class ProgressOut(BaseModel):
    gamification: GamificationOut
    lessons: list[LessonProgressOut]
    categories: list[CategoryProgressOut]


class WeakSign(BaseModel):
    sign_id: int
    label: str
    avg_score: float
    attempts: int


class RecommendationOut(BaseModel):
    message: str
    weak_signs: list[WeakSign]
    suggested_lesson_id: Optional[int] = None


class LeaderboardEntry(BaseModel):
    rank: int
    name: str
    xp: int
    level: int
    streak: int


# ---------- Conversations / assistant ----------
class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: Optional[str] = None
    created_at: datetime


class MessageCreate(BaseModel):
    direction: str = Field(max_length=30)
    text: str = Field(min_length=1, max_length=5000)
    meta: Optional[dict] = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    direction: str
    text: str
    meta: Optional[dict] = None
    created_at: datetime


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: Optional[int] = None


class AssistantResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    source: str
    mode: Optional[str] = None  # llm | extractive | refused | unavailable (mock: None)
