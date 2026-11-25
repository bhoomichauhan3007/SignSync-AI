import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import models  # noqa: F401  (registers tables on Base.metadata)
from .config import settings
from .database import Base, SessionLocal, engine
from .routers import (
    assistant,
    auth,
    conversations,
    learn,
    lessons,
    predict,
    progress,
    quiz,
    signs,
    translate,
    users,
    ws,
)
from .seed import seed_database
from .services import ai_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("signsync")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_database(db)
    logger.info("Started. AI mode: %s", settings.ai_mode)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Backend for SignSync AI - ISL translation and learning platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/health/ai", tags=["health"])
def health_ai():
    """Shows which AI hooks are real vs mocked - handy during integration."""
    return ai_adapter.status()


app.include_router(auth.router)
app.include_router(users.router)
app.include_router(signs.router)
app.include_router(lessons.router)
app.include_router(predict.router)
app.include_router(translate.router)
app.include_router(learn.router)
app.include_router(quiz.router)
app.include_router(progress.router)
app.include_router(progress.leaderboard_router)
app.include_router(conversations.router)
app.include_router(assistant.router)
app.include_router(ws.router)
