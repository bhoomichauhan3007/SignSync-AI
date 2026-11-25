"""Configuration for the RAG module.

Every setting comes from an environment variable. Nothing secret is hardcoded.
Values may also be placed in backend/rag/.env or backend/.env. Those files are read
with a tiny parser that only accepts RAG_*, GEMINI_* and OLLAMA_* keys and never
overrides a variable that is already set in the shell.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
BACKEND_DIR = RAG_DIR.parent
_ALLOWED_PREFIXES = ("RAG_", "GEMINI_", "OLLAMA_")


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    # utf-8-sig removes a Byte Order Mark that some editors add at the start of the file.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if " #" in value:  # inline comment, e.g. RAG_TOP_K=4   # number of passages
            value = value.split(" #", 1)[0]
        value = value.strip().strip('"').strip("'")
        if key.startswith(_ALLOWED_PREFIXES):
            # An empty variable already set in Windows must not block the value from the file.
            if not os.environ.get(key, "").strip():
                os.environ[key] = value


def _env(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


_load_env_file(RAG_DIR / ".env")
_load_env_file(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class RagConfig:
    knowledge_base_dir: Path
    index_dir: Path
    embedder: str              # auto | sentence-transformers | hashing
    embedding_model: str
    vector_store: str          # auto | faiss | numpy
    top_k: int
    min_score: float | None    # None = use the embedder's default threshold
    chunk_chars: int
    chunk_overlap_chars: int
    max_context_chars: int
    llm_provider: str          # auto (Gemini if a key is set, else extractive) | extractive | gemini | ollama
    gemini_api_key: str | None
    gemini_model: str
    ollama_url: str
    ollama_model: str
    llm_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "RagConfig":
        min_score_raw = _env("RAG_MIN_SCORE", None)
        return cls(
            knowledge_base_dir=Path(_env("RAG_KB_DIR", str(RAG_DIR / "knowledge_base"))),
            index_dir=Path(_env("RAG_INDEX_DIR", str(RAG_DIR / "index"))),
            embedder=_env("RAG_EMBEDDER", "auto").lower(),
            embedding_model=_env("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            vector_store=_env("RAG_VECTOR_STORE", "auto").lower(),
            top_k=int(_env("RAG_TOP_K", "4")),
            min_score=float(min_score_raw) if min_score_raw is not None else None,
            chunk_chars=int(_env("RAG_CHUNK_CHARS", "900")),
            chunk_overlap_chars=int(_env("RAG_CHUNK_OVERLAP_CHARS", "150")),
            max_context_chars=int(_env("RAG_MAX_CONTEXT_CHARS", "6000")),
            llm_provider=_env("RAG_LLM_PROVIDER", "auto").lower(),
            gemini_api_key=_env("GEMINI_API_KEY", None),
            gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
            ollama_url=_env("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=_env("OLLAMA_MODEL", "llama3.2"),
            llm_timeout_seconds=float(_env("RAG_LLM_TIMEOUT", "30")),
        )
