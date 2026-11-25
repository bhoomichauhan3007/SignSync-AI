# SignSync AI - Indian Sign Language communication and learning platform

Six-member college major project. This is the integrated repository.

    SignSync-AI/
    ├── backend/              Member 5 (FastAPI API, auth, progress, AI adapter)
    │   ├── rag/              Member 4 (AI tutor: ingestion, FAISS/numpy index, retrieval, Gemini/Ollama)
    │   ├── ml/               Member 2 adapter (alphabet recognition with alphabet_mlp.joblib)
    │   ├── nlp/              Member 3 adapter (wraps nlp_translation/isl_nlp.py)
    │   ├── integration_tests/  cross-member tests
    │   └── tests/            Member 5 API tests (pytest)
    ├── frontend/             Member 6 (React + Vite app; AI Tutor page connected to the API)
    ├── computer_vision/      Member 1 (MediaPipe landmark extraction, dataset export, Holistic extractor)
    ├── data/                 raw alphabet images (raw/alphabets), exports and processed files
    ├── sign_recognition/     Member 2 (training scripts, alphabet model)
    ├── nlp_translation/      Member 3 (isl_nlp.py, sign vocabulary)
    ├── scripts/run_tests.sh  Runs every suite that can run here
    ├── docs/                 API.md, INTEGRATION_STATUS.md, TESTING_REPORT.md, MEMBER4_ROADMAP.md
    ├── requirements.txt      Core install
    ├── requirements-optional.txt  Optional extras (Whisper, gTTS, MediaPipe, sentence-transformers, ...)
    └── .env.example          Settings (copy blocks into backend/.env and backend/rag/.env)

## Quick start

Backend (Python 3.11 or 3.12 recommended):

    python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example backend/.env                              # then set SECRET_KEY, ADMIN_PASSWORD; AI_MODE=real
    cd backend
    python -m rag.scripts.build_index --embedder hashing --vector-store numpy   # index already shipped; rebuild after adding documents
    python -m uvicorn app.main:app --reload --port 8000

Open http://localhost:8000/docs, or check http://localhost:8000/health/ai (ask_assistant, predict_sign, signs_to_sentence show "real").

Frontend:

    cd frontend
    cp .env.example .env
    npm install
    npm run dev          # http://localhost:5173 (CORS allows this origin)

Sign in on the AI Tutor page with the admin email and password from `backend/.env` (ADMIN_EMAIL, ADMIN_PASSWORD).

Tests:

    ./scripts/run_tests.sh

## What works, honestly

- Tutor (Member 4): answers from the knowledge base with sources; refuses when it has no answer. The shipped knowledge base is a single sample document, so answers are only as good as that document. Add verified ISL material in `backend/rag/knowledge_base/verified/`.
- Alphabet recognition (Member 2): 90.46% top-1 on Member 1's held-out test split. Dynamic word recognition is not connected.
- Translation (Member 3): text to sign-clip sequence works. Sentences without an LLM key are rule-based and weak. Speech needs Whisper and gTTS.
- Camera and landmark extraction (Member 1): Python recorder works on Windows-style webcams. The browser does not yet send landmarks to the backend.

See docs/INTEGRATION_STATUS.md for the full hook-by-hook status.

## Security

Change `SECRET_KEY` and `ADMIN_PASSWORD` before any shared deployment. Never commit `.env` files or API keys.

## Status

What is implemented, and what is still to do for each member, is in docs/WORK_STATUS.md.
The live recognition and speech features are described in docs/LIVE_TRANSLATOR.md.
