# SignSync AI - Backend (Member 5)

FastAPI + SQLAlchemy backend: auth, lessons, quizzes, progress/gamification, conversations,
REST + WebSocket inference endpoints, and an AI adapter that plugs in the other members' modules.

## Quick start (local, SQLite)
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # edit SECRET_KEY
uvicorn app.main:app --reload
```
Open http://localhost:8000/docs (interactive Swagger UI). Tables and seed data are created on startup.

Dev admin account (change in `.env`): `admin@signsync.dev` / `admin12345`.

## Run with PostgreSQL (Docker)
```bash
docker compose up --build
```

## Tests
```bash
pytest -q
```

## Project layout
```
app/
  main.py            app setup, CORS, logging, router registration
  config.py          settings from .env
  database.py        engine / session
  models.py          tables
  schemas.py         request / response models
  security.py        password hashing + JWT
  deps.py            current-user / admin dependencies
  ratelimit.py       simple login rate limiter
  seed.py            placeholder signs & lessons, CSV import
  routers/           auth, users, signs, lessons, predict, translate,
                     learn, quiz, progress, conversations, assistant, ws
  services/
    ai_adapter.py    <-- the ONLY place that touches teammates' AI code (mock fallback)
    inference.py     prediction + confidence threshold + emergency detection
    gamification.py  XP, streaks, badges
    progress.py      lesson completion
    conversation.py  message logging
tests/               pytest suite
INTEGRATION_CONTRACT.md   what each member must provide
```

## Main endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register`, `/auth/login`, `/auth/refresh` | accounts + JWT |
| GET/PATCH | `/users/me` | profile |
| GET | `/signs`, `/signs/{id}` | sign dictionary (admin: POST/PUT/DELETE, `/signs/bulk`) |
| GET | `/lessons`, `/lessons/{id}` | lessons (admin: create, set signs) |
| POST | `/predict/sign` | landmarks -> label + confidence, low-confidence + emergency flags |
| WS | `/ws/predict?token=...` | real-time stream, word stability, sentence flush |
| POST | `/translate/signs-to-sentence` | words -> sentence |
| POST | `/translate/speech-to-sign`, `/translate/text-to-sign` | -> sign sequence for the avatar (with fingerspelling fallback) |
| POST | `/translate/text-to-speech` | audio or "use browser TTS" |
| POST | `/learn/evaluate` | score a learner's attempt, feedback, XP, streak, badges, progress |
| GET | `/learn/attempts` | attempt history |
| GET/POST | `/quiz/{lesson_id}`, `/quiz/{lesson_id}/submit` | multiple-choice quiz |
| GET | `/progress/me`, `/progress/recommendations`, `/leaderboard` | dashboard data |
| CRUD | `/conversations`, `/conversations/{id}/messages` | two-way conversation history |
| POST | `/assistant/ask` | RAG tutor |
| GET | `/health`, `/health/ai` | health + which AI hooks are real vs mock |

### WebSocket protocol
Connect: `ws://localhost:8000/ws/predict?token=<access_token>`
```
client -> {"type":"landmarks","landmarks":[...],"mode":"static"}
server -> {"type":"prediction","label":"HELLO","confidence":0.96,"low_confidence":false,
           "words":["HELLO"],"word_committed":"HELLO", ...}
client -> {"type":"flush"}      server -> {"type":"sentence","sentence":"Hello.","words":["HELLO"]}
client -> {"type":"reset"} / {"type":"ping"}
```
A word is committed after the same confident label appears in `WS_STABILITY_FRAMES` (default 5) consecutive frames.

## Connecting the other members' work
1. Copy their packages next to `app/`.
2. Edit the `HOOKS` table in `app/services/ai_adapter.py` if their module/function names differ.
3. Set `AI_MODE=real` in `.env`, restart, and open `/health/ai` - each hook shows `real` or `mock`.

## Before production / final demo
- Set a long random `SECRET_KEY`, change the admin password, restrict `CORS_ORIGINS`.
- Replace the placeholder label set with the real ISL labels (`python -m app.seed --csv data/signs.csv`).
- Tables are created with `create_all`; add Alembic migrations if the schema will change on a live database.
- The login rate limiter is in-memory (single process); use Redis if you run several workers.
