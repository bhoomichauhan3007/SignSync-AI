# Testing report (21 November 2026)

Environment: Python 3.12 sandbox with numpy, scikit-learn, joblib, pypdf, PIL. No FastAPI, no pytest, no TensorFlow,
no Whisper, no gTTS, no Gemini key, no browser. Node.js was used for the browser code tests.

## Executed and passing

| Test | Result |
|---|---|
| Tutor (backend/rag/tests) | 63 run, 61 passed, 2 skipped (semantic-search packages not installed) |
| Cross-member integration (backend/integration_tests) | 14 run, 14 passed (includes the backend adapter, Member 3's NLP fallback, word input checks) |
| Browser letter model (frontend/tests/alphabetModel.test.cjs) | 400 of 400 labels match the Python model; max confidence difference 8.9e-16 |
| Letter smoothing and hand-gone rules (frontend/tests/liveLogic.test.cjs) | passed |
| Frontend TypeScript syntax check (all src files) | 0 errors |
| Clean extraction of the ZIP, then ./scripts/run_tests.sh | passed |

## Measured

| Measure | Value | Note |
|---|---|---|
| Letter model, 4,633 held-out images (old dataset) | 90.46% top-1 | Re-measure after the dataset cleaning |
| Letter model, first 400 test samples, browser math | 99.8% | a sanity check on a subset, not a new score |
| Tutor evaluation, 8 questions, sample knowledge base | source hit 1.0, refusal 1.0 | only one document; not a real retrieval test |

## Not executed here (run on your computer)

| Item | How |
|---|---|
| Backend API suite (Member 5) | cd backend, python -m pytest tests -q |
| Live letters in the browser | docs/LIVE_TRANSLATOR.md, steps 1 to 2 |
| Word recognition with TensorFlow | install TensorFlow, docs/LIVE_TRANSLATOR.md step 3 |
| Gemini sentences and tutor answers | set GEMINI_API_KEY in backend/.env |
| Whisper speech to text, gTTS speech | install openai-whisper, gTTS, ffmpeg |
| Production build (npm run build) | cd frontend, npm install, npm run build |

## Added on 21 November (second round)

- The tutor auto-selects Gemini when a key is set (test added; passes).
- scripts/check_setup.py runs on your computer and reports READY or NOT READY for speech, Gemini, semantic search, TensorFlow and the models. It was run here: 8 of 15 items ready in the sandbox, which has no Whisper, gTTS, Gemini key, sentence-transformers, FAISS or TensorFlow.
- Not verified here: Whisper speech, Gemini answers, sentence-transformers search. Please run check_setup and tell us the output.

## Added this round (third round)

- Integration tests: 20 run and pass (including 3 new tests for the hand-shape feedback, and 3 word-asset tests).
- Word package checker (sign_recognition/word/check_word_artifacts.py): ALL CHECKS PASSED (no TensorFlow needed).
- Hand-shape feedback measured on 650 held-out letter images: correct letter closer than the closest wrong letter in 82%; correct letters median score 94.
  Earlier method (one average shape per letter) was only 54%, and was replaced.
- My Progress and Live Translator: TypeScript syntax check passes, 0 errors.
- NOT tested here: TensorFlow word recognition, the live webcam word script, the My Progress page with a real learner account, the frontend build.

## Added this round (fourth round)

- Member 1 CV changes integrated. Their test_cv_pipeline.py: 8 of 8 pass inside the project.
- Tutor tests: 67 run, 65 pass, 2 skipped (semantic and FAISS tests need the real packages). New: semantic path (build, save, reload, retrieve) and Gemini answer path (with the network call stubbed).
- The stand-ins are used only when the real packages are missing, and they test our code, not the quality of the real model.
