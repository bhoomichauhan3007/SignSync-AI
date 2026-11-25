# SignSync AI - Integration Contract

This is the agreement between the backend (Member 5) and the other members.
The backend never imports teammates' code anywhere except `app/services/ai_adapter.py`.
Until a module exists, the adapter returns **mock** data, so everyone can work in parallel.

Put each member's package next to the `app/` folder (e.g. `backend/ml/`, `backend/nlp/`, `backend/rag/`, `backend/cv/`)
and set `AI_MODE=real`. Check `GET /health/ai` to see which hooks are real and which are still mock.

## Landmark format (Members 1 + 2 must confirm - TODO)
- Static sign: one frame = list of `[x, y, z]` points (e.g. 21 per hand, optionally pose points).
- Dynamic sign: list of frames, each frame as above.
- Coordinates normalised (MediaPipe style, 0..1). Agree on point ORDER and whether both hands are always present.

## Functions the backend calls

| Owner | Module.function (edit in `HOOKS`) | Input | Output |
|---|---|---|---|
| Member 2 | `ml.sign_model.predict(landmarks, mode)` | landmarks (see above), `mode` = `"static"`/`"dynamic"` | `{"label": "HELLO", "confidence": 0.96, "top_k": [{"label": "...", "confidence": 0.0}]}` (or a `(label, confidence)` tuple) |
| Member 1/2 | `cv.feedback.compare_with_reference(landmarks, label, reference)` | user landmarks, expected label, reference landmarks or `None` | `{"score": 0-100, "feedback": {"hand_position"\|"hand_shape"\|"movement"\|"orientation": "ok"\|"warn"\|"fail"}, "hint": "text or None"}` |
| Member 3 | `nlp.pipeline.signs_to_sentence(words)` | `list[str]` | `str` (or `{"sentence": str}`) |
| Member 3 | `nlp.pipeline.speech_to_text(audio_bytes, language)` | bytes, e.g. `"en"` | `str` (or `{"text": str}`) |
| Member 3 | `nlp.pipeline.text_to_sign_sequence(text)` | `str` | list of labels (`["THANK YOU","HELP"]`) or `[{"label","animation_id"}]` |
| Member 3 | `nlp.pipeline.text_to_speech(text, language)` | `str`, language | WAV `bytes` |
| Member 4 | `rag.assistant.ask(question, history)` | `str`, `list[{"role","content"}]` | `{"answer": str, "sources": [{"title": ...}]}` (or a plain `str`) |

Labels must match the `signs.label` values in the database (upper case). Member 2 should give the full label list
so it can be loaded with `python -m app.seed --csv data/signs.csv` (see `data/signs_example.csv`).

## Questions to settle in the kickoff meeting
1. Does the frontend send landmarks (extracted in the browser with MediaPipe JS) or raw frames? (Backend currently expects landmarks.)
2. Exact landmark shape and order.
3. Model file format and where it lives (backend loads it once at startup, not per request).
4. Dependency conflicts (TensorFlow vs PyTorch)? If yes, run that module as a separate HTTP service and call it from its adapter function.
5. Who owns avatar animation ids (`animation_id`) - Member 3 (sequence) and Member 6 (rendering).
