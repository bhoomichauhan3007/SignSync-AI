import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..config import settings
from ..database import SessionLocal
from ..security import decode_token
from ..services import ai_adapter as ai
from ..services.inference import run_prediction

logger = logging.getLogger("signsync.ws")
router = APIRouter(tags=["realtime"])


def _predict_sync(landmarks: list, mode: str) -> dict:
    with SessionLocal() as db:
        return run_prediction(db, landmarks, mode)


@router.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket, token: str = Query(...)):
    """
    Real-time sign recognition.  Connect with  ws://host/ws/predict?token=<access_token>

    Client -> server (JSON):
      {"type": "landmarks", "landmarks": [...], "mode": "static"|"dynamic"}
      {"type": "flush"}   # turn collected words into a sentence
      {"type": "reset"}   # clear collected words
      {"type": "ping"}

    Server -> client (JSON):
      {"type": "prediction", ...prediction fields..., "words": [...], "word_committed": "HELLO"?}
      {"type": "sentence", "sentence": "...", "words": [...]}
      {"type": "error", "detail": "..."}  /  {"type": "pong"}

    A word is committed after the same confident label appears in
    WS_STABILITY_FRAMES consecutive frames (avoids flicker).
    """
    if decode_token(token, "access") is None:
        await websocket.close(code=4401)
        return
    await websocket.accept()

    words: list[str] = []
    last_label = None
    streak = 0
    last_committed = None

    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except ValueError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
                continue

            mtype = msg.get("type", "landmarks") if isinstance(msg, dict) else None

            if mtype == "landmarks":
                landmarks = msg.get("landmarks")
                if not landmarks or not isinstance(landmarks, list):
                    await websocket.send_json({"type": "error", "detail": "landmarks must be a non-empty list"})
                    continue
                mode = msg.get("mode", "static")
                if mode not in ("static", "dynamic"):
                    mode = "static"

                result = await asyncio.to_thread(_predict_sync, landmarks, mode)
                payload = {"type": "prediction", **result}

                if not result["low_confidence"]:
                    label = result["label"]
                    if label == last_label:
                        streak += 1
                    else:
                        last_label, streak = label, 1
                        if last_committed != label:
                            last_committed = None
                    if streak >= settings.ws_stability_frames and last_committed != label:
                        words.append(label)
                        last_committed = label
                        payload["word_committed"] = label
                payload["words"] = list(words)
                await websocket.send_json(payload)

            elif mtype == "flush":
                if words:
                    out = await asyncio.to_thread(ai.signs_to_sentence, list(words))
                    await websocket.send_json({"type": "sentence", "sentence": out["sentence"], "words": list(words)})
                else:
                    await websocket.send_json({"type": "sentence", "sentence": "", "words": []})
                words.clear()
                last_label, streak, last_committed = None, 0, None

            elif mtype == "reset":
                words.clear()
                last_label, streak, last_committed = None, 0, None
                await websocket.send_json({"type": "reset", "words": []})

            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "detail": f"Unknown message type: {mtype}"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
