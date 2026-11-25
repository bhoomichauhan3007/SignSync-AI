
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS / "word_gru.keras"
NORMALIZATION_PATH = ARTIFACTS / "normalization.npz"
LABELS_PATH = ARTIFACTS / "label_encoder.json"

app = FastAPI(title="SignSync Word Recognition API")

# Load the trained model and preprocessing artifacts once
model = tf.keras.models.load_model(MODEL_PATH)

with np.load(NORMALIZATION_PATH) as data:
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)


def predict_sequence(sequence, top_k=5):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (30, 1662):
        raise ValueError(
            f"Expected shape (30, 1662), got {sequence.shape}"
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Sequence contains NaN or infinite values")

    normalized = (sequence - mean) / std
    probabilities = model.predict(
        normalized[np.newaxis, ...],
        verbose=0
    )[0]

    if len(probabilities) != len(labels):
        raise ValueError("Model output count does not match label mapping")

    top_k = max(1, min(int(top_k), len(labels)))
    indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = [
        {
            "word": labels[str(int(i))],
            "confidence": round(float(probabilities[i]) * 100, 2)
        }
        for i in indices
    ]

    return {
        "predicted_word": predictions[0]["word"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "GRU",
        "input_shape": [30, 1662],
        "classes": len(labels)
    }


@app.post("/predict-word")
async def predict_word(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Upload a .npy sequence file"
        )

    try:
        contents = await file.read()
        sequence = np.load(io.BytesIO(contents), allow_pickle=False)
        return predict_sequence(sequence)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
