"""Member 2 adapter: sign recognition used by the backend (HOOKS['predict_sign']).

Connected now: STATIC ALPHABET recognition (A-Z) with Member 2's scikit-learn MLP
(sign_recognition/models/alphabet_mlp.joblib).

NOT connected: dynamic word recognition. The Keras word models expect 1662 Holistic values per frame,
and the backend does not yet receive that data. predict() raises NotImplementedError for mode="dynamic",
and the backend's adapter then returns its mock answer and logs the failure.

Dynamic (words): Member 2's word_gru.keras (default), input 30 frames x 1662 Holistic values (Member 1's
holistic_features.py layout: pose 132, face 1404, left hand 63, right hand 63). Needs TensorFlow.

Input (static): 21 hand landmarks, each [x, y, z] or {"x","y","z"}, or the flat 63-value list in the same order.
Member 2 did not normalise landmarks during training, so none is applied here. This matches the
features exported by Member 1 (computer_vision/export_features.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "sign_recognition" / "models" / "alphabet_mlp.joblib"
NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3
_cache: dict[str, dict] = {}


def model_path() -> Path:
    return Path(os.getenv("SIGN_ALPHABET_MODEL", DEFAULT_MODEL))


def load_bundle(path: Path | None = None) -> dict:
    path = Path(path) if path else model_path()
    key = str(path)
    if key not in _cache:
        import joblib  # lazy import: only needed when real recognition is used
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle or "encoder" not in bundle:
            raise ValueError(f"{path.name} is not a model bundle with 'model' and 'encoder' keys")
        _cache[key] = bundle
    return _cache[key]


def to_feature_vector(landmarks) -> np.ndarray:
    """Validate and flatten landmarks into a (63,) float32 vector in exported order."""
    if not isinstance(landmarks, (list, tuple)):
        raise ValueError("landmarks must be a list")
    if len(landmarks) == 1 and isinstance(landmarks[0], (list, tuple)) and len(landmarks[0]) == NUM_LANDMARKS:
        landmarks = landmarks[0]  # one frame wrapped in an extra list
    if len(landmarks) == FEATURES and all(isinstance(v, (int, float)) for v in landmarks):
        values = list(landmarks)
    elif len(landmarks) == NUM_LANDMARKS:
        values = []
        for point in landmarks:
            if isinstance(point, dict):
                values.extend([point.get("x"), point.get("y"), point.get("z", 0.0)])
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                values.extend([point[0], point[1], point[2] if len(point) > 2 else 0.0])
            else:
                raise ValueError("each landmark must be [x, y, z] or {x, y, z}")
    else:
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks or {FEATURES} values, got {len(landmarks)}")
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("landmark values must be numbers") from exc
    if vector.shape != (FEATURES,) or not np.all(np.isfinite(vector)):
        raise ValueError("landmark values must be finite numbers")
    return vector


WORD_FRAMES = 30
WORD_FEATURES = 1662
# Member 2's word models (sign_recognition/word/artifacts). GRU is the default: best test accuracy (89.2%).
WORD_ARTIFACTS = REPO_ROOT / "sign_recognition" / "word" / "artifacts"
WORD_MODEL_DEFAULT = "word_gru.keras"
_word_cache: dict[str, object] = {}


def word_model_path() -> Path:
    return Path(os.getenv("SIGN_WORD_MODEL", WORD_ARTIFACTS / WORD_MODEL_DEFAULT))


def load_word_assets():
    """Load the word model, the normalisation statistics and the label names (Member 2's files).

    TensorFlow is imported only here, so the rest of the backend does not need it.
    """
    path = word_model_path()
    if "model" not in _word_cache:
        import json
        import tensorflow as tf  # optional dependency (requirements-optional.txt)
        _word_cache["model"] = tf.keras.models.load_model(path, compile=False)
        with np.load(WORD_ARTIFACTS / "normalization.npz") as stats:
            _word_cache["mean"] = stats["mean"].astype(np.float32)
            _word_cache["std"] = stats["std"].astype(np.float32)
        labels = json.loads((WORD_ARTIFACTS / "label_encoder.json").read_text(encoding="utf-8"))
        _word_cache["labels"] = [labels[str(i)] for i in range(len(labels))]
    return _word_cache["model"], _word_cache["mean"], _word_cache["std"], _word_cache["labels"]



def to_word_sequence(landmarks) -> np.ndarray:
    """Validate and shape a dynamic sign as (1, 30, 1662) float32 (raw values, before normalisation)."""
    if not isinstance(landmarks, (list, tuple)) or not landmarks:
        raise ValueError("dynamic sign needs a list of frames")
    if len(landmarks) == WORD_FRAMES * WORD_FEATURES and all(isinstance(v, (int, float)) for v in landmarks):
        frames = [landmarks[i * WORD_FEATURES:(i + 1) * WORD_FEATURES] for i in range(WORD_FRAMES)]
    else:
        frames = landmarks
    if len(frames) != WORD_FRAMES:
        raise ValueError(f"expected {WORD_FRAMES} frames, got {len(frames)}")
    try:
        sequence = np.asarray(frames, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame values must be numbers") from exc
    if sequence.shape != (WORD_FRAMES, WORD_FEATURES) or not np.all(np.isfinite(sequence)):
        raise ValueError(f"expected shape ({WORD_FRAMES}, {WORD_FEATURES}) with finite values")
    return sequence.reshape(1, WORD_FRAMES, WORD_FEATURES)


def predict_dynamic(landmarks, top_k: int = 3) -> dict:
    """Same steps as Member 2's predict_word.py: normalise with the training mean and std, then classify."""
    sequence = to_word_sequence(landmarks)
    model, mean, std, labels = load_word_assets()
    normalised = (sequence - mean) / std
    probabilities = np.asarray(model.predict(normalised, verbose=0))[0]
    order = np.argsort(-probabilities)[:top_k]
    ranked = [{"label": labels[int(i)], "confidence": round(float(probabilities[i]), 4)} for i in order]
    return {
        "label": ranked[0]["label"],
        "confidence": ranked[0]["confidence"],
        "top_k": ranked,
        "model": word_model_path().name,
    }


def predict(landmarks, mode: str = "static", top_k: int = 3) -> dict:
    """Return {"label", "confidence", "top_k": [{"label","confidence"}], "model"}."""
    if mode == "dynamic":
        return predict_dynamic(landmarks, top_k)
    bundle = load_bundle()
    model, encoder = bundle["model"], bundle["encoder"]
    vector = to_feature_vector(landmarks).reshape(1, -1)
    probabilities = model.predict_proba(vector)[0]
    order = np.argsort(-probabilities)[:top_k]
    labels = encoder.inverse_transform(np.asarray(model.classes_)[order].astype(int))
    ranked = [{"label": str(label), "confidence": round(float(probabilities[i]), 4)}
              for label, i in zip(labels, order)]
    return {
        "label": ranked[0]["label"],
        "confidence": ranked[0]["confidence"],
        "top_k": ranked,
        "model": model_path().name,
    }
