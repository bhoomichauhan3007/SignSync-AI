"""Export Member 2's alphabet model (scaler + MLP) to JSON so the browser can run it.

Writes frontend/public/models/alphabet.json. Run from backend/:  python -m ml.export_browser_model
The browser then recognises letters with no network round trip. The forward pass here must match
scikit-learn; the check at the end of this script confirms it on the held-out test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sign_model import REPO_ROOT, load_bundle, model_path

OUT = REPO_ROOT / "frontend" / "public" / "models" / "alphabet.json"
DATASET = REPO_ROOT / "data" / "exports" / "alphabet_dataset.npz"


def forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Same arithmetic the browser uses: standardise, two ReLU layers, softmax."""
    h = (x - np.array(params["mean"])) / np.array(params["scale"])
    layers = params["layers"]
    for i, layer in enumerate(layers):
        h = h @ np.array(layer["weights"]) + np.array(layer["bias"])
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


def export() -> dict:
    bundle = load_bundle()
    pipeline, encoder = bundle["model"], bundle["encoder"]
    scaler = pipeline.steps[0][1]
    mlp = pipeline.steps[-1][1]
    if mlp.activation != "relu" or mlp.out_activation_ != "softmax":
        raise ValueError("browser export supports the relu/softmax MLP only")
    letters = [str(c) for c in encoder.inverse_transform(np.asarray(mlp.classes_).astype(int))]
    params = {
        "model": model_path().name,
        "labels": letters,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "layers": [
            {"weights": w.astype(float).tolist(), "bias": b.astype(float).tolist()}
            for w, b in zip(mlp.coefs_, mlp.intercepts_)
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(params), encoding="utf-8")
    return params


def verify(params: dict, max_samples: int | None = None) -> dict:
    data = np.load(DATASET, allow_pickle=True)
    mask = np.asarray(data["splits"]) == "Testing"
    X = data["X"][mask].astype(np.float64)
    y = np.asarray(data["y"])[mask]
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]
    bundle = load_bundle()
    sk_prob = bundle["model"].predict_proba(X)
    my_prob = forward(X, params)
    labels = np.array(params["labels"])
    return {
        "samples": int(len(X)),
        "max_probability_difference": float(np.max(np.abs(sk_prob - my_prob))),
        "accuracy_browser_math": float(np.mean(labels[np.argmax(my_prob, axis=1)] == y)),
    }


if __name__ == "__main__":
    params = export()
    print("written:", OUT, f"({OUT.stat().st_size // 1024} KB)")
    print(json.dumps(verify(params), indent=2))
