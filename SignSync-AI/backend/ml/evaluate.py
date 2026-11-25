"""Measure the alphabet model on Member 1's held-out Testing split (real measurement, no estimates).

Usage (from backend/):  python -m ml.evaluate
Reads ../data/exports/alphabet_dataset.npz (X: 63 features, y: letter, splits).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .sign_model import REPO_ROOT, load_bundle, model_path

DATASET = REPO_ROOT / "data" / "exports" / "alphabet_dataset.npz"


def evaluate(dataset_path: Path = DATASET) -> dict:
    data = np.load(dataset_path, allow_pickle=True)
    mask = np.asarray(data["splits"]) == "Testing"
    X = data["X"][mask].astype(np.float32)
    y_true = np.asarray(data["y"])[mask]
    bundle = load_bundle()
    model, encoder = bundle["model"], bundle["encoder"]
    predicted = encoder.inverse_transform(np.asarray(model.predict(X)).astype(int))
    accuracy = float(np.mean(predicted == y_true))
    return {"model": model_path().name, "split": "Testing", "samples": int(mask.sum()),
            "top1_accuracy": round(accuracy, 4)}


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2))
    sys.exit(0)
