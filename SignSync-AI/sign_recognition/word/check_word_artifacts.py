"""Checks Member 2's word artifacts WITHOUT TensorFlow. Run from this folder: python check_word_artifacts.py

It checks: 40 labels in alphabetical order, the normalisation shapes and spread, that each model file says it
was saved with Keras 3 (needs TensorFlow 2.20 or newer to load), each model's input (30 x 1662) and output (40),
and the test sizes recorded in the metrics files.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
EXPECTED_CLASSES = 40
problems: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  OK   " if condition else "  FAIL ") + message)
    if not condition:
        problems.append(message)


labels = json.loads((ART / "label_encoder.json").read_text(encoding="utf-8"))
names = [labels[str(i)] for i in range(len(labels))]
print("Labels")
check(len(names) == EXPECTED_CLASSES, f"{len(names)} labels (expected {EXPECTED_CLASSES})")
check(names == sorted(names), "labels are in alphabetical order (the order the models were trained with)")
check(len(set(names)) == len(names), "no duplicate label names")

print("Normalisation")
with np.load(ART / "normalization.npz") as stats:
    check(stats["mean"].shape == (1662,) and stats["std"].shape == (1662,), "mean and std have 1662 values each")
    check(float(stats["std"].min()) > 0, f"no zero spread (smallest std {float(stats['std'].min()):.6f})")

print("Models")
for name in ["word_gru", "word_lstm", "word_transformer"]:
    path = ART / f"{name}.keras"
    if not path.exists():
        check(False, f"{name}.keras is missing")
        continue
    z = zipfile.ZipFile(path)
    meta = json.loads(z.read("metadata.json"))
    cfg = json.loads(z.read("config.json"))
    major = int(str(meta.get("keras_version", "0")).split(".")[0])
    check(major >= 3, f"{name}: saved with Keras {meta.get('keras_version')} (needs TensorFlow 2.20+ to load)")
    inp = next(l for l in cfg["config"]["layers"] if l["class_name"] == "InputLayer")["config"]["batch_shape"]
    check(inp[1:] == [30, 1662], f"{name}: input is 30 frames x 1662 values")
    last = cfg["config"]["layers"][-1]["config"].get("units")
    check(last == EXPECTED_CLASSES, f"{name}: output has {last} classes")

print("Metrics")
for name in ["gru", "lstm", "transformer"]:
    m = json.loads((ART / f"{name}_metrics.json").read_text(encoding="utf-8"))
    check(m["test_samples"] == 240, f"{name}: test set has {m['test_samples']} recordings (6 per word, a small test)")
    print(f"        {name}: test accuracy {m['test_accuracy'] * 100:.1f}%, macro F1 {m['test_macro_f1']:.3f}")

print()
print("ALL CHECKS PASSED" if not problems else f"{len(problems)} CHECK(S) FAILED")
sys.exit(1 if problems else 0)
