"""
SignSync AI - Check that CV output and the ML team's dataset share one schema.

Usage (from computer_vision/):
    python verify_schema_compat.py
    python verify_schema_compat.py --ml ../data/exports/word_dataset.npz \
                                   --cv ../data/exports/cv_word_dataset.npz

Reads both files, changes nothing. Exit code 1 if a hard incompatibility exists.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from feature_schema import FEATURES_PER_FRAME, SECTIONS, SEQUENCE_LENGTH

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ML = ROOT / "data" / "exports" / "word_dataset.npz"
DEFAULT_CV = ROOT / "data" / "exports" / "cv_word_dataset.npz"


def describe(name, path):
    print(f"\n=== {name}: {path}")

    if not path.exists():
        print("MISSING")
        return None

    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}

    X = out.get("X")

    if X is None:
        print("No 'X' array. Keys:", list(out))
        return None

    print("X:", X.shape, X.dtype, "| keys:", list(out))
    print("finite:", bool(np.isfinite(X).all()), "| min/max: %.3f / %.3f" % (X.min(), X.max()))

    for section, (start, end, _) in SECTIONS.items():
        block = X[..., start:end]
        print(f"  {section:<11} non-zero values: {np.mean(block != 0):6.1%}")

    return out


def mean_hand_x(X, section):
    start, end, _ = SECTIONS[section]
    block = X[..., start:end].reshape(-1, 21, 3)
    present = np.any(block != 0, axis=(1, 2))
    return float(block[present, :, 0].mean()) if present.any() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", type=Path, default=DEFAULT_ML)
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV)
    args = parser.parse_args()

    ml = describe("ML dataset", args.ml)
    cv = describe("CV dataset", args.cv)

    errors, warnings = [], []

    for name, d in (("ML", ml), ("CV", cv)):
        if d is None:
            continue
        shape = d["X"].shape
        if len(shape) != 3 or shape[1:] != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
            errors.append(f"{name}: X shape {shape}, expected (N, {SEQUENCE_LENGTH}, {FEATURES_PER_FRAME})")

    if ml is not None and cv is not None and not errors:
        ml_classes = [str(c) for c in ml.get("classes", [])]
        cv_classes = [str(c) for c in cv.get("classes", [])]
        unknown = sorted(set(cv_classes) - set(ml_classes))

        if unknown:
            warnings.append(f"CV classes not in ML dataset: {unknown}")

        print("\nClass overlap:", len(set(cv_classes) & set(ml_classes)), "of", len(cv_classes), "CV classes")

        # Left/right convention heuristic: the same hand should sit on the same
        # side of the image in both datasets.
        for section in ("left_hand", "right_hand"):
            a, b = mean_hand_x(ml["X"], section), mean_hand_x(cv["X"], section)
            if a is None or b is None:
                continue
            print(f"mean x of {section}: ML={a:.3f} CV={b:.3f}")
            if abs(a - b) > 0.2:
                warnings.append(
                    f"{section} mean x differs a lot (ML {a:.2f} vs CV {b:.2f}) - "
                    "possible mirrored/swapped hands (see MIRROR_INPUT_FOR_HOLISTIC)."
                )

    print()
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)

    if not errors and ml is not None and cv is not None:
        print("Structural check passed (shapes/dtypes). This does NOT prove the model works.")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
