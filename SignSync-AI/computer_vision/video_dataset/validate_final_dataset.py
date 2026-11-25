"""Check the final word dataset (Member 1, integrated and fixed).

Checks: shapes, masks, no NaN or Inf, labels in range, and NO recording group in more than one split.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_DIR = ROOT / "data" / "final_dataset"
problems = []


def check(ok, text):
    print(("  OK   " if ok else "  FAIL ") + text)
    if not ok:
        problems.append(text)


groups_by_split = {}
for split in ["train", "validation", "test"]:
    d = np.load(DATA_DIR / f"{split}.npz", allow_pickle=True)
    X, M, y, groups = d["X"], d["M"], d["y"], d["groups"]
    n_classes = len(d["class_names"])
    print(f"{split}: {len(X)} samples")
    check(X.ndim == 3 and X.shape[1:] == (30, 1662), "shape is (samples, 30, 1662)")
    check(M.shape == X.shape, "mask has the same shape as the features")
    check(bool(np.isfinite(X).all()), "no NaN or Inf values")
    check(bool(set(np.unique(M).tolist()) <= {0, 1}), "mask values are 0 or 1")
    check(bool(y.min() >= 0 and y.max() < n_classes), "labels are in range")
    groups_by_split[split] = set(groups.tolist())

for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    shared = groups_by_split[a] & groups_by_split[b]
    check(len(shared) == 0, f"no recording group shared by {a} and {b} (shared: {len(shared)})")

print()
print("ALL CHECKS PASSED" if not problems else f"{len(problems)} CHECK(S) FAILED")
raise SystemExit(1 if problems else 0)
