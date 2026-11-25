"""Checks the external video word dataset BEFORE anyone uses it (SignSync integration, Member 1).

Run from the computer_vision/video_dataset folder:  python check_video_dataset.py
It reads ../../data/final_dataset/ (generated locally; the data is not included in the shared project).

It reports:
  * shapes, class count and split sizes
  * how many of SignSync's 40 words the data covers
  * pose points that are guesses (low visibility) and points outside the picture
  * whether the same ISL500 session appears in both training and test (possible same signer)
  * the research-only licence reminder
It does not change any data.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "final_dataset"
VOCAB = HERE.parent.parent / "nlp_translation" / "sign_vocab.json"
POSE_VIS = 0.5


def main() -> int:
    if not (DATA / "train.npz").exists():
        print(f"Dataset not found in {DATA}. See docs/DATASET_REPORT.md section 4.for where to place it.")
        return 1
    splits = {name: np.load(DATA / f"{name}.npz", allow_pickle=True) for name in ("train", "validation", "test")}
    problems = []

    print("SHAPES AND SIZES")
    for name, d in splits.items():
        X = d["X"]
        ok = X.ndim == 3 and X.shape[1:] == (30, 1662) and np.isfinite(X).all()
        print(f"  {name:10s} clips {len(X):4d}  shape {X.shape}  finite {bool(np.isfinite(X).all())}")
        if not ok:
            problems.append(f"{name}: wrong shape or NaN/Inf")

    train = splits["train"]
    classes = {str(c) for c in train["class_names"]}
    vocab = {w.lower() for w in json.loads(VOCAB.read_text(encoding="utf-8")) if len(w) > 1}
    print("\nVOCABULARY OVERLAP WITH SIGNSYNC'S 40 WORDS")
    print(f"  shared words: {len(classes & vocab)} -> {sorted(classes & vocab)}")
    print(f"  words only in this data: {len(classes - vocab)}")
    print(f"  SignSync words with no data: {len(vocab - classes)}")
    if len(classes & vocab) < 40:
        problems.append("the data covers only some of SignSync's 40 words")

    print("\nPOSE QUALITY (pose has 33 points x x,y,z,visibility)")
    pose = train["X"][:, :, 0:132].reshape(-1, 30, 33, 4)
    vis = pose[..., 3]
    outside = (pose[..., 1] > 1.2) | (pose[..., 1] < -0.2)
    print(f"  pose points with visibility below {POSE_VIS}: {float((vis < POSE_VIS).mean()):.1%}")
    print(f"  pose points outside the picture: {float(outside.mean()):.1%}")
    print(f"  ... of which low visibility: {float((vis[outside] < POSE_VIS).mean()):.1%}")
    cleaned = pose.copy()
    cleaned[vis < POSE_VIS] = 0
    still = ((cleaned[..., 1] > 1.2) | (cleaned[..., 1] < -0.2)).any(axis=(1, 2)).mean()
    print(f"  clips still outside the picture after ignoring low-visibility points: {still:.1%}")
    if still > 0.2:
        problems.append("many clips still have points outside the picture after masking low-visibility points")

    print("\nSIGNER OVERLAP (ISL500 session IDs, a proxy for the person)")
    def sessions(names):
        return {m.group(1) for n in names for m in [re.search(r"session(\d+)", str(n))] if m}
    s_tr = sessions(train["video_names"])
    te_names = splits["test"]["video_names"]
    shared_clips = sum(1 for n in te_names if (m := re.search(r"session(\d+)", str(n))) and m.group(1) in s_tr)
    print(f"  test clips whose session also appears in training: {shared_clips} of {len(te_names)}")
    if shared_clips:
        problems.append("test clips share a session with training clips: scores will be too high")

    print("\nLICENCE")
    print("  Research and education use only. Do not use commercially. Do not redistribute.")
    print("  Check each source's terms yourself (INCLUDE, CISLR, ISL500). See docs/DATASET_REPORT.md section 4.")

    print("\nRESULT")
    if problems:
        for p in problems:
            print("  ISSUE:", p)
        return 2
    print("  No issues found by these checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
