"""Shape feedback for a learner's sign (backend hook: cv.feedback.compare_with_reference).

LETTERS: the learner's 21 hand points are compared with the reference shape of that letter.
Position and size are removed first (points are moved so the wrist is at the origin and the
wrist-to-middle-finger-base distance is 1), so only the SHAPE of the hand is compared.

The reference is 30 real training examples of that letter (data/exports), unless the sign stores its own
reference landmarks. Comparing with real examples works better than one average shape (see docs/TESTING_REPORT.md).

WORDS: a shape comparison does not describe a whole word. For words the score is a fixed 70 and the
hint says so. Word correctness comes from the word model.

Score: 100 for an identical shape, falling to 0 at a distance of 0.8 (calibrated, see docs/TESTING_REPORT.md).
Feedback: ok (70 or more), warn (45 to 69), fail (below 45). The hint names the finger that differs most.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHABET_DATASET = REPO_ROOT / "data" / "exports" / "alphabet_dataset.npz"
POINTS = 21
ZERO_DISTANCE = 0.8            # distance at which the score reaches 0 (calibrated on the alphabet images)
WORD_SCORE = 70.0              # fixed score for words (see module note)
FINGERS = {0: "palm", 1: "thumb", 2: "thumb", 3: "thumb", 4: "thumb",
           5: "index finger", 6: "index finger", 7: "index finger", 8: "index finger",
           9: "middle finger", 10: "middle finger", 11: "middle finger", 12: "middle finger",
           13: "ring finger", 14: "ring finger", 15: "ring finger", 16: "ring finger",
           17: "little finger", 18: "little finger", 19: "little finger", 20: "little finger"}


def shape_of(points63) -> np.ndarray:
    """21 points (x, y, z) -> 21 x 2 array of the hand SHAPE (position and size removed)."""
    pts = np.asarray(points63, dtype=np.float64).reshape(POINTS, 3)[:, :2]
    pts = pts - pts[0]                        # wrist at the origin
    scale = np.linalg.norm(pts[9])            # wrist to middle-finger base
    if scale < 1e-9:
        raise ValueError("hand is too small or not visible")
    return pts / scale


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def to_score(dist: float) -> float:
    return round(float(max(0.0, min(100.0, 100.0 * (1.0 - dist / ZERO_DISTANCE)))), 1)


def verdict(score: float) -> str:
    return "ok" if score >= 70 else "warn" if score >= 45 else "fail"


EXAMPLES_PER_LETTER = 30


@lru_cache(maxsize=1)
def letter_examples() -> dict[str, np.ndarray]:
    """Up to 30 real training examples per letter, as shapes (data/exports, Training split, fixed seed).

    Using real examples works better than one average shape: on test images the correct letter ranked
    first in 82% of cases (chance is about 4%), while the average shape ranked it first only ~54% of the time.
    """
    data = np.load(ALPHABET_DATASET, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = np.asarray(data["y"]).astype(str)
    split = np.asarray(data["splits"]).astype(str)
    rng = np.random.default_rng(0)
    examples: dict[str, np.ndarray] = {}
    for letter in np.unique(y):
        idx = np.where((y == letter) & (split == "Training") & (np.linalg.norm(X, axis=1) > 0))[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(EXAMPLES_PER_LETTER, len(idx)), replace=False)
        examples[str(letter)] = np.array([shape_of(X[i]) for i in chosen])
    return examples


def _flat_hand(landmarks) -> np.ndarray | None:
    from ml.sign_model import to_feature_vector   # same validation the recogniser uses
    try:
        return to_feature_vector(landmarks)
    except ValueError:
        return None


def compare_with_reference(landmarks, label: str, reference=None) -> dict:
    """Return {"score": 0..100, "feedback": {...}, "hint": str|None}."""
    label = (label or "").strip()
    is_letter = len(label) == 1 and label.isalpha()
    if not is_letter:
        return {"score": WORD_SCORE, "feedback": {"shape": "n/a"},
                "hint": "Words are checked by the word model. Shape feedback is for single letters."}

    vector = _flat_hand(landmarks)
    if vector is None:
        return {"score": 0.0, "feedback": {"overall": "fail"},
                "hint": "No hand was found. Show your whole hand to the camera and try again."}
    try:
        learner = shape_of(vector)
    except ValueError:
        return {"score": 0.0, "feedback": {"overall": "fail"},
                "hint": "Your hand is too small in the picture. Move it closer to the camera."}

    examples = letter_examples()
    if reference is not None:
        ref_vector = _flat_hand(reference)
        if ref_vector is None:
            return {"score": 0.0, "feedback": {}, "hint": "The reference shape for this sign is not valid."}
        target_examples = np.array([shape_of(ref_vector)])
    else:
        target_examples = examples.get(label.upper())
        if target_examples is None:
            return {"score": 0.0, "feedback": {}, "hint": f"No reference shape is stored for {label.upper()} yet."}

    dist = min(distance(learner, t) for t in target_examples)
    score = to_score(dist)
    nearest = min(examples, key=lambda k: min(distance(learner, t) for t in examples[k])) if examples else label.upper()
    level = verdict(score)
    # the point that differs most from the closest reference example, named by finger
    closest_ex = min(target_examples, key=lambda t: distance(learner, t))
    worst = int(np.argmax(np.linalg.norm(learner - closest_ex, axis=1)))
    part = FINGERS.get(worst, "hand")
    if level == "ok":
        hint = f"Good {label.upper()} shape."
    elif nearest != label.upper():
        hint = f"This looks more like {nearest}. Check your {part} for {label.upper()}."
    else:
        hint = f"Check your {part}: it differs most from the {label.upper()} shape."
    return {"score": score, "feedback": {"overall": level, "part": part}, "hint": hint}
