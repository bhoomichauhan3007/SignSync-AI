"""Live word recognition from a webcam (Member 2). Replaces the broken original (kept as
live_word_recognition_original.py, which imported a placeholder module that does not exist).

What it does:
1. Opens the webcam and records 30 frames (about two seconds) with MediaPipe Holistic (legacy API, 0.10.21 as in Member 1).
2. Builds the 1662 values per frame in Member 1's order: pose (33x4), face (468x3), left hand (21x3), right hand (21x3).
3. Normalises with artifacts/normalization.npz and classifies with artifacts/word_gru.keras.
4. Prints the top 3 words with their confidence. Press Q to quit, R to record again.

Run from this folder:  python live_word_recognition.py --model artifacts/word_gru.keras
NOT TESTED in the build environment (needs a webcam, MediaPipe 0.10.21 and TensorFlow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FRAMES = 30
HERE = Path(__file__).resolve().parent


def frame_vector(results) -> np.ndarray:
    """Member 1's layout. Missing parts are zeros."""
    def points(lms, count, per):
        if lms is None:
            return np.zeros(count * per, dtype=np.float32)
        vals = []
        for p in lms.landmark:
            vals.extend([p.x, p.y, p.z] + ([p.visibility] if per == 4 else []))
        return np.asarray(vals, dtype=np.float32)
    return np.concatenate([
        points(results.pose_landmarks, 33, 4),
        points(results.face_landmarks, 468, 3),
        points(results.left_hand_landmarks, 21, 3),
        points(results.right_hand_landmarks, 21, 3),
    ])


def main() -> None:
    import cv2
    import mediapipe as mp
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(HERE / "artifacts" / "word_gru.keras"))
    parser.add_argument("--artifacts", default=str(HERE / "artifacts"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    with np.load(Path(args.artifacts) / "normalization.npz") as stats:
        mean, std = stats["mean"], stats["std"]
    labels = json.loads((Path(args.artifacts) / "label_encoder.json").read_text(encoding="utf-8"))

    cap = cv2.VideoCapture(0)
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("Press R to record a word, Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera not available.")
            break
        cv2.imshow("SignSync word recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key != ord("r"):
            continue
        frames = []
        while len(frames) < FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame_vector(results))
            cv2.imshow("SignSync word recognition", frame)
            cv2.waitKey(1)
        if len(frames) < FRAMES:
            print("Recording was too short. Try again.")
            continue
        sequence = np.stack(frames)
        probs = model.predict(((sequence - mean) / std)[None, ...], verbose=0)[0]
        for i in np.argsort(probs)[::-1][: max(1, args.top_k)]:
            print(f"{labels[str(int(i))]}: {probs[i] * 100:.1f}%")
        print("---")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
