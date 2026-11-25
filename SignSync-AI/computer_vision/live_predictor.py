"""
SignSync AI - Live CV -> ML bridge.

Webcam -> Holistic -> 1662-feature frame -> 30-frame sliding window -> model
-> smoothed word prediction.

The decision logic lives in `LivePredictor` (no camera / MediaPipe needed, so it
can be unit-tested). `main()` wires it to the webcam.

Usage (from computer_vision/):
    python live_predictor.py --model path/to/model.keras --classes path/to/word_dataset.npz
    python live_predictor.py --no-model        # feature capture only, no prediction

IMPORTANT: no trained model is bundled. Until the ML member provides one, the
model path of this file is IMPLEMENTED but NOT TESTED against a real model.
"""

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from feature_schema import FEATURES_PER_FRAME, SEQUENCE_LENGTH
from sequence_utils import hand_frame_ratio


class LivePredictor:
    """
    Keeps a sliding window of the latest frames and turns it into a stable word.

    predict_fn : callable((30, 1662) float32 array) -> 1D probability vector
    classes    : list of class names, index-aligned with the probability vector
    """

    def __init__(
        self,
        predict_fn,
        classes,
        window=SEQUENCE_LENGTH,
        min_hand_ratio=0.6,
        min_confidence=0.7,
        stable_votes=3,
        predict_every=5,
    ):
        self.predict_fn = predict_fn
        self.classes = list(classes)
        self.window = window
        self.min_hand_ratio = min_hand_ratio
        self.min_confidence = min_confidence
        self.predict_every = predict_every
        self.buffer = deque(maxlen=window)
        self.votes = deque(maxlen=stable_votes)
        self.frames_seen = 0
        self.last = None            # (label, confidence) of last raw prediction
        self.stable_word = None     # only set when `stable_votes` agree

    def push(self, features):
        """Add one frame. Returns the stable word if it changed, else None."""
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (FEATURES_PER_FRAME,):
            raise ValueError(
                f"Expected ({FEATURES_PER_FRAME},), got {features.shape}"
            )

        self.buffer.append(features)
        self.frames_seen += 1

        if len(self.buffer) < self.window:
            return None

        if self.frames_seen % self.predict_every != 0:
            return None

        sequence = np.stack(self.buffer)

        # Do not predict while the signer's hands are out of view.
        if hand_frame_ratio(sequence) < self.min_hand_ratio:
            self.votes.clear()
            self.last = None
            return None

        probabilities = np.asarray(self.predict_fn(sequence), dtype=np.float64).reshape(-1)

        if probabilities.shape[0] != len(self.classes):
            raise ValueError(
                f"Model returned {probabilities.shape[0]} scores but there are "
                f"{len(self.classes)} classes - wrong class list or wrong model."
            )

        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        self.last = (self.classes[best], confidence)

        if confidence < self.min_confidence:
            self.votes.clear()
            return None

        self.votes.append(best)

        if len(self.votes) == self.votes.maxlen and len(set(self.votes)) == 1:
            word = self.classes[best]
            if word != self.stable_word:
                self.stable_word = word
                return word

        return None


def load_keras_predictor(model_path):
    """Load a Keras model lazily (TensorFlow is optional for the CV module)."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is not installed. Install it or use --no-model."
        ) from exc

    model = keras.models.load_model(model_path)
    expected = (None, SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    if tuple(model.input_shape) != expected:
        raise SystemExit(
            f"Model input shape {model.input_shape} does not match the CV output "
            f"{expected}. Agree on one schema with the ML member first."
        )

    def predict(sequence):
        return model.predict(sequence[np.newaxis, ...], verbose=0)[0]

    return predict


def load_classes(path):
    with np.load(path, allow_pickle=False) as data:
        if "classes" not in data.files:
            raise SystemExit(f"No 'classes' array in {path}")
        return [str(c) for c in data["classes"]]


def main():
    import cv2
    from holistic_features import create_holistic, process_frame

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--classes", type=Path, help="npz containing a 'classes' array")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    predictor = None

    if not args.no_model:
        if not args.model or not args.classes:
            raise SystemExit("Give --model and --classes, or use --no-model.")
        predictor = LivePredictor(load_keras_predictor(args.model), load_classes(args.classes))

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    buffer_only = deque(maxlen=SEQUENCE_LENGTH)
    sentence = []

    try:
        with create_holistic() as holistic:
            while True:
                ok, frame = cap.read()

                if not ok:
                    print("Camera frame failed.")
                    break

                features, _, display = process_frame(frame, holistic)

                if predictor is not None:
                    word = predictor.push(features)
                    if word:
                        sentence.append(word)
                        print("WORD:", word, flush=True)
                    status = (
                        f"{predictor.last[0]} {predictor.last[1]:.0%}"
                        if predictor.last else "..."
                    )
                else:
                    buffer_only.append(features)
                    status = f"capture only {len(buffer_only)}/{SEQUENCE_LENGTH}"

                display = cv2.flip(display, 1)
                cv2.putText(display, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 180), 2)
                cv2.putText(display, " ".join(sentence[-6:]), (15, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("SignSync AI - Live", display)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
