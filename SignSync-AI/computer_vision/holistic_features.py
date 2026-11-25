
"""
SignSync AI - Holistic Feature Extractor

Feature schema per frame:
1. Pose       : 33 x 4 = 132
2. Face      : 468 x 3 = 1404
3. Left Hand : 21 x 3 = 63
4. Right Hand: 21 x 3 = 63

Total = 1662 features/frame
"""

import cv2
import numpy as np
import mediapipe as mp

# Constants live in feature_schema.py (single source of truth).
from feature_schema import (
    FEATURES_PER_FRAME,
    POSE_FEATURES,
    FACE_FEATURES,
    HAND_FEATURES,
    MIRROR_INPUT_FOR_HOLISTIC,
)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def _pose_features(landmarks):
    """Extract pose landmarks as x, y, z, visibility."""
    if landmarks is None or len(landmarks.landmark) != 33:
        return np.zeros(POSE_FEATURES, dtype=np.float32)

    values = []

    for lm in landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.asarray(values, dtype=np.float32)


def _face_features(landmarks):
    """Extract 468 face landmarks as x, y, z."""
    if landmarks is None or len(landmarks.landmark) != 468:
        return np.zeros(FACE_FEATURES, dtype=np.float32)

    values = []

    for lm in landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])

    return np.asarray(values, dtype=np.float32)


def _hand_features(landmarks):
    """Extract 21 hand landmarks as x, y, z."""
    if landmarks is None or len(landmarks.landmark) != 21:
        return np.zeros(HAND_FEATURES, dtype=np.float32)

    values = []

    for lm in landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])

    return np.asarray(values, dtype=np.float32)


def extract_frame_features(results):
    """Convert Holistic results into a fixed 1662-feature vector."""

    if results is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

    pose = _pose_features(results.pose_landmarks)
    face = _face_features(results.face_landmarks)
    left_hand = _hand_features(results.left_hand_landmarks)
    right_hand = _hand_features(results.right_hand_landmarks)

    features = np.concatenate([
        pose,
        face,
        left_hand,
        right_hand
    ]).astype(np.float32)

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if features.shape != (FEATURES_PER_FRAME,):
        raise ValueError(
            f"Expected 1662 features, got {features.shape}"
        )

    return features


def create_holistic():
    """Create MediaPipe Holistic detector."""

    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,

        # Keep 468 face landmarks for the 1662 schema.
        refine_face_landmarks=False,

        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def process_frame(frame, holistic, mirror=MIRROR_INPUT_FOR_HOLISTIC):
    """
    Process one webcam frame.

    `mirror` must match the convention used when the training data was recorded
    (see feature_schema.MIRROR_INPUT_FOR_HOLISTIC). Recorder and live inference
    both go through this function so they cannot disagree.

    Returns:
        features     : (1662,) float32
        results      : MediaPipe Holistic results
        display_frame: annotated BGR frame (same orientation that was processed)
    """

    if frame is None or frame.size == 0:
        raise ValueError("Invalid camera frame")

    model_frame = cv2.flip(frame, 1) if mirror else frame

    rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    try:
        results = holistic.process(rgb)
    finally:
        rgb.flags.writeable = True

    display_frame = model_frame.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            display_frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            display_frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            display_frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    features = extract_frame_features(results)

    return features, results, display_frame


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    try:
        with create_holistic() as holistic:

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Failed to read camera frame.")
                    break

                features, results, display = process_frame(
                    frame, holistic
                )

                cv2.putText(
                    display,
                    f"Features: {features.shape[0]}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("SignSync Holistic", display)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
