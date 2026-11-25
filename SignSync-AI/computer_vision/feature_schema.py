"""
SignSync AI - Single source of truth for the CV -> ML feature schema.

Schema "holistic-1662-v1" (one frame):

    [  0: 132]  Pose        33 landmarks x (x, y, z, visibility)
    [132:1536]  Face       468 landmarks x (x, y, z)
    [1536:1599] Left hand   21 landmarks x (x, y, z)
    [1599:1662] Right hand  21 landmarks x (x, y, z)

Values are raw MediaPipe normalized coordinates (no normalization applied),
missing landmark groups are all zeros. A parallel 0/1 mask marks what is real.

Every recorder, preprocessing step, exporter and live predictor must import
from this file instead of re-declaring numbers.
"""

import numpy as np

SCHEMA_NAME = "holistic-1662-v1"
FEATURE_ORDER = "pose_face_left_hand_right_hand"

POSE_LANDMARKS, POSE_DIMS = 33, 4
FACE_LANDMARKS, FACE_DIMS = 468, 3
HAND_LANDMARKS, HAND_DIMS = 21, 3

POSE_FEATURES = POSE_LANDMARKS * POSE_DIMS      # 132
FACE_FEATURES = FACE_LANDMARKS * FACE_DIMS      # 1404
HAND_FEATURES = HAND_LANDMARKS * HAND_DIMS      # 63
FEATURES_PER_FRAME = POSE_FEATURES + FACE_FEATURES + 2 * HAND_FEATURES  # 1662

# name -> (start, end, values_per_landmark)
SECTIONS = {
    "pose": (0, 132, 4),
    "face": (132, 1536, 3),
    "left_hand": (1536, 1599, 3),
    "right_hand": (1599, 1662, 3),
}

# Word-level sequence format used by the ML dataset (MP_Data / word_dataset.npz)
SEQUENCE_LENGTH = 30

# A pose landmark below this visibility is treated as "not reliably seen".
POSE_VISIBILITY_THRESHOLD = 0.5

# Left/right convention.
# MediaPipe Holistic assumes a mirrored (selfie) image. Every sample recorded
# so far (landmark_extraction_holistic.py) was produced from the UNMIRRORED
# camera frame, so live inference must do the same or Left/Right swap.
# NOTE: confirm with the ML member that MP_Data used the same convention.
MIRROR_INPUT_FOR_HOLISTIC = False

# Words are rejected as training data when they have these labels.
EXCLUDED_LABELS = {"camera_test", "test", "testing", "debug"}


def is_excluded_label(label):
    label = str(label).strip().lower()
    return (not label) or label in EXCLUDED_LABELS or label.startswith("test_")


def section(features, name):
    """Return the slice of a (..., 1662) array that belongs to `name`."""
    start, end, _ = SECTIONS[name]
    return features[..., start:end]


def check_feature_array(features, name="features"):
    """Raise ValueError unless `features` is a finite (T, 1662) array."""
    arr = np.asarray(features)
    if arr.ndim != 2 or arr.shape[1] != FEATURES_PER_FRAME:
        raise ValueError(
            f"{name}: expected shape (T, {FEATURES_PER_FRAME}), got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name}: empty sequence")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains NaN or Inf")
    return arr


def build_landmark_mask(features, pose_visibility_threshold=POSE_VISIBILITY_THRESHOLD):
    """
    Per-value 0/1 mask, same shape as `features` (T, 1662).

    1 = landmark was detected and is usable, 0 = missing/unreliable.
      * Any landmark whose values are all zero is missing.
      * Pose landmarks with visibility < threshold are unreliable.
    The feature values themselves are NOT modified.
    """
    features = np.asarray(features)
    mask = np.ones(features.shape, dtype=np.uint8)

    for name, (start, end, step) in SECTIONS.items():
        block = features[:, start:end].reshape(features.shape[0], -1, step)
        missing = np.all(block == 0, axis=2)                    # (T, landmarks)
        if name == "pose":
            missing |= block[:, :, 3] < pose_visibility_threshold
        block_mask = np.repeat((~missing)[:, :, None], step, axis=2)
        mask[:, start:end] = block_mask.reshape(features.shape[0], -1).astype(np.uint8)

    return mask


def hand_presence(features):
    """Boolean arrays (left, right) of shape (T,) - was each hand detected?"""
    features = np.asarray(features)
    left = np.any(section(features, "left_hand") != 0, axis=-1)
    right = np.any(section(features, "right_hand") != 0, axis=-1)
    return left, right
