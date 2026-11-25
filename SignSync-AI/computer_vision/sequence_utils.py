"""
SignSync AI - Sequence helpers shared by export, evaluation and live inference.

Training and live inference MUST call the same functions here so a sequence is
treated identically in both places.
"""

import numpy as np

from feature_schema import (
    FEATURES_PER_FRAME,
    SECTIONS,
    SEQUENCE_LENGTH,
    build_landmark_mask,
    check_feature_array,
    hand_presence,
)

# MediaPipe pose indices of left / right shoulder
_L_SHOULDER, _R_SHOULDER = 11, 12


def fit_to_length(features, mask=None, length=SEQUENCE_LENGTH):
    """
    Turn a variable-length (T, 1662) recording into exactly `length` frames.

    * T > length : pick `length` real frames evenly spread over the recording
                   (no interpolation - every output frame is a real frame).
    * T < length : keep all real frames and zero-pad at the END. Padded frames
                   have mask 0, so they are never mistaken for real data.
    * T == length: unchanged.

    Returns (features[length,1662] float32, mask[length,1662] uint8, n_real).
    """
    features = check_feature_array(features).astype(np.float32, copy=False)
    if mask is None:
        mask = build_landmark_mask(features)
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != features.shape:
        raise ValueError(f"mask shape {mask.shape} != features shape {features.shape}")

    t = features.shape[0]
    if t >= length:
        idx = np.linspace(0, t - 1, length).round().astype(int)
        return features[idx], mask[idx], length

    out_f = np.zeros((length, FEATURES_PER_FRAME), dtype=np.float32)
    out_m = np.zeros((length, FEATURES_PER_FRAME), dtype=np.uint8)
    out_f[:t] = features
    out_m[:t] = mask
    return out_f, out_m, t


def hand_frame_ratio(features):
    """Fraction of frames in which at least one hand was detected."""
    left, right = hand_presence(check_feature_array(features))
    return float(np.mean(left | right))


def validate_sequence(features, min_hand_ratio=0.6, min_frames=5):
    """
    Return a list of human-readable problems (empty list = sequence is usable).
    """
    problems = []
    try:
        arr = check_feature_array(features)
    except ValueError as exc:
        return [str(exc)]

    if arr.shape[0] < min_frames:
        problems.append(f"only {arr.shape[0]} frames (min {min_frames})")

    ratio = hand_frame_ratio(arr)
    if ratio < min_hand_ratio:
        problems.append(f"hands detected in {ratio:.0%} of frames (min {min_hand_ratio:.0%})")

    if np.all(arr == 0):
        problems.append("all values are zero")
    return problems


def normalize_landmarks(features):
    """
    OPTIONAL position/scale normalization (NOT applied by default).

    Each frame is shifted so the shoulder midpoint is the origin and divided by
    the shoulder width, for x/y/z of pose, face and both hands. Visibility is
    untouched. Missing landmarks (all zero) stay zero. Frames where the
    shoulders are not both reliably visible are left unnormalized and reported
    through the returned boolean array.

    The ML team's existing word dataset is raw MediaPipe output, so enabling this
    is only valid if the model is trained on normalized data AND live inference
    applies this same function.

    Returns (normalized[T,1662] float32, ok[T] bool).
    """
    feats = check_feature_array(features).astype(np.float32, copy=True)
    pose = feats[:, 0:132].reshape(-1, 33, 4)
    ls, rs = pose[:, _L_SHOULDER], pose[:, _R_SHOULDER]

    ok = (ls[:, 3] >= 0.5) & (rs[:, 3] >= 0.5)
    center = (ls[:, :3] + rs[:, :3]) / 2.0
    width = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1)
    ok &= width > 1e-4

    for t in np.where(ok)[0]:
        for name, (start, end, step) in SECTIONS.items():
            block = feats[t, start:end].reshape(-1, step)
            present = np.any(block[:, :3] != 0, axis=1)
            block[present, :3] = (block[present, :3] - center[t]) / width[t]
            feats[t, start:end] = block.reshape(-1)
    return feats, ok
