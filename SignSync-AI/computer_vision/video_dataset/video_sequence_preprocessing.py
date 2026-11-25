"""Turn each video's features into exactly 30 frames, with a mask (Member 1, integrated and fixed).

Fix in this integration: the upload interpolated between frames. Where a hand was missing in one frame and
present in the next, that created half-visible hands that never existed, and every frame was marked valid.
Now the shared fit_to_length() from sequence_utils is used: real frames are kept, short videos are padded
at the end, and missing landmarks get mask 0.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CV_DIR = HERE.parent
ROOT = CV_DIR.parent
sys.path.insert(0, str(CV_DIR))

from sequence_utils import fit_to_length  # noqa: E402
from feature_schema import build_landmark_mask  # noqa: E402

INPUT_DIR = ROOT / "data" / "video_features_raw"
OUTPUT_DIR = ROOT / "data" / "video_sequences"
TARGET_FRAMES = 30


def preprocess_dataset():
    files = sorted(INPUT_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"No feature files in {INPUT_DIR}. Run video_dataset_extractor.py first.")
    X, M, labels, names, lengths = [], [], [], [], []
    skipped = 0
    for path in files:
        try:
            data = np.load(path, allow_pickle=True)
            features = data["features"]
            mask = build_landmark_mask(features)
            seq, seq_mask, n_real = fit_to_length(features, mask, TARGET_FRAMES)
        except Exception as exc:  # report and keep going
            skipped += 1
            print(f"[SKIPPED] {path.name}: {exc}")
            continue
        X.append(seq); M.append(seq_mask)
        labels.append(str(data["label"])); names.append(str(data["video_name"])); lengths.append(n_real)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "isl_word_sequences.npz"
    np.savez_compressed(out, X=np.asarray(X, dtype=np.float32), M=np.asarray(M, dtype=np.uint8),
                        y=np.asarray(labels), video_names=np.asarray(names),
                        original_frame_lengths=np.asarray(lengths, dtype=np.int32),
                        target_frames=TARGET_FRAMES, feature_count=1662)
    print(f"Sequences: {len(X)}  skipped: {skipped}  saved: {out}")


if __name__ == "__main__":
    preprocess_dataset()
