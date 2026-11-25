"""Extract 1662-value holistic features from word videos (Member 1, integrated and fixed).

Input : a folder of word videos. Set the folder with the environment variable SIGNSYNC_VIDEO_DATASET,
        for example:  set SIGNSYNC_VIDEO_DATASET=D:\\my_videos      (Windows)
        The word is taken from the name of the folder that contains each video.
Output: ../../data/video_features_raw/<word>_<video>.npz  (30 frames x 1662 values per video)

Fix in this integration: the upload had a hard-coded path on Member 1's computer. The folder is now
set by the environment variable, and the feature code is imported from computer_vision/ (one level up).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CV_DIR = HERE.parent                       # computer_vision/
ROOT = CV_DIR.parent                       # SignSync-AI/
sys.path.insert(0, str(CV_DIR))

from holistic_features import create_holistic, extract_frame_features  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TARGET_FRAMES = 30
FEATURE_COUNT = 1662
OUTPUT_DIR = ROOT / "data" / "video_features_raw"


def video_root() -> Path:
    value = os.getenv("SIGNSYNC_VIDEO_DATASET", "").strip()
    if not value:
        raise SystemExit("Set SIGNSYNC_VIDEO_DATASET to the folder that holds the word videos.")
    path = Path(value)
    if not path.is_dir():
        raise SystemExit(f"Video folder not found: {path}")
    return path


def process_video(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[FAILED] Cannot open: {video_path.name}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print(f"[SKIPPED] Invalid video: {video_path.name}")
        return None
    indices = np.linspace(0, total - 1, min(TARGET_FRAMES, total), dtype=int)
    fps = cap.get(cv2.CAP_PROP_FPS)
    features, timestamps = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            vector = extract_frame_features(holistic.process(rgb))
        finally:
            rgb.flags.writeable = True
        if vector.shape != (FEATURE_COUNT,):
            continue
        features.append(vector)
        timestamps.append(idx / fps if fps > 0 else 0.0)
    cap.release()
    if not features:
        print(f"[FAILED] No features extracted: {video_path.name}")
        return None
    return np.asarray(features, dtype=np.float32), np.asarray(timestamps, dtype=np.float32)


def extract_dataset():
    source = video_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = [v for v in source.rglob("*") if v.suffix.lower() in VIDEO_EXTENSIONS]
    print(f"Videos found: {len(videos)}")
    ok = failed = 0
    with create_holistic() as holistic:
        for video in videos:
            label = video.parent.name
            out = OUTPUT_DIR / f"{label}_{video.stem}.npz"
            if out.exists():
                continue
            result = process_video(video, holistic)
            if result is None:
                failed += 1
                continue
            features, timestamps = result
            np.savez_compressed(out, features=features, timestamps=timestamps, label=label,
                                video_name=video.name, feature_count=FEATURE_COUNT)
            ok += 1
    print(f"Successful: {ok}  Failed: {failed}  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_dataset()
