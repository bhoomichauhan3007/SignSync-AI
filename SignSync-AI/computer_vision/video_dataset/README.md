# Word dataset from videos (Member 1, integrated)

Run these from the computer_vision/video_dataset folder, in this order:

1. `set SIGNSYNC_VIDEO_DATASET=<folder with word videos>` (Windows) — needs the videos, which are not in the project
2. `python video_dataset_extractor.py` — videos -> data/video_features_raw (needs MediaPipe 0.10.21)
3. `python video_dataset_audit.py` — checks every file
4. `python video_sequence_preprocessing.py` — 30 frames each, with masks -> data/video_sequences
5. `python video_dataset_split.py` — train / validation / test, split by recording session -> data/final_dataset
6. `python validate_final_dataset.py` — must print ALL CHECKS PASSED

Steps 3 to 6 work on the feature files that are already in data/video_features_raw (no videos needed).

Licence: the source videos (ISL500, CISLR) are not in this project. Their licence and terms are not stated in
the upload. Check them before any use outside the team (see docs/DATASET_REPORT.md).

Data note: the feature files, the split and the final dataset are NOT in the project ZIP (the source datasets forbid redistribution).
To regenerate them, copy data/video_features_raw from the upload new__1_.zip into SignSync-AI/data/video_features_raw
(research and education use only), then run steps 3 to 6.
