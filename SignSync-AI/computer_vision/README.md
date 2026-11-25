
# SignSync AI – Computer Vision Module

## Overview
This module captures webcam video, detects hand and body landmarks, records landmark data, preprocesses it, and exports numerical features for the AI/ML module.

## Technologies
- Python 3.11
- OpenCV
- MediaPipe
- NumPy
- Pandas

## Project Structure
```text
SignSync-AI/
├── computer_vision/
│   ├── camera.py
│   ├── hand_tracking.py
│   ├── pose_estimation.py
│   ├── landmark_extraction.py
│   ├── landmark_extraction_holistic.py   # standard word recorder (1662 features)
│   ├── holistic_features.py              # shared per-frame extractor
│   ├── feature_schema.py                 # single source of truth for the schema
│   ├── sequence_utils.py                 # validation / 30-frame fitting / optional normalization
│   ├── preprocessing.py
│   ├── export_features.py
│   ├── live_predictor.py                 # webcam -> 30-frame window -> model
│   ├── verify_schema_compat.py           # CV vs ML dataset check
│   └── test_cv_pipeline.py
├── data/
│   ├── samples/
│   ├── processed/
│   └── exports/
├── tests/
├── requirements.txt
└── README.md
```

## Environment Setup

Create a virtual environment:

```bash
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Module Workflow

1. `camera.py` – Webcam handling.
2. `hand_tracking.py` – Hand landmark detection.
3. `pose_estimation.py` – Body pose estimation.
4. `landmark_extraction.py` – Records landmark sequences.
5. `preprocessing.py` – Cleans recorded data.
6. `export_features.py` – Exports numerical features and masks.

## Feature Schema (holistic-1662-v1)

Defined once in `feature_schema.py`: Pose 132 + Face 1404 + Left hand 63 + Right hand 63 = 1662 per frame,
raw MediaPipe coordinates, missing landmarks = zeros (a parallel 0/1 mask marks them).
The old 258-feature recorder (hands + pose) is legacy and its files are rejected by `preprocessing.py`.

## Commands (run inside `computer_vision/`)

```bash
python landmark_extraction_holistic.py   # record words (R start, S save, N label, Q quit)
python preprocessing.py                  # CSV -> *_processed.npz (+ mask), validates each sample
python export_features.py                # -> X_sequences.npy (raw) and cv_word_dataset.npz (N,30,1662)
python verify_schema_compat.py           # compare with ../data/exports/word_dataset.npz
python test_cv_pipeline.py               # logic tests on synthetic data
python live_predictor.py --no-model      # live capture only
python live_predictor.py --model m.keras --classes ../data/exports/word_dataset.npz
```

## Data Output

- `data/samples/` – Raw landmark CSV recordings.
- `data/processed/` – Preprocessed NumPy files.
- `data/exports/` – `X_sequences.npy`/`X_masks.npy` (variable length), `cv_word_dataset.npz` (fixed 30 frames), manifest.

## Important
- Recorder and live inference must use the same left/right convention: `MIRROR_INPUT_FOR_HOLISTIC` in `feature_schema.py`
  (currently `False` = unmirrored frame, as all recordings so far). Confirm it matches the ML member's MP_Data.
- `cv_word_dataset.npz` never overwrites the ML team's `word_dataset.npz`; merge by class NAME, not class index.
- Live prediction with a real trained model is not yet tested (no model available).
