# SignSync AI - CV module changes

## Bug fixes
- landmark_extraction_holistic.py: 8-second auto-stop and Q-while-recording called save_sample() with 5 args
  (needs 6) -> TypeError, recording lost. Fixed; saves are now atomic (.tmp then rename).
- Left/Right mismatch: recorder processed the UNMIRRORED frame, hand_tracking.py (live) flipped BEFORE
  processing, so hands would be swapped between training and live use. Both now follow
  feature_schema.MIRROR_INPUT_FOR_HOLISTIC.
- image_landmark_extraction.py: Holistic finds no hands on close-up hand images (old report: 10990 of 12598
  without hand). Added Hands-model fallback for the empty slot; counts reported separately. Now has a main guard.

## 258 vs 1662
- landmark_extraction.py option 2 now launches the 1662 holistic recorder; the 258 recorder is option 3 (legacy, warns).
- One schema in feature_schema.py; hand_tracking.py, landmark_extraction_holistic.py, preprocessing.py,
  export_features.py use it (duplicate extractors/mask code removed).

## New
- feature_schema.py, sequence_utils.py, live_predictor.py, verify_schema_compat.py, test_cv_pipeline.py

## Preprocessing / export
- Mask also flags pose landmarks with visibility < 0.5 (features unchanged).
- preprocessing.py rejects sequences with too few frames / too few hand frames / wrong schema_name.
- export_features.py also writes cv_word_dataset.npz (N,30,1662) + masks + lengths (even sub-sampling, zero-pad at end).
- Optional normalize_landmarks() exists but is OFF: ML's data is raw; only enable for both training and inference.
