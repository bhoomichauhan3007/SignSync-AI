# Member 2 hand-off / integration notes

1. Member 1 owns capture, MediaPipe Holistic landmarks, feature extraction and the exported NPZ.
2. Member 2 owns dataset validation, temporal model training, evaluation, model files and inference.
3. Backend integration should call the predictor with one `(30,1662)` sequence in the exact same feature order. Do not send raw webcam frames to this ML predictor; the Member 1 preprocessing/feature pipeline must run first.
4. Return the predicted word and confidence (top-k probabilities are useful for UI).
5. Before claiming model performance, report the held-out test metrics generated in `artifacts/*_metrics.json`.
6. If the export has an absent label such as `i`, the model cannot predict it reliably until that class has real training examples.
