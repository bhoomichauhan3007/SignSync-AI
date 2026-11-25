import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder

EXPECTED_FRAMES = 30
EXPECTED_FEATURES = 1662

def load_dataset(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}. Put word_dataset.npz here.")
    with np.load(p, allow_pickle=False) as z:
        if not {'X', 'y'}.issubset(z.files):
            raise ValueError(f"Expected NPZ keys X and y; found {z.files}")
        X = np.asarray(z['X'], dtype=np.float32)
        y = np.asarray(z['y']).reshape(-1)
        classes = np.asarray(z['classes']).astype(str).tolist() if 'classes' in z.files else None
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,T,F); got {X.shape}")
    if X.shape[1:] != (EXPECTED_FRAMES, EXPECTED_FEATURES):
        raise ValueError(f"Expected X shape (N,{EXPECTED_FRAMES},{EXPECTED_FEATURES}), got {X.shape}. Check Member 1 export.")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} samples but y has {len(y)} labels")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Integer IDs are decoded with the exported class list, when available.
    if np.issubdtype(y.dtype, np.integer) and classes:
        if y.min(initial=0) < 0 or y.max(initial=0) >= len(classes):
            raise ValueError('Integer label outside classes array range')
        y = np.asarray([classes[int(i)] for i in y], dtype=str)
    else:
        y = y.astype(str)
    enc = LabelEncoder()
    yi = enc.fit_transform(y)
    if len(enc.classes_) < 2:
        raise ValueError('At least two represented word classes are required')
    counts = {str(c): int((y == c).sum()) for c in enc.classes_}
    return X, yi, enc, counts

def save_labels(encoder, path):
    Path(path).write_text(json.dumps({str(i): str(c) for i,c in enumerate(encoder.classes_)}, indent=2), encoding='utf-8')
