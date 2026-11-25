from pathlib import Path
import numpy as np

def load_npz(path, x_key=None, y_key=None):
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)
    print("Available keys:", keys)
    if x_key is None:
        candidates = ["X", "x", "features", "X_sequences", "X_train", "data"]
        x_key = next((k for k in candidates if k in keys), None)
    if y_key is None:
        candidates = ["y", "Y", "labels", "y_train", "label", "targets"]
        y_key = next((k for k in candidates if k in keys), None)
    if x_key is None or x_key not in keys:
        raise KeyError(f"Cannot identify feature key. Available keys: {keys}. Pass --x_key.")
    X = np.asarray(z[x_key])
    y = np.asarray(z[y_key]) if y_key and y_key in keys else None
    return X, y, keys

def normalize_labels(y):
    y = np.asarray(y)
    if y.ndim > 1:
        if y.shape[-1] > 1:
            y = y.argmax(axis=-1)
        else:
            y = y.reshape(-1)
    return y.reshape(-1)

def flatten_features(X):
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)

def make_output_dirs():
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
