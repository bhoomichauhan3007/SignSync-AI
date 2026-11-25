import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()
    root = Path(args.data_dir)
    if not root.exists():
        print("Data directory does not exist:", root.resolve()); return
    paths = list(root.rglob("*"))
    files = [f for f in paths if f.is_file() and f.suffix.lower() in {".npz", ".npy", ".csv"}]
    if not files:
        print("No .npz/.npy/.csv files found in", root.resolve()); return
    for f in files:
        print("\\n" + "="*70, "\\nFILE:", f)
        try:
            if f.suffix.lower() == ".npz":
                z = np.load(f, allow_pickle=True)
                for k in z.files:
                    a = z[k]
                    print(f"  {k}: shape={a.shape}, dtype={a.dtype}")
                    if a.size and a.ndim == 1:
                        print("    unique labels (sample):", np.unique(a)[:20])
            elif f.suffix.lower() == ".npy":
                a = np.load(f, allow_pickle=True)
                print("  shape=", a.shape, "dtype=", a.dtype)
            else:
                df = pd.read_csv(f, nrows=5)
                print("  columns:", list(df.columns))
                print(df.head())
        except Exception as e:
            print("  ERROR:", repr(e))

if __name__ == "__main__":
    main()
