
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_WORDS = DATA / "raw" / "words" / "MP_Data"
EXPORTS = DATA / "exports"
PROCESSED = DATA / "processed"

EXPECTED_CLASSES = 41
EXPECTED_FRAMES = 30
EXPECTED_FEATURES = 1662


def audit_raw_words():
    print("\n========== RAW WORD DATA ==========")

    if not RAW_WORDS.exists():
        print("[ERROR] Missing:", RAW_WORDS)
        return

    classes = sorted(p for p in RAW_WORDS.iterdir() if p.is_dir())
    total_sequences = 0
    feature_lengths = Counter()
    invalid_files = 0

    for class_dir in classes:
        sequence_dirs = []

        for folder in class_dir.rglob("*"):
            if not folder.is_dir():
                continue

            frames = list(folder.glob("*.npy"))

            if len(frames) == EXPECTED_FRAMES:
                sequence_dirs.append(folder)

        valid_sequences = 0

        for seq_dir in sequence_dirs:
            files = sorted(seq_dir.glob("*.npy"))
            valid = True

            for frame_file in files:
                try:
                    arr = np.load(frame_file, allow_pickle=False)
                    arr = np.asarray(arr).reshape(-1)

                    feature_lengths[len(arr)] += 1

                    if len(arr) != EXPECTED_FEATURES:
                        valid = False
                        invalid_files += 1

                    if not np.isfinite(arr).all():
                        valid = False
                        invalid_files += 1

                except Exception as exc:
                    print("[INVALID]", frame_file, exc)
                    valid = False
                    invalid_files += 1

            if valid:
                valid_sequences += 1

        total_sequences += valid_sequences

        print(
            f"{class_dir.name}: "
            f"{valid_sequences} valid sequences"
        )

    print("\nTotal classes:", len(classes))
    print("Expected classes:", EXPECTED_CLASSES)
    print("Total valid sequences:", total_sequences)
    print("Feature length distribution:", dict(feature_lengths))
    print("Invalid frame files:", invalid_files)

    if len(classes) != EXPECTED_CLASSES:
        print("[WARNING] Class count mismatch!")

    if feature_lengths and set(feature_lengths) != {EXPECTED_FEATURES}:
        print("[WARNING] Feature length mismatch!")


def audit_npz(file_path):

    print(f"\n========== NPZ: {file_path.name} ==========")

    if not file_path.exists():
        print("[MISSING]", file_path)
        return

    try:
        with np.load(file_path, allow_pickle=False) as data:

            print("Keys:", data.files)

            for key in data.files:
                arr = data[key]

                print(
                    f"{key}: shape={arr.shape}, "
                    f"dtype={arr.dtype}"
                )

                if arr.size == 0:
                    print("[WARNING] Empty array:", key)

                if arr.dtype.kind in "iuf":
                    print(
                        "Non-finite values:",
                        int((~np.isfinite(arr)).sum())
                    )

            if "X" in data.files:

                X = data["X"]

                if X.ndim == 3:
                    print("Sequence shape:", X.shape)

                    if X.shape[1] != EXPECTED_FRAMES:
                        print("[WARNING] Frame count mismatch")

                    if X.shape[2] != EXPECTED_FEATURES:
                        print("[WARNING] Feature count mismatch")

                    if X.dtype.kind in "iuf":
                        print(
                            "All-zero sequences:",
                            int(np.all(X == 0, axis=(1, 2)).sum())
                        )

                        hashes = [
                            hashlib.sha256(
                                np.ascontiguousarray(s).tobytes()
                            ).hexdigest()
                            for s in X
                        ]

                        print(
                            "Exact duplicates:",
                            len(hashes) - len(set(hashes))
                        )

                elif X.ndim == 2:
                    print("2D dataset detected (not word sequences).")

                else:
                    print("[WARNING] Unexpected X dimensions:", X.ndim)

            if "y" in data.files:
                y = data["y"]
                print(
                    "Label distribution:",
                    dict(Counter(y.reshape(-1).tolist()))
                )

            if "classes" in data.files:
                classes = [str(c) for c in data["classes"].tolist()]

                print("Classes:", classes)
                print("Class count:", len(classes))

                if len(classes) != EXPECTED_CLASSES:
                    print("[WARNING] Expected 41 classes.")

    except Exception as exc:
        print("[ERROR] NPZ audit failed:", exc)


def main():

    print("========== SIGNSYNC DATASET AUDIT ==========")

    for path in [
        DATA,
        DATA / "raw",
        RAW_WORDS,
        PROCESSED,
        EXPORTS
    ]:
        print(f"{'OK' if path.exists() else 'MISSING'}: {path}")

    audit_raw_words()

    audit_npz(EXPORTS / "word_dataset.npz")
    audit_npz(PROCESSED / "word_dataset.npz")
    audit_npz(EXPORTS / "alphabet_dataset.npz")

    print("\n========== AUDIT FINISHED ==========")


if __name__ == "__main__":
    main()
