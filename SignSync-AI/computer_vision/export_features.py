
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
EXPORT_DIR = ROOT / "data" / "exports"

from feature_schema import (
    FEATURES_PER_FRAME,
    SCHEMA_NAME,
    SEQUENCE_LENGTH,
    is_excluded_label,
)
from sequence_utils import fit_to_length, validate_sequence

ALPHABET_FEATURES = 63
WORD_FEATURES = FEATURES_PER_FRAME


def export_alphabets():
    source = PROCESSED_DIR / "alphabet_landmarks.csv"

    if not source.exists():
        print("Alphabet landmark CSV not found.")
        return

    print("\nExporting alphabet dataset...")
    df = pd.read_csv(source)

    required = ["hand_detected", "label", "split", "image_path"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print("[ERROR] Missing columns:", missing)
        return

    feature_columns = [
        f"hand_{point}_{axis}"
        for point in range(21)
        for axis in ("x", "y", "z")
    ]

    missing_features = [
        col for col in feature_columns if col not in df.columns
    ]

    if missing_features:
        print("[ERROR] Missing feature columns:", missing_features[:10])
        print("Total missing:", len(missing_features))
        return

    df = df[df["hand_detected"] == 1].copy()
    df = df.dropna(subset=["label", "split", "image_path"])

    if df.empty:
        print("[WARNING] No valid alphabet samples found.")
        return

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    splits = df["split"].astype(str).to_numpy()
    paths = df["image_path"].astype(str).to_numpy()

    if X.shape[1] != ALPHABET_FEATURES:
        print("[ERROR] Incorrect alphabet feature count:", X.shape)
        return

    if not np.isfinite(X).all():
        print("[WARNING] NaN or Inf found in alphabet features.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = EXPORT_DIR / "alphabet_dataset.npz"

    np.savez_compressed(
        output,
        X=X,
        y=y,
        splits=splits,
        image_paths=paths,
        feature_names=np.array(feature_columns, dtype=str)
    )

    manifest_columns = ["split", "label", "image_path"]

    for column in ["handedness", "confidence"]:
        if column in df.columns:
            manifest_columns.append(column)

    df[manifest_columns].to_csv(
        EXPORT_DIR / "alphabet_manifest.csv",
        index=False
    )

    print("Alphabet samples:", len(X))
    print("Feature count:", X.shape[1])
    print("Labels:", len(np.unique(y)))
    print("Saved:", output)


def export_recorded_sequences():

    files = sorted(PROCESSED_DIR.glob("*_processed.npz"))

    if not files:
        print("\nNo processed webcam recordings found.")
        return

    samples = []
    manifest = []

    for file_path in files:

        try:
            with np.load(file_path, allow_pickle=False) as data:

                required_keys = {"features", "mask", "label"}
                missing_keys = required_keys.difference(data.files)

                if missing_keys:
                    raise ValueError(f"Missing keys: {missing_keys}")

                features = np.asarray(data["features"], dtype=np.float32)
                mask = np.asarray(data["mask"])

                if features.ndim != 2:
                    raise ValueError(
                        f"Invalid feature dimensions: {features.shape}"
                    )

                if features.shape[1] != WORD_FEATURES:
                    raise ValueError(
                        f"Expected 1662 features, got {features.shape[1]}"
                    )

                if mask.shape != features.shape:
                    raise ValueError(
                        f"Feature/mask mismatch: {features.shape} vs {mask.shape}"
                    )

                if features.shape[0] == 0:
                    raise ValueError("Empty sequence")

                if not np.isfinite(features).all():
                    raise ValueError("NaN or Inf detected")

                label = str(data["label"].item()).strip().lower()

                if is_excluded_label(label):
                    print("Skipped test/debug sample:", file_path.name)
                    continue

                problems = validate_sequence(features)

                if problems:
                    raise ValueError("; ".join(problems))

                sample_id = (
                    str(data["sample_id"].item())
                    if "sample_id" in data.files
                    else file_path.stem
                )

                signer_id = (
                    str(data["signer_id"].item())
                    if "signer_id" in data.files
                    else "unknown"
                )

                recording_type = (
                    str(data["recording_type"].item())
                    if "recording_type" in data.files
                    else "unknown"
                )

                samples.append({
                    "features": features,
                    "mask": mask.astype(np.float32)
                })

                manifest.append({
                    "sample_id": sample_id,
                    "signer_id": signer_id,
                    "recording_type": recording_type,
                    "label": label,
                    "frames": features.shape[0],
                    "feature_count": features.shape[1],
                    "processed_file": file_path.name
                })

        except Exception as error:
            print("Skipped:", file_path.name, "-", error)

    if not samples:
        print("No valid sequences found.")
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.empty(len(samples), dtype=object)
    M = np.empty(len(samples), dtype=object)

    for i, sample in enumerate(samples):
        X[i] = sample["features"]
        M[i] = sample["mask"]

    np.save(EXPORT_DIR / "X_sequences.npy", X, allow_pickle=True)
    np.save(EXPORT_DIR / "X_masks.npy", M, allow_pickle=True)

    pd.DataFrame(manifest).to_csv(
        EXPORT_DIR / "sample_manifest.csv",
        index=False
    )

    print("\nRecorded sequences:", len(samples))
    print("Unique labels:", len({item["label"] for item in manifest}))
    print("Feature dimensions:", sorted({
        item["feature_count"] for item in manifest
    }))

    print("NOTE: X_sequences.npy keeps variable frame counts (raw recordings).")

    export_fixed_length_dataset(samples, manifest)


def export_fixed_length_dataset(samples, manifest, length=SEQUENCE_LENGTH):
    """
    Build the model-ready word dataset: X (N, 30, 1662), same layout as the ML
    team's word_dataset.npz, plus masks and per-sample lengths.

    Written to cv_word_dataset.npz - it NEVER overwrites word_dataset.npz.
    Longer recordings are evenly sub-sampled (real frames only); shorter ones
    are zero-padded at the end with mask 0 (see sequence_utils.fit_to_length).
    """

    X, M, lengths = [], [], []

    for sample in samples:
        f, m, n_real = fit_to_length(
            sample["features"], np.asarray(sample["mask"]).astype(np.uint8), length
        )
        X.append(f)
        M.append(m)
        lengths.append(n_real)

    X = np.stack(X).astype(np.float32)
    M = np.stack(M).astype(np.uint8)

    labels = [item["label"] for item in manifest]
    classes = sorted(set(labels))
    class_to_index = {name: i for i, name in enumerate(classes)}
    y = np.array([class_to_index[name] for name in labels], dtype=np.int32)

    assert X.shape[1:] == (length, FEATURES_PER_FRAME), X.shape

    output = EXPORT_DIR / "cv_word_dataset.npz"

    np.savez_compressed(
        output,
        X=X,
        M=M,
        y=y,
        classes=np.array(classes, dtype=str),
        lengths=np.array(lengths, dtype=np.int32),
        signer_ids=np.array([item["signer_id"] for item in manifest], dtype=str),
        sample_ids=np.array([item["sample_id"] for item in manifest], dtype=str),
        schema_name=np.array(SCHEMA_NAME),
    )

    padded = int(sum(n < length for n in lengths))

    print("\nFixed-length dataset:", output)
    print("X shape:", X.shape, "| classes:", len(classes))
    print("Samples zero-padded (recorded shorter than", length, "frames):", padded)
    print("NOTE: classes here only cover YOUR recordings. Class order differs "
          "from word_dataset.npz unless the ML member merges by class NAME.")


def main():

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    export_alphabets()
    export_recorded_sequences()

    print("\n========== FINAL EXPORT COMPLETE ==========")
    print("Export folder:", EXPORT_DIR)


if __name__ == "__main__":
    main()
