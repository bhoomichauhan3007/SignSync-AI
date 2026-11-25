
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# SIGNSYNC AI - PREPROCESSING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SAMPLES_DIR = ROOT / "data" / "samples"
PROCESSED_DIR = ROOT / "data" / "processed"

# Feature layout and mask logic are shared with the exporter and the live
# predictor (feature_schema.py) so training and inference cannot drift apart.
from feature_schema import (
    FEATURES_PER_FRAME,
    SCHEMA_NAME,
    build_landmark_mask,
    is_excluded_label,
)
from sequence_utils import validate_sequence

EXPECTED_FEATURES = FEATURES_PER_FRAME


def create_landmark_mask(features):
    """Kept for backward compatibility; delegates to the shared implementation."""
    return build_landmark_mask(features)


# ============================================================
# PREPROCESS SINGLE SAMPLE
# ============================================================

def preprocess_sample(csv_path):

    csv_path = Path(csv_path)
    json_path = csv_path.with_suffix(".json")

    # --------------------------------------------------------
    # 1. Check metadata
    # --------------------------------------------------------

    if not json_path.exists():
        raise FileNotFoundError(
            f"Metadata missing: {json_path.name}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_metadata = [
        "sample_id",
        "signer_id",
        "recording_type",
        "label",
    ]

    for key in required_metadata:

        if key not in metadata:
            raise ValueError(
                f"Metadata missing required field: {key}"
            )

    label = str(metadata["label"]).strip().lower()

    if is_excluded_label(label):

        raise ValueError(
            f"Invalid or test label rejected: {label}"
        )

    # Newer recordings carry the schema name; refuse a different one.
    recorded_schema = metadata.get("schema_name")

    if recorded_schema is not None and recorded_schema != SCHEMA_NAME:

        raise ValueError(
            f"Schema mismatch: recording uses {recorded_schema!r}, "
            f"pipeline expects {SCHEMA_NAME!r}"
        )

    if metadata["recording_type"] != "word":

        raise ValueError(
            f"Expected word recording, got: "
            f"{metadata['recording_type']}"
        )

    # --------------------------------------------------------
    # 2. Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV contains no frames.")

    required_csv_columns = {"frame", "timestamp"}

    if not required_csv_columns.issubset(df.columns):

        raise ValueError(
            "CSV must contain frame and timestamp columns."
        )

    # --------------------------------------------------------
    # 3. Validate feature schema
    # --------------------------------------------------------

    expected_columns = [
        f"feature_{i}" for i in range(EXPECTED_FEATURES)
    ]

    feature_columns = [
        column for column in df.columns
        if column not in {"frame", "timestamp"}
    ]

    if feature_columns != expected_columns:

        raise ValueError(
            "Feature schema mismatch!\n"
            f"Expected exactly {EXPECTED_FEATURES} features "
            "in correct order.\n"
            f"Found: {len(feature_columns)} features.\n"
            "Required order: Pose + Face + Left Hand + Right Hand."
        )

    if len(df) < 1:
        raise ValueError("No frame data available.")

    # --------------------------------------------------------
    # 4. Convert features to numeric
    # --------------------------------------------------------

    numeric = df[expected_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    raw_features = numeric.to_numpy(dtype=np.float32)

    # Valid numeric values mask
    finite_mask = np.isfinite(raw_features).astype(np.uint8)

    # Replace NaN and infinity with zero
    features = np.nan_to_num(
        raw_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # 5. Create landmark mask
    # --------------------------------------------------------

    landmark_mask = create_landmark_mask(features)

    mask = landmark_mask & finite_mask

    # Reject recordings with too few frames / hands (sequence validation).
    problems = validate_sequence(features)

    if problems:
        raise ValueError("Sequence failed validation: " + "; ".join(problems))

    # --------------------------------------------------------
    # 6. Validate final shapes
    # --------------------------------------------------------

    if features.shape[1] != EXPECTED_FEATURES:

        raise ValueError(
            f"Invalid feature count: {features.shape[1]}"
        )

    if mask.shape != features.shape:

        raise ValueError(
            "Feature and mask shapes do not match."
        )

    # --------------------------------------------------------
    # 7. Create processed directory
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DIR /
        f"{csv_path.stem}_processed.npz"
    )

    # --------------------------------------------------------
    # 8. Save processed dataset
    # --------------------------------------------------------

    np.savez_compressed(
        output_path,

        features=features,
        mask=mask,

        frames=pd.to_numeric(
            df["frame"],
            errors="coerce"
        ).fillna(-1).to_numpy(dtype=np.int32),

        timestamps=pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=np.float64),

        feature_columns=np.asarray(
            expected_columns
        ),

        sample_id=str(metadata["sample_id"]),
        signer_id=str(metadata["signer_id"]),
        recording_type=str(metadata["recording_type"]),
        label=label,
        schema_name=SCHEMA_NAME,
        mirror_input=str(metadata.get("mirror_input", "unknown")),
    )

    # --------------------------------------------------------
    # 9. Report
    # --------------------------------------------------------

    print("\n========== PREPROCESSING SUCCESS ==========")

    print("Sample:", csv_path.name)
    print("Label:", label)
    print("Frames:", len(features))
    print("Feature shape:", features.shape)
    print("Mask shape:", mask.shape)
    print("Valid features:", f"{mask.mean():.2%}")
    print("Saved:", output_path)

    return output_path


# ============================================================
# PREPROCESS ALL SAMPLES
# ============================================================

def preprocess_all_samples():

    if not SAMPLES_DIR.exists():

        print("Samples directory does not exist:")
        print(SAMPLES_DIR)
        return

    csv_files = sorted(
        SAMPLES_DIR.glob("*.csv")
    )

    if not csv_files:

        print("No CSV recordings found.")
        return

    processed = 0
    skipped = 0

    print("\n========== PREPROCESSING STARTED ==========")
    print("Total CSV files:", len(csv_files))

    for csv_path in csv_files:

        try:

            preprocess_sample(csv_path)
            processed += 1

        except Exception as error:

            skipped += 1

            print("\n[SKIPPED]")
            print("File:", csv_path.name)
            print("Reason:", error)

    print("\n========== PREPROCESSING COMPLETE ==========")
    print("Successfully processed:", processed)
    print("Skipped:", skipped)
    print("Output directory:", PROCESSED_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_all_samples()
