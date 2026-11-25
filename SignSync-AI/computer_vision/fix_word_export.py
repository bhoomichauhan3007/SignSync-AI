
from pathlib import Path
import numpy as np
import shutil
from datetime import datetime
import re

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw" / "words" / "MP_Data"
OUT = ROOT / "data" / "exports" / "word_dataset.npz"

EXPECTED_CLASSES = 41
FRAMES = 30
FEATURES = 1662


def natural_sort_key(path):
    parts = re.split(r"(\d+)", str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


if not RAW.exists():
    raise FileNotFoundError(f"Dataset folder not found: {RAW}")

# Read class folders
class_dirs = sorted(
    [p for p in RAW.iterdir() if p.is_dir()],
    key=lambda p: p.name.lower()
)

classes = [p.name for p in class_dirs]

if len(classes) != EXPECTED_CLASSES:
    raise RuntimeError(
        f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}"
    )

# Preserve previous class order if an existing dataset is available
if OUT.exists():
    try:
        with np.load(OUT, allow_pickle=False) as old:
            if "classes" in old.files:
                old_classes = [str(c) for c in old["classes"]]

                if set(old_classes).issubset(set(classes)):
                    classes = old_classes + [
                        c for c in classes if c not in old_classes
                    ]

        print("Previous class order preserved.")

    except Exception as error:
        print("Could not read previous class order:", error)

X_list = []
y_list = []
report = []

# Process each class
for label, class_name in enumerate(classes):

    class_dir = RAW / class_name
    seq_dirs = []

    # Find folders containing exactly 30 frame files
    for folder in class_dir.rglob("*"):

        if not folder.is_dir():
            continue

        npy_files = [
            f for f in folder.glob("*.npy")
            if f.is_file()
        ]

        if len(npy_files) == FRAMES:
            seq_dirs.append(folder)

    seq_dirs = sorted(
        set(seq_dirs),
        key=lambda p: natural_sort_key(p.relative_to(class_dir))
    )

    count = 0

    for seq_dir in seq_dirs:

        files = sorted(
            [
                f for f in seq_dir.glob("*.npy")
                if f.is_file()
            ],
            key=lambda p: natural_sort_key(p.stem)
        )

        if len(files) != FRAMES:
            continue

        frames = []

        for file in files:

            arr = np.load(file, allow_pickle=False)
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)

            if arr.shape[0] != FEATURES:
                raise ValueError(
                    f"Wrong feature length: {file}\n"
                    f"Expected: {FEATURES}, Got: {arr.shape[0]}"
                )

            if not np.isfinite(arr).all():
                raise ValueError(f"NaN or Inf detected: {file}")

            frames.append(arr)

        sequence = np.stack(frames)

        if sequence.shape != (FRAMES, FEATURES):
            raise ValueError(
                f"Invalid sequence shape: {seq_dir} -> {sequence.shape}"
            )

        X_list.append(sequence)
        y_list.append(label)
        count += 1

    report.append((class_name, count))
    print(f"{class_name}: {count} sequences")


# Stop if no valid data exists
if not X_list:
    raise RuntimeError("No valid sequences found. Export cancelled.")

# Build dataset
X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.int32)

# Final validation
if X.ndim != 3:
    raise RuntimeError(f"Invalid X dimensions: {X.shape}")

if X.shape[1:] != (FRAMES, FEATURES):
    raise RuntimeError(f"Unexpected X shape: {X.shape}")

if len(classes) != EXPECTED_CLASSES:
    raise RuntimeError(f"Unexpected class count: {len(classes)}")

if not np.isfinite(X).all():
    raise RuntimeError("X contains NaN or Inf")

if len(y) != len(X):
    raise RuntimeError("X and y sample counts do not match")

# Classes with no valid sequences are reported, not fatal. Labels keep their position in
# `classes`, so the saved label indices stay aligned with Member 2's word models.
present = set(y.tolist())
empty_classes = [classes[i] for i in range(len(classes)) if i not in present]
if empty_classes:
    print("[WARNING] Classes with no valid sequences (kept in label order):", empty_classes)

# Print summary
print("\n========== DATASET SUMMARY ==========")
print("Total sequences:", len(X))
print("Total classes:", len(classes))
print("X shape:", X.shape)
print("y shape:", y.shape)

# Backup previous dataset
OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():

    backup = OUT.with_name(
        f"word_dataset_backup_{datetime.now():%Y%m%d_%H%M%S}.npz"
    )

    shutil.copy2(OUT, backup)
    print("\nBackup created:", backup)

# Save safely using temporary file
TEMP_OUT = OUT.with_name("word_dataset_temp.npz")

np.savez_compressed(
    TEMP_OUT,
    X=X,
    y=y,
    classes=np.array(classes, dtype=str)
)

# Replace only after successful validation and saving
TEMP_OUT.replace(OUT)

print("\n========== EXPORT SUCCESS ==========")
print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("Classes:", len(classes))
print("Saved:", OUT)
