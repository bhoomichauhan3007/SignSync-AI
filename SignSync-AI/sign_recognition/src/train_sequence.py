
import argparse
import json
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from common import load_npz, normalize_labels, make_output_dirs


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def build_model(kind, frames, features, num_classes):

    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(frames, features))

    if kind == "lstm":

        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(inp)

        x = layers.Dropout(0.3)(x)

        x = layers.Bidirectional(
            layers.LSTM(64)
        )(x)

    elif kind == "gru":

        x = layers.Bidirectional(
            layers.GRU(128, return_sequences=True)
        )(inp)

        x = layers.Dropout(0.3)(x)

        x = layers.Bidirectional(
            layers.GRU(64)
        )(x)

    else:

        x = layers.Dense(128)(inp)

        positions = tf.range(
            start=0,
            limit=frames,
            delta=1
        )

        positional_encoding = layers.Embedding(
            input_dim=frames,
            output_dim=128
        )(positions)

        x = x + positional_encoding

        for _ in range(2):

            attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32,
                dropout=0.1
            )(x, x)

            x = layers.LayerNormalization()(x + attention)

            ff = layers.Dense(
                256,
                activation="relu"
            )(x)

            ff = layers.Dense(128)(ff)

            x = layers.LayerNormalization()(x + ff)

        x = layers.GlobalAveragePooling1D()(x)

    output = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inp,
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SignSync AI - Word Sequence Recognition Training"
    )

    parser.add_argument("--file", required=True)
    parser.add_argument("--x_key", default="X")
    parser.add_argument("--y_key", default="y")

    parser.add_argument(
        "--model",
        choices=["lstm", "gru", "transformer"],
        default="lstm"
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    # --------------------------------------------------------
    # TensorFlow check
    # --------------------------------------------------------

    try:
        import tensorflow as tf

    except ImportError:

        raise SystemExit(
            "TensorFlow is missing. Install using: "
            "python -m pip install tensorflow"
        )

    print("\n========================================")
    print("       SIGNSYNC AI SEQUENCE TRAINING")
    print("========================================")

    make_output_dirs()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    X, y, _ = load_npz(
        args.file,
        args.x_key,
        args.y_key
    )

    y = normalize_labels(y)

    if X.ndim != 3:

        raise SystemExit(
            f"Expected X shape (samples, frames, features), got {X.shape}"
        )

    if len(X) != len(y):

        raise SystemExit(
            f"Sample mismatch: X={len(X)}, y={len(y)}"
        )

    # --------------------------------------------------------
    # Memory-efficient preprocessing
    # --------------------------------------------------------

    print("Preprocessing features...")

    # Convert only when required
    X = np.asarray(X, dtype=np.float32)

    # Modify the existing array instead of creating another copy
    np.nan_to_num(
        X,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Load original class names
    # --------------------------------------------------------

    dataset = np.load(
        args.file,
        allow_pickle=True
    )

    if "classes" not in dataset.files:

        raise SystemExit(
            "The NPZ file does not contain 'classes'."
        )

    original_classes = np.asarray(
        dataset["classes"]
    ).astype(str)

    # --------------------------------------------------------
    # Handle missing classes
    # --------------------------------------------------------

    active_labels = np.unique(
        y.astype(int)
    )

    if (
        np.any(active_labels < 0)
        or np.any(active_labels >= len(original_classes))
    ):

        raise SystemExit(
            "Labels do not match indexes in the classes array."
        )

    class_names = original_classes[active_labels]

    label_mapping = {
        int(old_label): int(new_label)
        for new_label, old_label in enumerate(active_labels)
    }

    y = np.array(
        [label_mapping[int(label)] for label in y],
        dtype=np.int32
    )

    num_classes = len(class_names)

    missing_classes = [
        name
        for index, name in enumerate(original_classes)
        if index not in active_labels
    ]

    print("\n========== DATASET INFORMATION ==========")

    print("Total samples:", len(X))
    print("Frames per sample:", X.shape[1])
    print("Features per frame:", X.shape[2])
    print("Active classes:", num_classes)
    print("Class names:", class_names.tolist())

    if missing_classes:

        print("Classes without samples:", missing_classes)

    print("==========================================")

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    indices = np.arange(len(X))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Use views where possible instead of copying the full dataset
    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding", args.model.upper(), "model...")

    model = build_model(
        args.model,
        X.shape[1],
        X.shape[2],
        num_classes
    )

    model_path = (
        Path("models") /
        f"word_{args.model}.keras"
    )

    # --------------------------------------------------------
    # Training callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True
        )

    ]

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\n========== TRAINING STARTED ==========\n")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    print("\nEvaluating model...")

    predictions = model.predict(
        X_test,
        batch_size=args.batch_size,
        verbose=0
    ).argmax(axis=1)

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0
    )

    print("\n========== MODEL PERFORMANCE ==========")

    print("Model:", args.model.upper())

    print(
        "Test Accuracy:",
        round(test_accuracy, 4)
    )

    print("\nClassification Report:\n")

    print(report)

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    classes_path = (
        Path("models") /
        f"word_{args.model}_classes.npy"
    )

    np.save(
        classes_path,
        class_names
    )

    # Save original label indexes for decoding if needed
    label_indices_path = (
        Path("models") /
        f"word_{args.model}_label_indices.npy"
    )

    np.save(
        label_indices_path,
        active_labels
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {

        "model": args.model,

        "accuracy": float(test_accuracy),

        "total_samples": int(len(X)),

        "training_samples": int(len(X_train)),

        "testing_samples": int(len(X_test)),

        "frames": int(X.shape[1]),

        "features": int(X.shape[2]),

        "num_classes": int(num_classes),

        "class_names": class_names.tolist(),

        "missing_classes": missing_classes,

        "classification_report": classification_report(
            y_test,
            predictions,
            labels=np.arange(num_classes),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

    }

    metrics_path = (
        Path("reports") /
        f"word_{args.model}_metrics.json"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n========== SAVED FILES ==========")

    print("Model:", model_path)

    print("Classes:", classes_path)

    print("Label indexes:", label_indices_path)

    print("Metrics:", metrics_path)

    print("\nSequence model training completed successfully!")

    print("========================================")


if __name__ == "__main__":

    main()
