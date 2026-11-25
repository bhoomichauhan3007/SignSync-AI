import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate(name, model, X, y, encoder, split_name):
    pred = model.predict(X)
    if np.asarray(pred).ndim > 1:
        pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y, pred)

    print(f"\n{name} {split_name} accuracy: {accuracy:.4f}")
    print(classification_report(
        y, pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_.astype(str),
        zero_division=0
    ))

    Path("outputs").mkdir(exist_ok=True)

    report = classification_report(
        y, pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_.astype(str),
        output_dict=True,
        zero_division=0
    )

    with open(f"outputs/{name}_{split_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    np.save(
        f"outputs/{name}_{split_name}_confusion_matrix.npy",
        confusion_matrix(y, pred, labels=np.arange(len(encoder.classes_)))
    )

    return accuracy


def train_cnn(X_train, y_train, X_val, y_val, n_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    X_train = X_train.astype("float32")[..., None]
    X_val = X_val.astype("float32")[..., None]

    model = models.Sequential([
        layers.Input(shape=X_train.shape[1:]),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/alphabet_1d_cnn_best.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    model = tf.keras.models.load_model("models/alphabet_1d_cnn_best.keras")
    model.save("models/alphabet_1d_cnn.keras")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", choices=["rf", "mlp", "cnn", "all"], default="all")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    data = np.load(args.file, allow_pickle=True)

    X = np.asarray(data["X"], dtype=np.float32).reshape(len(data["X"]), -1)
    raw_y = np.asarray(data["y"]).reshape(-1).astype(str)
    splits = np.char.lower(np.asarray(data["splits"]).astype(str))

    if not (len(X) == len(raw_y) == len(splits)):
        raise ValueError("X, y and splits have different sample counts.")

    train_mask = np.isin(splits, ["training", "train"])
    val_mask = np.isin(splits, ["validation", "valid", "val"])
    test_mask = np.isin(splits, ["testing", "test"])

    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError("Could not identify all three predefined splits.")

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train_raw = raw_y[train_mask]
    y_val_raw = raw_y[val_mask]
    y_test_raw = raw_y[test_mask]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)

    unseen = (set(y_val_raw) | set(y_test_raw)) - set(encoder.classes_)
    if unseen:
        raise ValueError(f"Classes missing from training split: {unseen}")

    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)

    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)
    print("Classes:", len(encoder.classes_))

    np.save("models/alphabet_classes.npy", encoder.classes_)

    selected = ["rf", "mlp", "cnn"] if args.model == "all" else [args.model]

    for name in selected:
        print(f"\n========== TRAINING {name.upper()} ==========")

        if name == "rf":
            model = RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            joblib.dump(
                {"model": model, "encoder": encoder},
                "models/alphabet_rf.joblib"
            )

        elif name == "mlp":
            model = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    early_stopping=True,
                    max_iter=300,
                    random_state=42
                )
            )
            model.fit(X_train, y_train)
            joblib.dump(
                {"model": model, "encoder": encoder},
                "models/alphabet_mlp.joblib"
            )

        else:
            model = train_cnn(
                X_train, y_train,
                X_val, y_val,
                len(encoder.classes_)
            )

        evaluate(name, model, X_val, y_val, encoder, "validation")
        evaluate(name, model, X_test, y_test, encoder, "test")

    print("\nAlphabet training and evaluation completed.")


if __name__ == "__main__":
    main()
