import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

from feature_schema import FEATURES_PER_FRAME

# ================ CONFIG =================
DATASET_ROOT = Path(r"C:\Users\Trishika\Downloads\Dataset")
OUTPUT_DIR = Path(__file__).parent / "extracted_holistic_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CLASSES = list("ABCDEFGHIJKLMNOPQR")
VAL_CLASSES = list("STUVWXYZ")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


# ================ FEATURE EXTRACTION =================
def extract_features(results, fallback_hand=None):
    """
    1662-feature vector (schema: feature_schema.py).

    Holistic needs to see a body to find hands, so close-up hand images (the
    alphabet dataset) often get NO hands from it. `fallback_hand` is an optional
    (label, landmarks) from the dedicated MediaPipe Hands model, used only for
    the slot Holistic left empty.
    """

    pose = np.zeros((33, 4), dtype=np.float32)
    face = np.zeros((468, 3), dtype=np.float32)
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )

    if results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark[:468]],
            dtype=np.float32
        )

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )

    if fallback_hand is not None:
        label, landmarks = fallback_hand
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                       dtype=np.float32)
        if label == "Left" and not left_hand.any():
            left_hand = arr
        elif label == "Right" and not right_hand.any():
            right_hand = arr

    features = np.concatenate([
        pose.flatten(),
        face.flatten(),
        left_hand.flatten(),
        right_hand.flatten()
    ])

    if features.shape != (FEATURES_PER_FRAME,):
        raise ValueError(f"Expected {FEATURES_PER_FRAME} features, got {features.shape}")

    return features


# ================ PROCESS SPLIT =================
def process_split(split_name, class_names, holistic, hands):

    split_dir = DATASET_ROOT / split_name

    X, y = [], []
    failed_images = []
    no_hand_count = 0
    fallback_count = 0
    class_counts = {}

    print(f"\nProcessing {split_name}...")

    for class_name in class_names:

        class_dir = split_dir / class_name

        if not class_dir.exists():
            print(f"Missing folder: {class_name}")
            continue

        images = sorted([
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
            ])

        class_counts[class_name] = 0
        print(f"{class_name}: {len(images)} images")

        for image_path in images:

            image = cv2.imread(str(image_path))

            if image is None:
                failed_images.append(str(image_path))
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            fallback_hand = None

            if (results.left_hand_landmarks is None and
                    results.right_hand_landmarks is None):

                hands_result = hands.process(rgb)

                if (hands_result.multi_hand_landmarks and
                        hands_result.multi_handedness):
                    fallback_hand = (
                        hands_result.multi_handedness[0].classification[0].label,
                        hands_result.multi_hand_landmarks[0]
                    )
                    fallback_count += 1
                else:
                    no_hand_count += 1

            features = extract_features(results, fallback_hand)

            X.append(features)
            y.append(class_name)
            class_counts[class_name] += 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    output_name = f"{split_name.lower()}_alphabet_holistic_features.npz"

    np.savez_compressed(
        OUTPUT_DIR / output_name,
        X=X,
        y=y,
        classes=np.asarray(class_names)
    )

    # ================ REPORT =================
    report_path = OUTPUT_DIR / f"{split_name.lower()}_report.txt"

    with open(report_path, "w") as f:

        f.write(f"Split: {split_name}\n")
        f.write(f"Total images processed: {len(X)}\n")
        f.write(f"Feature shape: {X.shape}\n")
        f.write(f"Feature count per image: {FEATURES_PER_FRAME}\n")
        f.write(f"Hands found only by Hands-model fallback: {fallback_count}\n")
        f.write(f"No hand detected (after fallback): {no_hand_count}\n")
        f.write(f"Failed images: {len(failed_images)}\n\n")

        f.write("Class counts:\n")

        for cls, count in class_counts.items():
            f.write(f"{cls}: {count}\n")

        if failed_images:
            f.write("\nFailed images:\n")
            for path in failed_images:
                f.write(path + "\n")

    print(f"\n{split_name} completed!")
    print(f"X shape: {X.shape}")
    print(f"Hand via fallback: {fallback_count}")
    print(f"No hand detected (after fallback): {no_hand_count}")
    print(f"Saved: {OUTPUT_DIR / output_name}")

    return X, y


# ================ MAIN =================
def main():

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5
    ) as holistic, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands:

        process_split("Training", TRAIN_CLASSES, holistic, hands)
        process_split("Validation", VAL_CLASSES, holistic, hands)

    print("\nALL HOLISTIC EXTRACTION COMPLETED!")
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
