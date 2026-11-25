import cv2
import mediapipe as mp
import csv
import json
import time
import uuid
import re
from pathlib import Path
from datetime import datetime

from feature_schema import (
    FEATURES_PER_FRAME,
    FEATURE_ORDER,
    SCHEMA_NAME,
    MIRROR_INPUT_FOR_HOLISTIC,
)
from holistic_features import extract_frame_features

# ============================================================
# SIGNSYNC AI - COMPUTER VISION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 15
MAX_DURATION = 8

MIN_HAND_FRAMES = 5
MIN_HAND_RATIO = 0.60

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

WINDOW = "SignSync AI - Holistic Recorder"

# 1662 FEATURES
FEATURE_COUNT = FEATURES_PER_FRAME
COLUMNS = ["frame", "timestamp"] + [
    f"feature_{i}" for i in range(FEATURE_COUNT)
]


# ============================================================
# SIGN INFORMATION
# ============================================================

def get_info():
    signer = input("Signer ID [test01]: ").strip() or "test01"

    label = input("Word label: ").strip().lower()

    while not label:
        label = input("Word label cannot be empty: ").strip().lower()

    return signer, label


# ============================================================
# HOLISTIC FEATURE EXTRACTION
# Single shared implementation (holistic_features.py), schema in
# feature_schema.py: POSE 132 + FACE 1404 + LEFT HAND 63 + RIGHT HAND 63
# ============================================================

def extract_features(results):
    return extract_frame_features(results).tolist()


# ============================================================
# SAVE WORD SAMPLE
# ============================================================

def save_sample(rows, signer, label, sample_id, hand_frames, pose_frames=0):

    if not rows:
        print("[NOT SAVED] No frames recorded.")
        return False

    total = len(rows)
    ratio = hand_frames / total

    if hand_frames < MIN_HAND_FRAMES:
        print("[NOT SAVED] Too few hand-detected frames.")
        return False

    if ratio < MIN_HAND_RATIO:
        print("[NOT SAVED] Hand detection ratio too low.")
        return False

    safe_signer = re.sub(r"[^A-Za-z0-9_-]", "_", signer)
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)

    timestamp = datetime.now()
    filename = (
        f"{safe_signer}_{safe_label}_{sample_id}_"
        f"{timestamp:%Y%m%d_%H%M%S}"
    )

    csv_path = OUTPUT_DIR / f"{filename}.csv"
    json_path = OUTPUT_DIR / f"{filename}.json"

    metadata = {
        "sample_id": sample_id,
        "signer_id": signer,
        "recording_type": "word",
        "label": label,
        "total_frames": total,
        "record_fps_target": FPS,
        "valid_hand_frames": hand_frames,
        "valid_hand_ratio": round(ratio, 4),
        "valid_pose_frames": pose_frames,
        "hand_valid_frames": hand_frames,
        "hand_valid_ratio": round(ratio, 4),
        "feature_count": FEATURE_COUNT,
        "feature_order": FEATURE_ORDER,
        "schema_name": SCHEMA_NAME,
        "mirror_input": MIRROR_INPUT_FOR_HOLISTIC,
        "csv_file": csv_path.name,
        "created_at": timestamp.isoformat(),
        "coordinate_note": "Holistic normalized landmark features"
    }

    csv_tmp = csv_path.with_suffix(".csv.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")

    try:
        with csv_tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()

            for row in rows:
                writer.writerow(dict(zip(COLUMNS, row)))

        with json_tmp.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        csv_tmp.replace(csv_path)
        json_tmp.replace(json_path)

    except Exception as e:
        for tmp in (csv_tmp, json_tmp):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        print("[SAVE ERROR]", e)
        return False

    print("\n========== SAMPLE SAVED ==========")
    print("Label:", label)
    print("Frames:", total)
    print("Hand detection:", f"{ratio:.0%}")
    print("Features/frame:", FEATURE_COUNT)
    print("CSV:", csv_path)
    print("JSON:", json_path)

    return True


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):

        camera = cv2.VideoCapture(CAMERA_INDEX, backend)

        if camera.isOpened():

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            success, frame = camera.read()

            if success and frame is not None:
                print("[CAMERA] Connected")
                return camera

        camera.release()

    return None


# ============================================================
# WORD RECORDER - HOLISTIC
# ============================================================

def run_recorder():

    signer, label = get_info()
    camera = open_camera()

    if camera is None:
        print("[ERROR] Camera could not be opened.")
        return

    recording = False
    rows = []
    hand_frames = 0
    pose_frames = 0
    frame_number = 0

    start_time = 0
    last_capture = 0
    sample_id = None

    print("\n========== CONTROLS ==========")
    print("R = Start recording")
    print("S = Stop and save")
    print("N = Change label")
    print("Q / ESC = Quit")
    print("Output:", OUTPUT_DIR)

    try:

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while True:

                success, frame = camera.read()

                if not success:
                    print("[ERROR] Camera frame failed.")
                    break

                frame_number += 1

                # Orientation convention shared with live inference.
                if MIRROR_INPUT_FOR_HOLISTIC:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                results = holistic.process(rgb)

                rgb.flags.writeable = True

                hand_detected = (
                    results.left_hand_landmarks is not None
                    or results.right_hand_landmarks is not None
                )

                # DRAW POSE
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS
                    )

                # DRAW FACE
                if results.face_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS
                    )

                # DRAW LEFT HAND
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS
                    )

                # DRAW RIGHT HAND
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS
                    )

                now = time.perf_counter()
                elapsed = now - start_time if recording else 0

                # RECORD AT 15 FPS
                if recording and now - last_capture >= 1 / FPS:

                    features = extract_features(results)

                    rows.append([
                        frame_number,
                        round(elapsed, 4)
                    ] + features)

                    if hand_detected:
                        hand_frames += 1
                    if results.pose_landmarks:
                        pose_frames += 1

                    last_capture = now

                # DISPLAY
                display = cv2.flip(frame, 1)

                cv2.rectangle(
                    display, (0, 0), (640, 90), (30, 30, 30), -1
                )

                cv2.putText(
                    display,
                    "SIGNSYNC AI - HOLISTIC RECORDER",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 180),
                    2
                )

                cv2.putText(
                    display,
                    f"Label: {label} | Hand: {'YES' if hand_detected else 'NO'}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

                if recording:
                    status = f"RECORDING {elapsed:.1f}s | FRAMES {len(rows)}"
                    color = (0, 0, 255)
                else:
                    status = "READY - PRESS R"
                    color = (0, 255, 180)

                cv2.putText(
                    display,
                    status,
                    (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

                cv2.imshow(WINDOW, display)

                key = cv2.waitKey(1) & 0xFF

                # QUIT
                if key in (ord("q"), 27):

                    if recording:
                        save_sample(
                            rows, signer, label, sample_id, hand_frames, pose_frames
                        )

                    break

                # START
                elif key == ord("r") and not recording:

                    rows = []
                    hand_frames = 0
                    pose_frames = 0
                    frame_number = 0
                    sample_id = uuid.uuid4().hex[:8]

                    start_time = time.perf_counter()
                    last_capture = 0
                    recording = True

                    print(f"[RECORDING] {label}")

                # STOP AND SAVE
                elif key == ord("s") and recording:

                    recording = False

                    save_sample(
                        rows, signer, label, sample_id, hand_frames, pose_frames
                    )

                    rows = []

                # CHANGE LABEL
                elif key == ord("n") and not recording:

                    cv2.destroyWindow(WINDOW)

                    signer, label = get_info()

                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

                # AUTO STOP AT 8 SECONDS
                if recording and elapsed >= MAX_DURATION:

                    recording = False

                    save_sample(
                        rows, signer, label, sample_id, hand_frames, pose_frames
                    )

                    rows = []

    except KeyboardInterrupt:
        print("\n[STOP] Recording interrupted.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nRecorder closed.")


# ============================================================
# ALPHABET IMAGE EXTRACTION - PRESERVED
# ============================================================

def extract_alphabet_dataset():

    dataset_dir = ROOT / "data" / "raw" / "alphabets"
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "alphabet_landmarks.csv"

    if not dataset_dir.exists():
        print("[ERROR] Alphabet folder not found:", dataset_dir)
        return

    if output_file.exists():
        answer = input(
            f"{output_file.name} exists. Overwrite? (y/N): "
        ).strip().lower()

        if answer != "y":
            print("Existing file kept.")
            return

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    feature_columns = [
        f"hand_{point}_{axis}"
        for point in range(21)
        for axis in ("x", "y", "z")
    ]

    output_columns = [
        "split", "label", "image_path",
        "hand_detected", "handedness", "confidence"
    ] + feature_columns

    total = 0
    detected = 0
    unreadable = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands:

        with output_file.open(
            "w", newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file, fieldnames=output_columns
            )
            writer.writeheader()

            for split in ("Training", "Testing", "Validation"):

                split_dir = dataset_dir / split

                if not split_dir.exists():
                    print("[INFO] Split not found:", split)
                    continue

                print(f"\nProcessing {split}...")

                for class_dir in sorted(split_dir.iterdir()):

                    if not class_dir.is_dir():
                        continue

                    label = class_dir.name.upper()

                    for image_path in sorted(class_dir.iterdir()):

                        if image_path.suffix.lower() not in extensions:
                            continue

                        total += 1

                        row = {
                            "split": split,
                            "label": label,
                            "image_path": str(image_path.relative_to(ROOT)),
                            "hand_detected": 0,
                            "handedness": "",
                            "confidence": 0.0
                        }

                        image = cv2.imread(str(image_path))

                        if image is None:
                            unreadable += 1
                            writer.writerow(row)
                            continue

                        rgb = cv2.cvtColor(
                            image, cv2.COLOR_BGR2RGB
                        )

                        results = hands.process(rgb)

                        if results.multi_hand_landmarks:

                            landmarks = results.multi_hand_landmarks[0]
                            row["hand_detected"] = 1
                            detected += 1

                            if results.multi_handedness:
                                classification = (
                                    results.multi_handedness[0]
                                    .classification[0]
                                )

                                row["handedness"] = classification.label
                                row["confidence"] = float(
                                    classification.score
                                )

                            for index, point in enumerate(
                                landmarks.landmark
                            ):
                                row[f"hand_{index}_x"] = float(point.x)
                                row[f"hand_{index}_y"] = float(point.y)
                                row[f"hand_{index}_z"] = float(point.z)

                        writer.writerow(row)

                        if total % 500 == 0:
                            print(
                                f"Processed {total} | "
                                f"detected {detected} | "
                                f"unreadable {unreadable}"
                            )

    print("\n========== ALPHABET EXTRACTION COMPLETE ==========")
    print("Images:", total)
    print("Hands detected:", detected)
    print("Images without detected hand:", total - detected - unreadable)
    print("Unreadable images:", unreadable)
    print("Output:", output_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========== SIGNSYNC AI - COMPUTER VISION ==========")
    print("1. Alphabet image extraction")
    print("2. Holistic word recorder")

    choice = input("Choose (1/2): ").strip()

    if choice == "1":
        extract_alphabet_dataset()

    elif choice == "2":
        run_recorder()

    else:
        print("Invalid choice.")