
import csv
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

# ============================================================
# SIGNSYNC AI - COMPUTER VISION / DATASET RECORDER
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
RECORD_FPS = 15
MAX_WORD_DURATION = 8

MIN_VALID_HAND_FRAMES = 5
MIN_VALID_HAND_RATIO = 0.60

HAND_CONFIDENCE = 0.60
HAND_TRACKING = 0.60
POSE_CONFIDENCE = 0.55
POSE_TRACKING = 0.55
POSE_VISIBILITY_THRESHOLD = 0.50

WINDOW_NAME = "SignSync AI - Word Recorder"

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Dataset columns: 2 hands * 21 * 3 + 33 pose * 4
COLUMNS = ["frame", "timestamp"]

for hand_name in ("Left", "Right"):
    for point_index in range(21):
        for axis in ("x", "y", "z"):
            COLUMNS.append(
                f"{hand_name}_hand_{point_index}_{axis}"
            )

for point_index in range(33):
    for axis in ("x", "y", "z", "visibility"):
        COLUMNS.append(f"pose_{point_index}_{axis}")


# ============================================================
# SIGN INFORMATION
# ============================================================

def get_sample_information():
    print("\n========== SIGNSYNC WORD DATASET ==========")

    signer_id = input("Enter signer ID [test01]: ").strip()
    if not signer_id:
        signer_id = "test01"

    label = input("Enter exact word/gloss label: ").strip().lower()

    while not label:
        print("Label cannot be empty.")
        label = input("Enter exact word/gloss label: ").strip().lower()

    return signer_id, "word", label


# ============================================================
# HAND LANDMARK EXTRACTION
# ============================================================

def extract_hand_data(results, frame_width=None, frame_height=None):

    hand_data = {
        "Left": None,
        "Right": None
    }

    hand_confidence = {
        "Left": 0.0,
        "Right": 0.0
    }

    if not results.multi_hand_landmarks:
        return hand_data, hand_confidence

    if not results.multi_handedness:
        return hand_data, hand_confidence

    for landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):

        if not handedness.classification:
            continue

        classification = handedness.classification[0]
        confidence = float(classification.score)

        if confidence < HAND_CONFIDENCE:
            continue

        # MediaPipe assumes mirrored selfie input.
        # Our camera processing uses the unmirrored frame.
        anatomical_label = (
            "Right" if classification.label == "Left"
            else "Left"
        )

        if hand_data[anatomical_label] is None:
            hand_data[anatomical_label] = landmarks
            hand_confidence[anatomical_label] = confidence

    return hand_data, hand_confidence


# ============================================================
# POSE LANDMARK EXTRACTION
# ============================================================

def extract_pose_data(results):

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    important_points = (11, 12, 13, 14, 15, 16, 23, 24)

    visible_count = sum(
        landmarks[index].visibility >= POSE_VISIBILITY_THRESHOLD
        for index in important_points
    )

    return landmarks if visible_count >= 5 else None


# ============================================================
# BUILD ONE FRAME
# ============================================================

def build_frame_row(
    frame_number,
    timestamp,
    hand_data,
    pose_landmarks
):

    row = {column: "" for column in COLUMNS}

    row["frame"] = frame_number
    row["timestamp"] = round(float(timestamp), 4)

    for hand_name, landmarks in hand_data.items():

        if landmarks is None:
            continue

        for index, point in enumerate(landmarks.landmark):

            row[f"{hand_name}_hand_{index}_x"] = float(point.x)
            row[f"{hand_name}_hand_{index}_y"] = float(point.y)
            row[f"{hand_name}_hand_{index}_z"] = float(point.z)

    if pose_landmarks is not None:

        for index, point in enumerate(pose_landmarks):

            if point.visibility < POSE_VISIBILITY_THRESHOLD:
                continue

            row[f"pose_{index}_x"] = float(point.x)
            row[f"pose_{index}_y"] = float(point.y)
            row[f"pose_{index}_z"] = float(point.z)
            row[f"pose_{index}_visibility"] = float(point.visibility)

    return row


# ============================================================
# SAVE RECORDING
# ============================================================

def save_sample(
    rows,
    signer_id,
    sample_type,
    label,
    sample_id,
    hand_valid_frames,
    pose_valid_frames
):

    if not rows:
        print("[NOT SAVED] No frames were recorded.")
        return False

    total_frames = len(rows)
    valid_ratio = hand_valid_frames / total_frames

    if hand_valid_frames < MIN_VALID_HAND_FRAMES:
        print(
            f"[NOT SAVED] Only {hand_valid_frames} hand frames detected. "
            f"At least {MIN_VALID_HAND_FRAMES} required."
        )
        return False

    if valid_ratio < MIN_VALID_HAND_RATIO:
        print(
            f"[NOT SAVED] Hand detected in {valid_ratio:.0%} of frames. "
            f"At least {MIN_VALID_HAND_RATIO:.0%} required."
        )
        return False

    safe_signer = re.sub(
        r"[^A-Za-z0-9_-]+", "_", signer_id
    ).strip("_") or "signer"

    safe_label = re.sub(
        r"[^A-Za-z0-9_-]+", "_", label
    ).strip("_") or "word"

    created = datetime.now()

    filename = (
        f"{safe_signer}_{safe_label}_{sample_id}_"
        f"{created:%Y%m%d_%H%M%S}"
    )

    csv_path = OUTPUT_DIR / f"{filename}.csv"
    json_path = OUTPUT_DIR / f"{filename}.json"

    csv_temp = OUTPUT_DIR / f"{filename}.csv.tmp"
    json_temp = OUTPUT_DIR / f"{filename}.json.tmp"

    metadata = {
        "sample_id": sample_id,
        "signer_id": signer_id,
        "recording_type": sample_type,
        "label": label,
        "total_frames": total_frames,
        "record_fps_target": RECORD_FPS,
        "valid_hand_frames": hand_valid_frames,
        "valid_hand_ratio": round(valid_ratio, 4),
        "valid_pose_frames": pose_valid_frames,
        "csv_file": csv_path.name,
        "created_at": created.isoformat(),
        "coordinate_note": (
            "Normalized MediaPipe x/y/z landmarks; "
            "pose visibility included."
        )
    }

    try:

        with csv_temp.open(
            "w", newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=COLUMNS
            )

            writer.writeheader()
            writer.writerows(rows)

        with json_temp.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        csv_temp.replace(csv_path)
        json_temp.replace(json_path)

    except Exception:

        for path in (csv_temp, json_temp):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        raise

    print("\n========== SAMPLE SAVED SUCCESSFULLY ==========")
    print("Label:", label)
    print("Frames:", total_frames)
    print("Hand-valid frames:", hand_valid_frames)
    print("Pose-valid frames:", pose_valid_frames)
    print("CSV:", csv_path)
    print("JSON:", json_path)

    return True


# ============================================================
# ALPHABET IMAGE DATASET EXTRACTION
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
            f"{output_file.name} already exists. "
            "Overwrite it? (y/N): "
        ).strip().lower()

        if answer != "y":
            print("Kept existing alphabet landmarks file.")
            return

    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    feature_columns = [
        f"hand_{point}_{axis}"
        for point in range(21)
        for axis in ("x", "y", "z")
    ]

    output_columns = [
        "split",
        "label",
        "image_path",
        "hand_detected",
        "handedness",
        "confidence"
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
                file,
                fieldnames=output_columns
            )

            writer.writeheader()

            for split in ("Training", "Testing", "Validation"):

                split_dir = dataset_dir / split

                if not split_dir.exists():
                    print("[INFO] Split not found:", split)
                    continue

                print(f"\nProcessing {split}...", flush=True)

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
                            "image_path": str(
                                image_path.relative_to(ROOT)
                            ),
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
                            image,
                            cv2.COLOR_BGR2RGB
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
                                f"unreadable {unreadable}",
                                flush=True
                            )

    print("\n========== ALPHABET EXTRACTION COMPLETE ==========")
    print("Images:", total)
    print("Hands detected:", detected)
    print("Images without detected hand:", total - detected - unreadable)
    print("Unreadable images:", unreadable)
    print("Output:", output_file)


# ============================================================
# OPEN CAMERA
# ============================================================

def open_camera():

    backends = (
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
        cv2.CAP_ANY
    )

    for backend in backends:

        camera = cv2.VideoCapture(CAMERA_INDEX, backend)

        if camera.isOpened():

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT
            )

            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            success, frame = camera.read()

            if success and frame is not None:

                print(
                    f"[CAMERA] Opened successfully. Backend: {backend}"
                )

                return camera

        camera.release()

    return None


# ============================================================
# KEYBOARD INPUT
# ============================================================

def read_control_key():

    import sys

    key = None

    # Read keys from Windows terminal.
    if sys.platform == "win32":

        import msvcrt

        while msvcrt.kbhit():

            char = msvcrt.getwch()

            if char not in ("\x00", "\xe0"):
                key = char.lower()

    # Also accept keys from the OpenCV window.
    window_key = cv2.waitKey(1)

    if window_key != -1:

        value = window_key & 0xFF

        if value in (
            27,
            ord("r"),
            ord("s"),
            ord("n"),
            ord("q")
        ):

            key = (
                chr(value).lower()
                if value != 27
                else "esc"
            )

    return key


# ============================================================
# WEBCAM WORD RECORDER
# ============================================================

def run_webcam_recorder():

    signer_id, sample_type, label = get_sample_information()

    camera = open_camera()

    if camera is None:

        print("[ERROR] Could not open webcam.")
        print("Check camera permission or close other camera apps.")
        return

    recording = False
    rows = []
    sample_id = None

    frame_number = 0
    hand_valid_frames = 0
    pose_valid_frames = 0

    record_start = 0.0
    last_capture = 0.0
    last_status = 0.0

    exit_reason = "Unknown"

    print("\n========== RECORDER CONTROLS ==========")
    print("R = Start recording")
    print("S = Stop and save")
    print("N = Change word label")
    print("Q / ESC = Exit")
    print("Ctrl+C in terminal = Exit")
    print("Keys work in terminal or camera window.")
    print("Output folder:", OUTPUT_DIR, flush=True)

    try:

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=HAND_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING
        ) as hands, mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=POSE_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING
        ) as pose:

            while True:

                success, frame = camera.read()

                if not success or frame is None:

                    exit_reason = "Camera frame read failed"
                    print("[ERROR] Camera stopped returning frames.")
                    break

                frame_number += 1

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                rgb.flags.writeable = False

                hand_results = hands.process(rgb)
                pose_results = pose.process(rgb)

                rgb.flags.writeable = True

                hand_data, _ = extract_hand_data(
                    hand_results,
                    frame.shape[1],
                    frame.shape[0]
                )

                pose_landmarks = extract_pose_data(pose_results)

                now = time.perf_counter()

                hand_count = sum(
                    value is not None
                    for value in hand_data.values()
                )

                # Draw hand landmarks.
                for landmarks in hand_data.values():

                    if landmarks is not None:

                        mp_draw.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                # Draw pose landmarks.
                if pose_landmarks is not None:

                    mp_draw.draw_landmarks(
                        frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                elapsed = (
                    now - record_start
                    if recording
                    else 0.0
                )

                # Record frames at target FPS.
                if (
                    recording
                    and now - last_capture >= 1.0 / RECORD_FPS
                ):

                    rows.append(
                        build_frame_row(
                            frame_number,
                            elapsed,
                            hand_data,
                            pose_landmarks
                        )
                    )

                    if hand_count > 0:
                        hand_valid_frames += 1

                    if pose_landmarks is not None:
                        pose_valid_frames += 1

                    last_capture = now

                # Camera display.
                display = cv2.flip(frame, 1)

                height, width = display.shape[:2]

                cv2.rectangle(
                    display,
                    (0, 0),
                    (width, 85),
                    (25, 25, 25),
                    -1
                )

                cv2.putText(
                    display,
                    "SIGNSYNC AI - WORD RECORDER",
                    (15, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (0, 255, 180),
                    2
                )

                cv2.putText(
                    display,
                    f"LABEL: {label}   HANDS: {hand_count}/2",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

                if recording:

                    status = (
                        f"RECORDING {elapsed:.1f}s | "
                        f"FRAMES {len(rows)}"
                    )

                    color = (0, 0, 255)

                else:

                    status = "READY - PRESS R TO RECORD"
                    color = (0, 255, 180)

                cv2.putText(
                    display,
                    status,
                    (15, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display
                )

                key = read_control_key()

                # EXIT.
                if key in ("q", "esc"):

                    if recording:

                        save_sample(
                            rows,
                            signer_id,
                            sample_type,
                            label,
                            sample_id,
                            hand_valid_frames,
                            pose_valid_frames
                        )

                    exit_reason = "Exit key pressed"
                    break

                # START RECORDING.
                elif key == "r" and not recording:

                    rows = []

                    frame_number = 0
                    hand_valid_frames = 0
                    pose_valid_frames = 0

                    sample_id = uuid.uuid4().hex[:8]

                    record_start = time.perf_counter()
                    last_capture = 0.0

                    recording = True

                    print(
                        f"\n[RECORDING STARTED] "
                        f"label={label} | max={MAX_WORD_DURATION}s",
                        flush=True
                    )

                # STOP AND SAVE.
                elif key == "s" and recording:

                    recording = False

                    print("\n[STOP] Saving sample...", flush=True)

                    save_sample(
                        rows,
                        signer_id,
                        sample_type,
                        label,
                        sample_id,
                        hand_valid_frames,
                        pose_valid_frames
                    )

                    rows = []

                # CHANGE LABEL.
                elif key == "n" and not recording:

                    print(
                        "\n[CHANGE LABEL] Enter details in terminal.",
                        flush=True
                    )

                    cv2.destroyWindow(WINDOW_NAME)

                    signer_id, sample_type, label = (
                        get_sample_information()
                    )

                    cv2.namedWindow(
                        WINDOW_NAME,
                        cv2.WINDOW_NORMAL
                    )

                # AUTO SAVE AT MAXIMUM DURATION.
                if (
                    recording
                    and elapsed >= MAX_WORD_DURATION
                ):

                    recording = False

                    print(
                        "\n[INFO] Maximum duration reached. Saving...",
                        flush=True
                    )

                    save_sample(
                        rows,
                        signer_id,
                        sample_type,
                        label,
                        sample_id,
                        hand_valid_frames,
                        pose_valid_frames
                    )

                    rows = []

                # Terminal status every 3 seconds.
                if now - last_status >= 3.0:

                    print(
                        f"[LIVE] Camera=OK | "
                        f"Hands={hand_count} | "
                        f"Recording={recording} | "
                        f"Frames={len(rows)}",
                        flush=True
                    )

                    last_status = now

                # Detect manual closing of camera window.
                try:

                    if cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE
                    ) < 1:

                        if recording:

                            save_sample(
                                rows,
                                signer_id,
                                sample_type,
                                label,
                                sample_id,
                                hand_valid_frames,
                                pose_valid_frames
                            )

                        exit_reason = "Preview window closed"
                        break

                except cv2.error:

                    exit_reason = "Preview window closed"
                    break

    except KeyboardInterrupt:

        exit_reason = "Ctrl+C received"

        print("\n[STOP] Ctrl+C received.", flush=True)

        if recording:

            save_sample(
                rows,
                signer_id,
                sample_type,
                label,
                sample_id,
                hand_valid_frames,
                pose_valid_frames
            )

    except Exception as exc:

        exit_reason = f"Error: {type(exc).__name__}"

        print(
            f"\n[RECORDER ERROR] {type(exc).__name__}: {exc}",
            flush=True
        )

    finally:

        camera.release()
        cv2.destroyAllWindows()

        try:
            cv2.waitKey(1)
        except cv2.error:
            pass

        print("\n========== RECORDER STOPPED ==========")
        print("Reason:", exit_reason)
        print("Camera released.")
        print("Samples folder:", OUTPUT_DIR, flush=True)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n========== SIGNSYNC AI - COMPUTER VISION ==========")
    print("1. Extract landmarks from downloaded alphabet images")
    print("2. Start webcam word recorder (1662-feature holistic schema)")
    print("3. [LEGACY] Hands+Pose recorder (258 features, NOT ML compatible)")
    print("Q. Exit")

    choice = input("Choose option (1/2/3/Q): ").strip().lower()

    if choice == "1":

        extract_alphabet_dataset()

    elif choice == "2":

        # The ML dataset uses the 1662-feature holistic schema, so the
        # standard recorder is the holistic one (see feature_schema.py).
        from landmark_extraction_holistic import run_recorder

        run_recorder()

    elif choice == "3":

        print("\nWARNING: this recorder writes 258 features/frame.")
        print("preprocessing.py will REJECT these files (expects 1662).")

        if input("Continue anyway? (y/N): ").strip().lower() == "y":
            run_webcam_recorder()

    elif choice in ("q", "quit", "exit"):

        print("Exited.")

    else:

        print("Invalid option. Run the program again.")


if __name__ == "__main__":
    main()
