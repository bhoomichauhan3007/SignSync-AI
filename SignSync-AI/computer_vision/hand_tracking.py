
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# =====================================================
# CONFIGURATION
# =====================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

from feature_schema import FEATURES_PER_FRAME as FEATURE_COUNT
SEQUENCE_LENGTH = 30

DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

WINDOW_NAME = "SignSync AI - Live Computer Vision"

# =====================================================
# MEDIAPIPE
# =====================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =====================================================
# FEATURE EXTRACTION (1662 FEATURES)
# Single shared implementation: holistic_features.py
# Schema: Pose 132 + Face 1404 + Left hand 63 + Right hand 63
# =====================================================
from holistic_features import extract_frame_features as extract_features
from feature_schema import MIRROR_INPUT_FOR_HOLISTIC


# =====================================================
# DRAW LANDMARKS
# =====================================================
def draw_landmarks(frame, results):

    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    # Face contours
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 180, 80),
                thickness=1,
                circle_radius=1
            )
        )

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )


# =====================================================
# CAMERA
# =====================================================
def open_camera():

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    return cap


# =====================================================
# MAIN
# =====================================================
def main():

    print("=" * 45)
    print("SignSync AI - Live Computer Vision")
    print(f"Feature extraction: {FEATURE_COUNT} per frame")
    print(f"Sequence buffer: {SEQUENCE_LENGTH} frames")
    print("Press Q to exit")
    print("=" * 45)

    cap = open_camera()

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Feature sequence buffer
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # Rolling hand detection history
    detection_history = deque(maxlen=SEQUENCE_LENGTH)

    # FPS calculation timestamps
    fps_timestamps = deque(maxlen=60)

    total_frames = 0
    start_time = time.time()

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        ) as holistic:

            while True:

                frame_start = time.perf_counter()

                ret, frame = cap.read()

                if not ret:
                    print("ERROR: Failed to read camera frame.")
                    break

                # Same orientation as the recorder (see feature_schema.py), otherwise
                # Left/Right hand features would be swapped versus training data.
                if MIRROR_INPUT_FOR_HOLISTIC:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                results = holistic.process(rgb)

                rgb.flags.writeable = True

                # Extract features
                feature_vector = extract_features(results)
                sequence_buffer.append(feature_vector)

                # Hand count
                left_detected = results.left_hand_landmarks is not None
                right_detected = results.right_hand_landmarks is not None

                hands_detected = int(left_detected) + int(right_detected)

                detection_history.append(hands_detected > 0)

                # FPS
                total_frames += 1
                now = time.perf_counter()
                fps_timestamps.append(now)

                if len(fps_timestamps) >= 2:
                    elapsed_fps = (
                        fps_timestamps[-1] - fps_timestamps[0]
                    )
                    fps = (
                        (len(fps_timestamps) - 1) / elapsed_fps
                        if elapsed_fps > 0 else 0
                    )
                else:
                    fps = 0

                processing_ms = (
                    time.perf_counter() - frame_start
                ) * 1000

                # Detection stability over rolling 30 frames
                stability = (
                    sum(detection_history) / len(detection_history) * 100
                    if detection_history else 0
                )

                # Draw landmarks, then mirror ONLY the preview for natural viewing
                draw_landmarks(frame, results)
                frame = cv2.flip(frame, 1)

                # =================================================
                # DISPLAY OVERLAY
                # =================================================
                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (410, 190),
                    (25, 25, 25),
                    -1
                )

                frame = cv2.addWeighted(
                    overlay, 0.65, frame, 0.35, 0
                )

                hand_color = (
                    (0, 255, 0) if hands_detected > 0
                    else (0, 0, 255)
                )

                cv2.putText(
                    frame,
                    f"Hands Detected: {hands_detected}/2",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    hand_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Processing: {processing_ms:.1f} ms/frame",
                    (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Sequence: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                    (25, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Hand Stability: {stability:.1f}%",
                    (25, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(WINDOW_NAME, frame)

                # Terminal log every 30 frames
                if total_frames % 30 == 0:

                    print(
                        f"Frame: {total_frames} | "
                        f"Shape: {feature_vector.shape} | "
                        f"Hands: {hands_detected}/2 | "
                        f"FPS: {fps:.1f} | "
                        f"Processing: {processing_ms:.1f}ms | "
                        f"Stability: {stability:.1f}% | "
                        f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
                    )

                # Quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed_total = time.time() - start_time

        print("\nComputer Vision stopped.")
        print(f"Total frames processed: {total_frames}")
        print(f"Final buffer size: {len(sequence_buffer)}")
        print(f"Total runtime: {elapsed_total:.2f} seconds")
        print("Camera released safely.")


if __name__ == "__main__":
    main()
