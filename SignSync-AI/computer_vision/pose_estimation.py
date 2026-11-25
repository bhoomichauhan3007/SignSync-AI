
import cv2
import mediapipe as mp


class PoseEstimator:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        """Detect body pose landmarks from a camera frame."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        return results.pose_landmarks

    def draw(self, frame, landmarks):
        """Draw detected body landmarks on the frame."""

        if landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame

    def close(self):
        self.pose.close()


def main():

    estimator = PoseEstimator()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        estimator.close()
        return

    print("Pose Estimation Started")
    print("Press Q to exit")

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame not received.")
            break

        frame = cv2.flip(frame, 1)

        landmarks = estimator.process(frame)

        if landmarks:
            frame = estimator.draw(frame, landmarks)
            cv2.putText(
                frame,
                "BODY DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "BODY NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("SignSync AI - Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    estimator.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
