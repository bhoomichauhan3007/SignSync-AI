
import cv2


class Camera:

    def __init__(self, camera_index=0, width=960, height=540):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """Open and configure the webcam."""

        self.cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():
            print("Error: Unable to open camera.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Camera opened successfully.")
        return True

    def read(self):
        """Read one frame from the webcam."""

        if self.cap is None or not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()

        if not success:
            return False, None

        return True, frame

    def release(self):
        """Release the webcam."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":

    camera = Camera()

    if camera.open():

        print("Press Q to close the camera.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            cv2.imshow("SignSync AI - Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
