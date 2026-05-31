"""
Hand Sign Recognition using MediaPipe Hands

This module performs real-time hand gesture recognition using MediaPipe Hands.
It recognizes two robot-safe control gestures:
    - Thumbs Up: resume / continue
    - Open Hand: stop / pause

Requires:
    - mediapipe
    - numpy
    - opencv-python
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe is required. Install with: pip install mediapipe opencv-python")
import importlib

# Try to import the 'solutions' submodule in a few ways to be compatible
# with different mediapipe package layouts / deprecations.
mp_solutions = None
try:
    # preferred: top-level import
    from mediapipe import solutions as mp_solutions
except Exception:
    try:
        # fallback: mediapipe.python.solutions
        mp_solutions = importlib.import_module("mediapipe.python.solutions")
    except Exception:
        mp_solutions = None

if mp_solutions is None:
    # final attempt: if mediapipe exposes 'solutions' as attribute on mp
    mp_solutions = getattr(mp, 'solutions', None)

if mp_solutions is None:
    raise ImportError("Could not import MediaPipe 'solutions' module. Ensure mediapipe is installed and up-to-date.")


class HandSignRecognizer:
    """
    Recognizes simple hand control gestures using MediaPipe Hands.

    Example:
        >>> recognizer = HandSignRecognizer(camera_id=0)
        >>> recognizer.run()
    """

    def __init__(self, model_url=None, camera_id=0, confidence_threshold=0.5):
        """
        Initialize the hand sign recognizer.

        Args:
            model_url (str): Optional compatibility parameter. Ignored when using MediaPipe.
            camera_id (int): Camera device ID (default 0)
            confidence_threshold (float): Minimum confidence for gesture output
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.camera = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None

        self.class_names = ["Thumbs Up", "Open Hand"]
        self.mp_hands = mp_solutions.hands
        self.mp_drawing = mp_solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self._init_camera()

    def _init_camera(self):
        """Initialize the camera."""
        self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        print(f"Camera {self.camera_id} opened successfully")

    def _pixel_coordinates(self, landmark, image_width, image_height):
        return np.array([landmark.x * image_width, landmark.y * image_height])

    def _is_finger_extended(self, tip, pip, image_height):
        return tip[1] < pip[1] - max(10, image_height * 0.02)

    def _is_thumb_up(self, landmarks, image_width, image_height):
        thumb_tip = self._pixel_coordinates(landmarks[4], image_width, image_height)
        thumb_ip = self._pixel_coordinates(landmarks[3], image_width, image_height)
        return thumb_tip[1] < thumb_ip[1] - max(10, image_height * 0.02)

    def _classify_gesture(self, hand_landmarks, image_shape, handedness_label=None):
        image_height, image_width = image_shape[:2]
        landmarks = hand_landmarks.landmark

        index_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[8], image_width, image_height),
            self._pixel_coordinates(landmarks[6], image_width, image_height),
            image_height,
        )
        middle_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[12], image_width, image_height),
            self._pixel_coordinates(landmarks[10], image_width, image_height),
            image_height,
        )
        ring_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[16], image_width, image_height),
            self._pixel_coordinates(landmarks[14], image_width, image_height),
            image_height,
        )
        pinky_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[20], image_width, image_height),
            self._pixel_coordinates(landmarks[18], image_width, image_height),
            image_height,
        )

        thumb_up = self._is_thumb_up(landmarks, image_width, image_height)

        if thumb_up and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return "Thumbs Up", 0.95

        if all([index_extended, middle_extended, ring_extended, pinky_extended]):
            return "Open Hand", 0.95

        return None, 0.0

    def set_calibration(self, camera_matrix, dist_coeffs):
        """
        Set camera calibration parameters for undistortion.

        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix
            dist_coeffs (np.ndarray): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        ret, frame = self.camera.read()
        if ret:
            h, w = frame.shape[:2]
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
            )
            print("Camera calibration applied")

    def detect_once(self):
        """
        Capture one frame and detect hand signs once.

        Returns:
            tuple: (class_name, confidence, frame)
        """
        ret, frame = self.camera.read()
        if not ret:
            return None, 0.0, None

        if self.map1 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture_name = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = None
            if results.multi_handedness:
                handedness_label = results.multi_handedness[0].classification[0].label

            gesture_name, confidence = self._classify_gesture(
                hand_landmarks, frame.shape, handedness_label
            )
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2),
            )

        return gesture_name, confidence, frame

    def run(self, display=True):
        """
        Run real-time hand sign detection.

        Args:
            display (bool): Show video window with predictions
        """
        print("Starting hand sign recognition. Press 'q' to quit.")
        print(f"Classes: {self.class_names}")

        try:
            while True:
                gesture_name, confidence, frame = self.detect_once()
                if frame is None:
                    break

                status = (
                    f"✓ {gesture_name}: {confidence:.1%}"
                    if gesture_name and confidence >= self.confidence_threshold
                    else "No valid gesture detected"
                )
                print(status, end="\r")

                if display:
                    display_frame = frame.copy()
                    label = gesture_name if gesture_name else "Waiting for gesture..."
                    color = (0, 255, 0) if gesture_name else (0, 0, 255)
                    cv2.putText(
                        display_frame,
                        label,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        2,
                    )
                    if gesture_name:
                        cv2.putText(
                            display_frame,
                            f"Confidence: {confidence:.1%}",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2,
                        )
                    cv2.imshow("Hand Sign Recognition", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    
    def close(self):
        """Release camera and close windows."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("\nClosed.")


if __name__ == "__main__":
    # Example usage
    print("Hand Sign Recognition Module")
    print("=" * 50)

    model_url = input(
        "Enter your Teachable Machine model URL\n"
        "(e.g., https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/):\n> "
    ).strip()

    if not model_url:
        print("No URL provided. Exiting.")
        exit(1)

    try:
        recognizer = HandSignRecognizer(model_url, camera_id=0)
        recognizer.run(display=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
