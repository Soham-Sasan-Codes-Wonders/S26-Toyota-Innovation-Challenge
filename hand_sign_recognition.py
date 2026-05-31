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
import os
import urllib.request
import sys
import ctypes

# Fix for MediaPipe's Tasks API on Python 3.13 (Windows)
if sys.platform == "win32":
    _original_getattr = ctypes.CDLL.__getattr__
    def _patched_getattr(self, name):
        if name == "free":
            try:
                return _original_getattr(self, name)
            except AttributeError:
                try:
                    return ctypes.cdll.ucrtbase.free
                except Exception:
                    pass
        return _original_getattr(self, name)
    ctypes.CDLL.__getattr__ = _patched_getattr

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    raise ImportError("MediaPipe is required. Install with: pip install mediapipe opencv-python")


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

        self.class_names = ["Thumbs Up / Down", "Open Hand"]
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        self._init_detector()

        if self.camera_id is not None:
            self._init_camera()
        else:
            print("Camera initialized externally (shared mode).")

    def _init_detector(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _init_camera(self):
        """Initialize the camera."""
        self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        print(f"Camera {self.camera_id} opened successfully")

    def _pixel_coordinates(self, landmark, image_width, image_height):
        return np.array([landmark.x * image_width, landmark.y * image_height])

    def _is_finger_extended(self, tip, pip, wrist):
        return np.hypot(tip[0]-wrist[0], tip[1]-wrist[1]) > np.hypot(pip[0]-wrist[0], pip[1]-wrist[1])

    def _is_thumb_extended(self, thumb_tip, thumb_ip, pinky_base):
        return np.hypot(thumb_tip[0]-pinky_base[0], thumb_tip[1]-pinky_base[1]) > np.hypot(thumb_ip[0]-pinky_base[0], thumb_ip[1]-pinky_base[1]) * 1.2

    def _classify_gesture(self, hand_landmarks, image_shape, handedness_label=None):
        image_height, image_width = image_shape[:2]
        landmarks = hand_landmarks
        
        wrist = self._pixel_coordinates(landmarks[0], image_width, image_height)
        pinky_base = self._pixel_coordinates(landmarks[17], image_width, image_height)

        index_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[8], image_width, image_height),
            self._pixel_coordinates(landmarks[6], image_width, image_height),
            wrist,
        )
        middle_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[12], image_width, image_height),
            self._pixel_coordinates(landmarks[10], image_width, image_height),
            wrist,
        )
        ring_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[16], image_width, image_height),
            self._pixel_coordinates(landmarks[14], image_width, image_height),
            wrist,
        )
        pinky_extended = self._is_finger_extended(
            self._pixel_coordinates(landmarks[20], image_width, image_height),
            self._pixel_coordinates(landmarks[18], image_width, image_height),
            wrist,
        )

        thumb_ext = self._is_thumb_extended(
            self._pixel_coordinates(landmarks[4], image_width, image_height),
            self._pixel_coordinates(landmarks[3], image_width, image_height),
            pinky_base,
        )

        if thumb_ext and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return "Thumbs Up / Down", 0.95

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

    def process_frame(self, frame):
        """Process a single frame and return results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        gesture_name = None
        confidence = 0.0

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            handedness_label = None
            if results.handedness:
                handedness_label = results.handedness[0][0].category_name

            gesture_name, confidence = self._classify_gesture(
                hand_landmarks, frame.shape, handedness_label
            )
            for connection in self.hand_connections:
                start_pt = (int(hand_landmarks[connection[0]].x * frame.shape[1]), int(hand_landmarks[connection[0]].y * frame.shape[0]))
                end_pt = (int(hand_landmarks[connection[1]].x * frame.shape[1]), int(hand_landmarks[connection[1]].y * frame.shape[0]))
                cv2.line(frame, start_pt, end_pt, (0, 128, 255), 2)
            for lm in hand_landmarks:
                pt = (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        return gesture_name, confidence, frame

    def detect_once(self):
        """
        Capture one frame and detect hand signs once.

        Returns:
            tuple: (class_name, confidence, frame)
        """
        if self.camera is None:
            return None, 0.0, None
            
        ret, frame = self.camera.read()
        if not ret:
            return None, 0.0, None

        if self.map1 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        return self.process_frame(frame)
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
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
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
