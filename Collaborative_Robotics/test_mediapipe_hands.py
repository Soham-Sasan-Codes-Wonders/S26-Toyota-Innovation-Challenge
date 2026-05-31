"""
Small standalone MediaPipe Hands test.

Usage:
    python Collaborative_Robotics/test_mediapipe_hands.py --camera 0

Press 'q' or ESC to exit.
"""

import argparse
import time
import sys
import cv2
import numpy as np
import os
import urllib.request
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

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    print("Failed to import MediaPipe:", e)
    print("Install with: python -m pip install mediapipe opencv-python")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands quick test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true", help="Run without showing the GUI window")
    args = parser.parse_args()

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    def download_model():
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    # Camera (use CAP_DSHOW on Windows for more reliable capture)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        sys.exit(1)

    model_path = download_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        prev_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # mirror for natural interaction
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)

                gesture = None
                confidence = 0.0

                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        
                        def xy(i):
                            lm = hand_landmarks[i]
                            return np.array([lm.x * w, lm.y * h])
                            
                        def get_dist(p1, p2):
                            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                            
                        def finger_extended(tip_idx, pip_idx):
                            return get_dist(xy(0), xy(tip_idx)) > get_dist(xy(0), xy(pip_idx))
                            
                        thumb_ext = get_dist(xy(17), xy(4)) > get_dist(xy(17), xy(3)) * 1.2
                        
                        index_ext = finger_extended(8, 6)
                        middle_ext = finger_extended(12, 10)
                        ring_ext = finger_extended(16, 14)
                        pinky_ext = finger_extended(20, 18)
                        
                        if thumb_ext and not any([index_ext, middle_ext, ring_ext, pinky_ext]):
                            gesture = "Thumbs Up"
                            confidence = 0.95
                        elif index_ext and middle_ext and ring_ext and pinky_ext:
                            gesture = "Open Hand"
                            confidence = 0.95

                        # draw landmarks
                        for connection in HAND_CONNECTIONS:
                            start_pt = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[0]].y * h))
                            end_pt = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                            cv2.line(frame, start_pt, end_pt, (0, 128, 255), 2)
                        for lm in hand_landmarks:
                            pt = (int(lm.x * w), int(lm.y * h))
                            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

                # annotate
                label = f"{gesture} ({int(confidence*100)}%)" if gesture else "No gesture"
                color = (0, 255, 0) if gesture else (0, 0, 255)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                # fps
                now = time.time()
                fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                prev_time = now
                cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                if not args.no_display:
                    cv2.imshow("MediaPipe Hands Test", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
