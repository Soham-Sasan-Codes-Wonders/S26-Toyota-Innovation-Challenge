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

try:
    import mediapipe as mp
except Exception as e:
    print("Failed to import MediaPipe:", e)
    print("Install with: python -m pip install mediapipe opencv-python")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands quick test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6)
    parser.add_argument("--no-display", action="store_true", help="Run without showing the GUI window")
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Camera (use CAP_DSHOW on Windows for more reliable capture)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        sys.exit(1)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
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
                results = hands.process(rgb)

                gesture = None
                confidence = 0.0

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness or []):
                        # draw landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2),
                        )

                        # helper to get pixel coords
                        def xy(i):
                            lm = hand_landmarks.landmark[i]
                            return np.array([lm.x * w, lm.y * h])

                        def finger_extended(tip_idx, pip_idx):
                            tip = xy(tip_idx)
                            pip = xy(pip_idx)
                            return tip[1] < pip[1] - max(10, h * 0.02)

                        index_ext = finger_extended(8, 6)
                        middle_ext = finger_extended(12, 10)
                        ring_ext = finger_extended(16, 14)
                        pinky_ext = finger_extended(20, 18)

                        thumb_tip = xy(4)
                        thumb_ip = xy(3)
                        thumb_up = thumb_tip[1] < thumb_ip[1] - max(10, h * 0.02)

                        # Simple rules
                        if thumb_up and not any([index_ext, middle_ext, ring_ext, pinky_ext]):
                            gesture = "Thumbs Up"
                            confidence = 0.95
                        elif index_ext and middle_ext and ring_ext and pinky_ext:
                            gesture = "Open Hand"
                            confidence = 0.95

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
