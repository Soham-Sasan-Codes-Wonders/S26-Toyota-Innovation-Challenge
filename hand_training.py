"""
Hand Training for Dobot Control

This script implements hand gesture recognition using MediaPipe Hands.
It detects two gestures:
- Thumbs Up gesture = Continue/Execute
- Open Hand gesture = Stop/Pause

The script continuously monitors hand gestures from a webcam and triggers
appropriate actions based on recognized signs.

Usage:
    1. Install MediaPipe, OpenCV, and NumPy
    2. Run: python hand_training.py
"""

import cv2
import numpy as np
from hand_sign_recognition import HandSignRecognizer
import time
from collections import deque
import sys
import os

# Add Collaborative_Robotics to path for robot control
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Collaborative_Robotics'))

try:
    import lib.DobotDllType as dType
    import dobotArm
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("⚠ Warning: Dobot libraries not found. Robot control disabled.")


class HandTrainingController:
    """
    Controller for hand gesture-based training and control.
    
    Recognizes Thumbs Up (Continue) and Flat Hand (Stop) gestures.
    Can optionally control a Dobot robot.
    """

    def __init__(self, model_url, camera_id=0, confidence_threshold=0.7, use_robot=False):
        """
        Initialize the hand training controller.

        Args:
            model_url (str): URL to Teachable Machine model with Thumbs Up and Flat Hand classes
            camera_id (int): Camera device ID (default 0)
            confidence_threshold (float): Confidence threshold for predictions (0-1)
            use_robot (bool): Whether to enable robot control (default False)
        """
        self.recognizer = HandSignRecognizer(
            model_url, 
            camera_id=camera_id, 
            confidence_threshold=confidence_threshold
        )
        self.running = True
        self.training_active = False
        self.gesture_history = deque(maxlen=5)  # Track last 5 gestures for stability
        self.last_action_time = 0
        self.action_cooldown = 0.5  # Prevent rapid repeated actions (seconds)
        
        # Robot control
        self.use_robot = use_robot and ROBOT_AVAILABLE
        self.api = None
        self.robot_initialized = False
        self.robot_paused = False
        
        if self.use_robot:
            try:
                self._init_robot()
            except Exception as e:
                print(f"⚠ Failed to initialize robot: {e}")
                self.use_robot = False

    def _is_thumbs_up(self, gesture_name):
        """Check if gesture matches Thumbs Up."""
        return gesture_name and "thumbs" in gesture_name.lower() and "up" in gesture_name.lower()

    def _is_flat_hand(self, gesture_name):
        """Check if gesture matches Flat Hand/Stop."""
        gesture_name_lower = gesture_name.lower() if gesture_name else ""
        return any(x in gesture_name_lower for x in ["flat", "open", "stop", "hand"])

    def _init_robot(self):
        """Initialize connection to Dobot robot."""
        if not ROBOT_AVAILABLE:
            return
        
        print("\n🤖 Initializing Dobot robot...")
        self.api = dType.load()
        dobotArm.initialize_robot(self.api)
        self.robot_initialized = True
        self.robot_paused = False
        print("✓ Dobot robot initialized and ready")

    def _pause_robot(self):
        """Pause the robot's motion."""
        if not self.use_robot or not self.robot_initialized or self.robot_paused:
            return
        
        try:
            print("\n⏸ Pausing robot motion...")
            dType.SetQueuedCmdStopExec(self.api)
            self.robot_paused = True
            print("✓ Robot paused")
        except Exception as e:
            print(f"✗ Error pausing robot: {e}")

    def _resume_robot(self):
        """Resume the robot's motion."""
        if not self.use_robot or not self.robot_initialized or not self.robot_paused:
            return
        
        try:
            print("\n▶ Resuming robot motion...")
            dType.SetQueuedCmdStartExec(self.api)
            self.robot_paused = False
            print("✓ Robot resumed")
        except Exception as e:
            print(f"✗ Error resuming robot: {e}")

    def _get_stable_gesture(self):
        """
        Get the most stable gesture from recent history.
        Requires 3 of last 5 frames to match the same gesture.
        """
        if len(self.gesture_history) < 3:
            return None

        # Count occurrences of each gesture
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        # Return gesture if it appears 3+ times in recent history
        for gesture, count in gesture_counts.items():
            if count >= 3:
                return gesture

        return None

    def _handle_continue(self):
        """Handle Thumbs Up (Continue) action."""
        if time.time() - self.last_action_time > self.action_cooldown:
            print("\n✓ CONTINUE: Thumbs Up detected - Starting/Resuming training...")
            self.training_active = True
            if self.use_robot:
                self._resume_robot()
            self.last_action_time = time.time()
            return True
        return False

    def _handle_stop(self):
        """Handle Flat Hand (Stop) action."""
        if time.time() - self.last_action_time > self.action_cooldown:
            print("\n✗ STOP: Flat Hand detected - Pausing training...")
            self.training_active = False
            if self.use_robot:
                self._pause_robot()
            self.last_action_time = time.time()
            return True
        return False

    def run(self, display=True):
        """
        Run the hand training controller.

        Args:
            display (bool): Show video window with gesture recognition
        """
        print("\n" + "=" * 70)
        print("HAND TRAINING CONTROLLER")
        print("=" * 70)
        print("\nGestures:")
        print("  • Thumbs Up     → Continue/Start training")
        print("  • Open Hand     → Stop/Pause training")
        if self.use_robot:
            print("\n🤖 Robot Control ENABLED")
            print("  • Thumbs Up     → Resume robot motion")
            print("  • Open Hand     → Pause robot motion")
        else:
            print("\n🤖 Robot Control DISABLED")
        print("\nPress 'q' to quit\n")
        print(f"Recognized gestures: {self.recognizer.class_names}")
        print("-" * 70)

        frame_count = 0
        fps_time = time.time()

        try:
            while self.running:
                class_name, top_conf, display_frame = self.recognizer.detect_once()
                if display_frame is None:
                    break

                if top_conf >= self.recognizer.confidence_threshold:
                    self.gesture_history.append(class_name)
                else:
                    self.gesture_history.append(None)

                stable_gesture = self._get_stable_gesture()

                if stable_gesture:
                    if self._is_thumbs_up(stable_gesture):
                        self._handle_continue()
                    elif self._is_flat_hand(stable_gesture):
                        self._handle_stop()

                status_line = f"Training: {'🟢 ACTIVE' if self.training_active else '🔴 PAUSED'}"
                if self.use_robot:
                    status_line += f" | Robot: {'🟢 RUNNING' if not self.robot_paused else '🔴 PAUSED'}"
                gesture_line = (
                    f"Gesture: {stable_gesture} (✓ confirmed)"
                    if stable_gesture
                    else f"Gesture: {class_name or 'None'}: {top_conf:.1%}"
                )
                print(f"{status_line} | {gesture_line}", end="\r")

                if display:
                    h, w = display_frame.shape[:2]
                    text_area_height = 150 if not self.use_robot else 190
                    cv2.rectangle(display_frame, (10, 10), (w - 10, text_area_height), (0, 0, 0), -1)
                    cv2.rectangle(display_frame, (10, 10), (w - 10, text_area_height), (200, 200, 200), 2)

                    status_color = (0, 255, 0) if self.training_active else (0, 0, 255)
                    status_text = "TRAINING: ACTIVE" if self.training_active else "TRAINING: PAUSED"
                    cv2.putText(
                        display_frame,
                        status_text,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        status_color,
                        2,
                    )

                    gesture_color = (0, 255, 0) if top_conf >= self.recognizer.confidence_threshold else (0, 0, 255)
                    gesture_text = f"{class_name or 'No gesture'}: {top_conf:.1%}"
                    cv2.putText(
                        display_frame,
                        gesture_text,
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        gesture_color,
                        2,
                    )

                    if self.use_robot:
                        robot_color = (0, 255, 0) if not self.robot_paused else (0, 0, 255)
                        robot_text = "ROBOT: RUNNING" if not self.robot_paused else "ROBOT: PAUSED"
                        cv2.putText(
                            display_frame,
                            robot_text,
                            (30, 135),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            robot_color,
                            2,
                        )

                    cv2.putText(
                        display_frame,
                        "Thumbs Up = Continue | Open Hand = Stop | Q = Quit",
                        (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow("Hand Training Controller", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuitting...")
                    self.running = False

                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_time
                    fps = 30 / elapsed
                    fps_time = time.time()

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.close()

    def close(self):
        """Close camera and cleanup."""
        # Stop robot if it's running
        if self.use_robot and self.robot_initialized:
            try:
                print("Stopping robot...")
                dType.SetQueuedCmdStopExec(self.api)
            except:
                pass
        
        if self.recognizer.camera:
            self.recognizer.camera.release()
        cv2.destroyAllWindows()
        print("Closed.")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    # TODO: Replace with your trained Teachable Machine model URL
    # Instructions:
    # 1. Go to https://teachablemachine.withgoogle.com/
    # 2. Create a Pose project
    # 3. Train two classes:
    #    - "Thumbs Up"
    #    - "Flat Hand"
    # 4. Export as TensorFlow.js (Cloud upload)
    # 5. Copy the URL here
    MODEL_URL = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"

    # Configuration
    CAMERA_ID = 0
    CONFIDENCE_THRESHOLD = 0.7
    USE_CALIBRATION = True
    USE_ROBOT = True  # Set to False to disable robot control

    # Load camera calibration if available
    camera_matrix = None
    dist_coeffs = None

    if USE_CALIBRATION:
        try:
            data = np.load("camera_params.npz")
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
            print("✓ Camera calibration loaded")
        except FileNotFoundError:
            print("⚠ camera_params.npz not found. Using uncalibrated camera.")

    # Create and run controller
    controller = HandTrainingController(
        MODEL_URL, 
        camera_id=CAMERA_ID, 
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_robot=USE_ROBOT
    )

    # Apply calibration if available
    if camera_matrix is not None:
        controller.recognizer.set_calibration(camera_matrix, dist_coeffs)

    # Run the controller
    controller.run(display=True)
