"""
Example: Hand Sign Recognition with Calibrated Camera

This script demonstrates how to use hand sign recognition 
with the Dobot's calibrated camera.

Usage:
    1. First, train and export a Teachable Machine pose model
    2. Replace MODEL_URL with your model's URL
    3. Run this script
"""

import numpy as np
import cv2
from hand_sign_recognition import HandSignRecognizer
import lib.DobotDllType as dType
import dobotArm


# ========================================
# CONFIGURATION
# ========================================

# Your Teachable Machine model URL
# Replace with your actual model URL from Teachable Machine
MODEL_URL = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"

# Confidence threshold (0-1)
CONFIDENCE_THRESHOLD = 0.7

# Use the calibrated camera
USE_CALIBRATION = True


# ========================================
# MAIN
# ========================================

def main():
    # Load camera calibration if available
    camera_matrix = None
    dist_coeffs = None

    if USE_CALIBRATION:
        try:
            data = np.load("camera_params.npz")
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
            print("✓ Camera calibration loaded from camera_params.npz")
        except FileNotFoundError:
            print("⚠ camera_params.npz not found. Using uncalibrated camera.")

    # Initialize hand sign recognizer
    print("\nInitializing hand sign recognizer...")
    recognizer = HandSignRecognizer(
        MODEL_URL, 
        camera_id=0, 
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # Apply calibration if available
    if camera_matrix is not None and dist_coeffs is not None:
        recognizer.set_calibration(camera_matrix, dist_coeffs)

    # Run detection
    print("\n" + "=" * 60)
    print("Hand Sign Recognition")
    print("=" * 60)
    print("Press 'q' to quit\n")

    try:
        recognizer.run(display=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recognizer.close()


# ========================================
# EXAMPLE: Using with Robot Actions
# ========================================

def example_with_robot():
    """
    Example showing how to map hand signs to robot actions.
    
    Uncomment and modify as needed for your use case.
    """
    # Initialize recognizer
    recognizer = HandSignRecognizer(MODEL_URL, camera_id=0)

    # Initialize robot (only if you want robot control)
    # api = dType.load()
    # dobotArm.initialize_robot(api)

    print("\nHand Sign -> Robot Action Mapping")
    print("=" * 60)

    gesture_to_action = {
        # Map your hand sign classes to actions
        "Thumbs Up": lambda: print("  ➜ Action: Thumbs Up detected"),
        "Peace Sign": lambda: print("  ➜ Action: Peace Sign detected"),
        "Open Hand": lambda: print("  ➜ Action: Open Hand detected"),
        "Fist": lambda: print("  ➜ Action: Fist detected"),
    }

    print(f"Monitoring for: {list(gesture_to_action.keys())}\n")

    frame_count = 0
    while frame_count < 300:  # Run for ~10 seconds at 30 FPS
        sign, confidence = recognizer.detect_once()

        if sign and sign in gesture_to_action:
            print(f"\n[{frame_count}] Recognized: {sign} ({confidence:.1%})")
            gesture_to_action[sign]()
            # Example: Uncomment to control robot
            # if sign == "Thumbs Up":
            #     dobotArm.move_to_xyz(api, 200, 50, 100)

        frame_count += 1

    recognizer.close()
    print("\nDone.")


if __name__ == "__main__":
    # Run basic example
    main()

    # Uncomment to run with robot actions:
    # example_with_robot()
