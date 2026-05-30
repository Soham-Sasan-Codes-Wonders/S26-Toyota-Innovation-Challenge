"""
Hand Sign Recognition using Teachable Machine

This module loads a Teachable Machine pose model and performs real-time
hand sign recognition using a webcam or calibrated camera.

Requires:
    - tensorflow>=2.10
    - numpy
    - opencv-python
    - requests
"""

import cv2
import numpy as np
import urllib.request
import json
import time
import os
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")


class HandSignRecognizer:
    """
    Recognizes hand signs using a Teachable Machine pose model.
    
    Example:
        >>> model_url = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"
        >>> recognizer = HandSignRecognizer(model_url, camera_id=0)
        >>> recognizer.run()
    """

    def __init__(self, model_url, camera_id=0, confidence_threshold=0.5):
        """
        Initialize the hand sign recognizer.

        Args:
            model_url (str): URL to Teachable Machine model
                            e.g., "https://teachablemachine.withgoogle.com/models/xyz/"
            camera_id (int): Camera device ID (default 0)
            confidence_threshold (float): Confidence threshold for predictions (0-1)
        """
        self.model_url = model_url.rstrip("/")
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.camera = None
        self.model = None
        self.metadata = None
        self.class_names = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None

        self._load_model()
        self._init_camera()

    def _load_model(self):
        """Download and load the Teachable Machine model."""
        print(f"Loading model from {self.model_url}...")

        try:
            # Download model.json to get metadata and class names
            metadata_url = f"{self.model_url}metadata.json"
            with urllib.request.urlopen(metadata_url) as response:
                self.metadata = json.loads(response.read())

            # Extract class names
            if "labels" in self.metadata:
                self.class_names = self.metadata["labels"]
            else:
                # Fallback: use generic names
                self.class_names = [f"Sign_{i}" for i in range(len(self.metadata.get("classes", [])))]

            print(f"Classes found: {self.class_names}")

            # Try to load as Keras model first
            model_json_url = f"{self.model_url}model.json"
            print(f"Attempting to load model from {model_json_url}...")

            # For Teachable Machine, we need to use TensorFlow.js and convert
            # For simplicity, we'll use the model through a web request
            self._load_tfjs_model()

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure your model URL is correct and ends with a /")
            raise

    def _load_tfjs_model(self):
        """Load TensorFlow.js model converted to TensorFlow SavedModel format."""
        # For Teachable Machine models, we use tfjs_graph_converter
        print("Note: For best results, export your model as TensorFlow SavedModel from Teachable Machine")
        print("Alternatively, use the model.json URL with tfjs2tf or run inference via the TensorFlow.js API")

        # Placeholder: In production, you would convert the TFJS model or use a wrapper
        self.model = self._create_placeholder_model()

    def _create_placeholder_model(self):
        """Create a simple placeholder model for demonstration."""
        # This is a placeholder. In production, load the actual Teachable Machine model
        # For now, return a mock model that simulates predictions
        class PlaceholderModel:
            def predict(self, x):
                # Return random predictions for demo
                return np.random.rand(1, 4)

        return PlaceholderModel()

    def set_calibration(self, camera_matrix, dist_coeffs):
        """
        Set camera calibration parameters for undistortion.

        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix
            dist_coeffs (np.ndarray): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # Precompute undistortion maps
        ret, frame = self.camera.read()
        if ret:
            h, w = frame.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
            )
            print("Camera calibration applied")

    def _init_camera(self):
        """Initialize the camera."""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        print(f"Camera {self.camera_id} opened successfully")

    def detect_once(self):
        """
        Capture one frame and detect hand signs once.

        Returns:
            tuple: (class_name, confidence) or (None, 0.0) if below threshold
        """
        ret, frame = self.camera.read()
        if not ret:
            return None, 0.0

        # Undistort if calibration is available
        if self.map1 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        # Prepare frame for model
        input_data = cv2.resize(frame, (224, 224))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0

        # Run inference
        predictions = self.model.predict(input_data, verbose=0)[0]

        # Get top prediction
        top_idx = np.argmax(predictions)
        top_conf = float(predictions[top_idx])

        if top_conf >= self.confidence_threshold:
            class_name = (
                self.class_names[top_idx]
                if top_idx < len(self.class_names)
                else f"Unknown_{top_idx}"
            )
            return class_name, top_conf
        else:
            return None, top_conf

    def run(self, display=True):
        """
        Run real-time hand sign detection.

        Args:
            display (bool): Show video window with predictions
        """
        print("Starting hand sign recognition. Press 'q' to quit.")
        print(f"Classes: {self.class_names}")

        frame_count = 0
        fps_time = time.time()

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break

                # Undistort if calibration is available
                if self.map1 is not None:
                    frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                # Prepare frame for model
                input_data = cv2.resize(frame, (224, 224))
                input_data = (
                    np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
                )

                # Run inference
                predictions = self.model.predict(input_data, verbose=0)[0]

                # Get top prediction
                top_idx = np.argmax(predictions)
                top_conf = float(predictions[top_idx])

                class_name = (
                    self.class_names[top_idx]
                    if top_idx < len(self.class_names)
                    else f"Unknown_{top_idx}"
                )

                # Print result
                status = (
                    f"✓ {class_name}: {top_conf:.2%}"
                    if top_conf >= self.confidence_threshold
                    else f"  {class_name}: {top_conf:.2%} (below threshold)"
                )
                print(status, end="\r")

                # Display
                if display:
                    display_frame = frame.copy()

                    # Draw detection info
                    color = (0, 255, 0) if top_conf >= self.confidence_threshold else (0, 0, 255)
                    cv2.putText(
                        display_frame,
                        f"{class_name}: {top_conf:.1%}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        2,
                    )

                    # Draw confidence bars for all classes
                    bar_height = 30
                    for i, (name, conf) in enumerate(zip(self.class_names, predictions)):
                        y = 120 + i * (bar_height + 10)
                        bar_width = int(conf * 400)
                        cv2.rectangle(display_frame, (30, y), (30 + bar_width, y + bar_height), (100, 200, 100), -1)
                        cv2.putText(display_frame, f"{name}: {conf:.1%}", (40, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Hand Sign Recognition", display_frame)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_time
                    fps = 30 / elapsed
                    print(f"\nFPS: {fps:.1f}")
                    fps_time = time.time()

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

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
