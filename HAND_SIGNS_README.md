# Hand Sign Recognition with Teachable Machine

This guide shows how to set up hand sign recognition using Google's Teachable Machine and integrate it with the Dobot camera.

## Step 1: Train Your Hand Sign Model in Teachable Machine

1. Go to https://teachablemachine.withgoogle.com/
2. Click **"Get Started"**
3. Select **"Pose"** (for hand/body poses)
4. Create a new project and sign in with Google
5. Add classes for each hand sign you want to recognize:
   - Class 1: "Thumbs Up"
   - Class 2: "Peace Sign"
   - Class 3: "Open Hand"
   - Class 4: "Fist"
   - (Add more as needed)

6. For each class:
   - Click **"Hold"** to record samples
   - Position your hand in front of the camera
   - Collect 20-30 samples per class
   - Move your hand around to capture different angles and distances

7. Once trained, test in the preview on the right side

## Step 2: Export the Model

1. In Teachable Machine, click **"Export Model"** (top right)
2. Select **"TensorFlow.js"**
3. Check **"Upload (cloud)"** (easier) or **"Offline"** (works without internet)
4. Copy the model URL or download the files
6. There is no separate "model number" to enter. Use the exported URL provided by Teachable Machine.
7. If using cloud upload, you'll get a URL like:
   ```
   https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/
   ```

## Step 3: Install Required Libraries

```bash
pip install tensorflow
pip install numpy
pip install opencv-python
pip install requests  # if using cloud models
```

## Step 4: Use the Python Script

See `hand_sign_recognition.py` in this folder.

### Quick Start:

```python
from hand_sign_recognition import HandSignRecognizer
import lib.DobotDllType as dType
import dobotArm

# Your Teachable Machine model URL
model_url = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"

recognizer = HandSignRecognizer(model_url, camera_id=0)
recognizer.run()
```

### Using with Calibrated Camera:

```python
from hand_sign_recognition import HandSignRecognizer
import numpy as np
import cv2

# Load camera calibration from your existing calibration
data = np.load("camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

model_url = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"
recognizer = HandSignRecognizer(model_url, camera_id=0)
recognizer.set_calibration(camera_matrix, dist_coeffs)
recognizer.run()
```

## Step 5: Extend for Robot Control

Once you have detection working, you can map signs to robot actions:

```python
recognizer = HandSignRecognizer(model_url, camera_id=0)
api = dType.load()
dobotArm.initialize_robot(api)

while True:
    sign = recognizer.detect_once()
    
    if sign == "Thumbs Up":
        dobotArm.move_to_xyz(api, 200, 50, 100)
    elif sign == "Peace Sign":
        dobotArm.move_to_home(api)
    elif sign == "Open Hand":
        dobotArm.open_gripper(api)
    elif sign == "Fist":
        dobotArm.close_gripper(api)
```

## Troubleshooting

- **Model not loading**: Check your model URL is correct
- **Low confidence**: Add more training samples, especially edge cases
- **Slow inference**: Try on GPU if available, or reduce image resolution
- **Camera not detected**: Verify the camera_id (0 is usually the default)

## File Format

When exporting from Teachable Machine as TensorFlow.js, you get:
- `model.json` - Model architecture and weights
- `metadata.json` - Class names and other metadata
- Option to use cloud URL directly (recommended for first time)
