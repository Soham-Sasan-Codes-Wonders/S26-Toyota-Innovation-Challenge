# Hand Training with Thumbs Up & Flat Hand Gestures

This script implements interactive hand gesture recognition for training and control with **optional Dobot robot integration**:
- **Thumbs Up** 👍 = Continue/Start Training (Resume robot motion)
- **Flat Hand** ✋ = Stop/Pause Training (Pause robot motion)

## 🚀 Quick Setup

### Step 1: Train Your Model in Teachable Machine

1. Go to [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/)
2. Click **"Get Started"**
3. Select **"Pose"** project type (important!)
4. Sign in with Google
5. Create a new project

### Step 2: Add Two Classes

Create exactly these two classes:

#### Class 1: "Thumbs Up"
- Make a thumbs up gesture with your hand
- Click **"Hold"** to record 20-30 samples
- Move your hand around at different angles and distances
- Include variations: different arms, positions, speeds

#### Class 2: "Flat Hand"
- Extend your hand flat/open (like a stop gesture)
- Click **"Hold"** to record 20-30 samples
- Vary the angles, distances, and hand orientations
- Capture different speeds of the gesture

### Step 3: Train the Model

1. Teachable Machine automatically trains as you add samples
2. Use the preview on the right to test your gestures
3. Aim for 90%+ accuracy on both classes
4. If accuracy is low, add more training samples

### Step 4: Export the Model

1. Click **"Export Model"** (top right)
2. Select **"TensorFlow.js"**
3. Click **"Upload (cloud)"** (recommended for ease)
4. Wait for upload to complete
5. Copy the model URL

Example URL:
```
https://teachablemachine.withgoogle.com/models/abc123xyz/
```

### Step 5: Configure the Script

Edit `hand_training.py` and replace:
```python
MODEL_URL = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"
```

With your actual model URL.

### Step 6: Run the Script

```bash
python hand_training.py
```

## 📊 Usage

Once running, the script displays:
- **Training Status**: 🟢 ACTIVE or 🔴 PAUSED
- **Current Gesture**: Real-time prediction with confidence
- **Confidence Bars**: Visual representation of all classes

### Controls
- **Thumbs Up Gesture** → Starts/Resumes training
- **Flat Hand Gesture** → Pauses training
- **Press 'q'** → Quit the application

## 🤖 Robot Integration (Optional)

The script can control a **Dobot robot** to pause/resume its motion based on hand gestures.

### Enable Robot Control

Edit `hand_training.py` and set:
```python
USE_ROBOT = True  # Enable robot control
```

### How It Works

When robot control is enabled:
- **Thumbs Up** gesture → Robot **resumes** motion
- **Flat Hand** gesture → Robot **pauses** motion

The script automatically:
1. Detects the robot on COM5
2. Initializes the robot
3. Monitors hand gestures in real-time
4. Pauses/resumes robot based on gestures

### Robot Status Display

The video window shows:
- `ROBOT: RUNNING` 🟢 - Robot is executing motion
- `ROBOT: PAUSED` 🔴 - Robot is paused by flat hand gesture

### Disabling Robot Control

To run without robot control (hand gesture training only):
```python
USE_ROBOT = False
```

Or simply disconnect the Dobot robot from USB and the script will run in training mode only.

## ⚙️ Features

### Gesture Stability
The script uses gesture history (5 frames) to prevent false positives:
- A gesture must be detected consistently across frames to trigger an action
- This reduces noise and accidental triggers

### Confidence Thresholding
- Default threshold: 70%
- Gestures below threshold are shown but don't trigger actions
- Adjust `CONFIDENCE_THRESHOLD` in the script if needed

### Action Cooldown
- Prevents rapid repeated actions
- Default: 0.5 seconds between actions
- Adjust `action_cooldown` in `HandTrainingController` if needed

### Camera Calibration (Optional)
If you have calibrated your camera:
```python
# camera_params.npz will be automatically loaded
USE_CALIBRATION = True
```

## 🔧 Troubleshooting

### Model Not Loading
- Verify the model URL is correct (should end with `/`)
- Check your internet connection
- Ensure TensorFlow is installed: `pip install tensorflow`

### Low Accuracy
- Add more training samples (30-50 per class recommended)
- Ensure good lighting
- Include varied angles and distances
- Retrain the model

### Gestures Not Being Detected
- Lower the confidence threshold slightly: `CONFIDENCE_THRESHOLD = 0.5`
- Make gestures more clearly
- Ensure camera can see your hands
- Check that your model has "Thumbs Up" and "Flat Hand" classes

### Camera Not Opening
- Verify camera ID (0 is default, try 1 or 2 if first doesn't work)
- Check permissions: `CAMERA_ID = 1`
- Ensure no other application is using the camera

### Robot Not Initializing
- Verify Dobot is connected to USB
- Check that the robot is powered on
- Ensure COM5 is the correct port (check Device Manager)
- If using a different port, modify `dobotArm.py`: change `"COM5"` to your port
- Set `USE_ROBOT = False` to disable robot control and test camera-only mode

### Robot Motion Not Pausing
- Ensure robot is not in another control mode
- Check that gestures are being detected (look for gesture in output)
- Verify `use_robot=True` is set in configuration
- Check robot status display in video feed

## 📝 Advanced Usage

### Robot Control Already Integrated

The robot control is already built into `HandTrainingController`. Simply enable it:

```python
USE_ROBOT = True
```

The controller will automatically:
- Initialize the Dobot robot
- Pause/resume motion based on hand gestures
- Display robot status in the video feed
- Handle robot errors gracefully

### Custom Robot Behavior

To extend the robot behavior, modify the `_handle_continue()` and `_handle_stop()` methods:

```python
def _handle_continue(self):
    if super()._handle_continue():
        # Default: resumes robot
        # Add custom logic here:
        # dobotArm.move_to_xyz(self.api, 200, 100, 100)
        return True
    return False

def _handle_stop(self):
    if super()._handle_stop():
        # Default: pauses robot
        # Add custom logic here:
        # dobotArm.move_to_home(self.api)
        return True
    return False
```

### Troubleshooting Robot Connection

If the robot fails to initialize:
1. Ensure Dobot is connected to USB (COM5)
2. Check that the Dobot drivers are installed
3. Set `USE_ROBOT = False` to run in camera-only mode
4. Check terminal output for specific error messages

## 📚 Dependencies

```bash
pip install tensorflow opencv-python numpy
```

## 📄 Files

- `hand_training.py` - Main controller script
- `hand_sign_recognition.py` - Core recognition module (used by this script)
- `camera_params.npz` - Optional calibration file (auto-loaded if present)
