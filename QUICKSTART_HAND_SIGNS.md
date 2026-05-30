# Quick Start: Hand Sign Recognition for Dobot

## 📋 Overview

This guide helps you set up hand sign recognition using **Google Teachable Machine** with your Dobot robot and calibrated camera.

**What you'll get:**
- Real-time hand sign detection via webcam
- Integration with calibrated camera (undistorted)
- Optional: Robot actions triggered by hand signs

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install tensorflow opencv-python numpy requests
```

### Step 2: Train a Model in Teachable Machine

1. Go to **[teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/)**
2. Click **"Get Started"**
3. Select **"Pose"** project type (important: NOT "Image")
4. Add classes for each hand sign (e.g., "Thumbs Up", "Peace", "Open Hand", "Fist")
5. For each class:
   - Collect 20-30 hand poses at different angles
   - Be consistent with hand shape and position
6. Train (automatic)
7. Test in preview

### Step 3: Export Model

1. Click **"Export Model"**
2. Select **"TensorFlow.js"**
3. Choose **"Upload (Cloud)"** (easier)
4. Copy the model URL
> There is no separate "model number" to enter. The Teachable Machine export page gives you a URL that contains the model identifier.
Example URL:
```
https://teachablemachine.withgoogle.com/models/abc123xyz/
```

### Step 4: Run the Example

```python
# Edit example_hand_signs.py and replace:
MODEL_URL = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"

# Then run:
python example_hand_signs.py
```

---

## 📁 Files in This Folder

| File | Purpose |
|------|---------|
| `hand_sign_recognition.py` | Core module for detection |
| `example_hand_signs.py` | Simple usage example |
| `HAND_SIGNS_README.md` | Detailed reference guide |

---

## 💡 Quick Examples

### Example 1: Basic Detection

```python
from hand_sign_recognition import HandSignRecognizer

model_url = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"
recognizer = HandSignRecognizer(model_url, camera_id=0)
recognizer.run()
```

### Example 2: With Calibrated Camera

```python
import numpy as np
from hand_sign_recognition import HandSignRecognizer

# Load calibration
data = np.load("camera_params.npz")
recognizer = HandSignRecognizer(model_url, camera_id=0)
recognizer.set_calibration(data["camera_matrix"], data["dist_coeffs"])
recognizer.run()
```

### Example 3: Detect Single Frame

```python
recognizer = HandSignRecognizer(model_url, camera_id=0)
sign, confidence = recognizer.detect_once()
print(f"Detected: {sign} ({confidence:.1%})")
```

### Example 4: Map to Robot Actions (Advanced)

```python
import dobotArm
import lib.DobotDllType as dType

api = dType.load()
dobotArm.initialize_robot(api)

recognizer = HandSignRecognizer(model_url)

while True:
    sign, conf = recognizer.detect_once()
    
    if sign == "Thumbs Up" and conf > 0.8:
        dobotArm.move_to_xyz(api, 200, 50, 100)
    elif sign == "Open Hand" and conf > 0.8:
        dobotArm.open_gripper(api)
    elif sign == "Fist" and conf > 0.8:
        dobotArm.close_gripper(api)
```

---

## ⚙️ Configuration Options

### Confidence Threshold

```python
recognizer = HandSignRecognizer(
    model_url, 
    confidence_threshold=0.7  # 0.0-1.0
)
```

- Higher threshold = fewer false positives, but may miss signs
- Typical range: 0.6 - 0.8

### Camera ID

```python
recognizer = HandSignRecognizer(
    model_url, 
    camera_id=0  # Usually 0 for default camera
)
```

---

## 🎓 Teachable Machine Tips

### ✓ Do This

- **Collect diverse samples**: Different lighting, angles, distances
- **Use consistent gestures**: Each class should look the same every time
- **Add extra samples**: More data = better model
- **Test thoroughly**: Preview window shows live accuracy
- **Multiple hand poses**: Include from different sides

### ✗ Avoid This

- **Too few samples**: <10 per class → poor accuracy
- **Blurry captures**: Move slowly to avoid motion blur
- **Mixed backgrounds**: Stick to similar environments
- **Inconsistent shapes**: Same sign should always look the same
- **Ambiguous classes**: If classes are too similar, model gets confused

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not found" | Check URL ends with `/` |
| "Camera not opening" | Try `camera_id=1` or different value |
| "Low accuracy" | Add more training samples, better lighting |
| "Slow performance" | Reduce frame resolution or use GPU |
| "False positives" | Increase confidence threshold |
| "Can't import tensorflow" | Run `pip install tensorflow` |

---

## 📊 Model Performance

Expected accuracy with good training:
- **Well-separated poses** (e.g., fist vs. open hand): 90%+ accuracy
- **Similar poses** (e.g., peace vs. thumbs up): 70-80% accuracy
- **Poor training** (few samples): 40-60% accuracy

**Tip**: If accuracy is low, add more varied training samples!

---

## 🎬 What's Next?

1. Train your model with good data
2. Export and test with `example_hand_signs.py`
3. Add robot control logic for your use case
4. Fine-tune confidence threshold for your environment

---

## 📚 Resources

- **Teachable Machine**: https://teachablemachine.withgoogle.com/
- **TensorFlow Docs**: https://www.tensorflow.org/
- **OpenCV Docs**: https://opencv.org/
- **Dobot API**: See `lib/DobotDllType.py`

---

## 🆘 Need Help?

- Check `HAND_SIGNS_README.md` for detailed reference
- Review examples in `example_hand_signs.py`
- Test with `python example_hand_signs.py` first
- Verify model URL is correct before running

**Happy training!** 🚀
