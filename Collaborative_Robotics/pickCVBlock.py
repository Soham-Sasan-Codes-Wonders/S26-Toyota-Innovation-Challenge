#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in three phases: scanning for plates, 
#scanning for targets, and executing the pick/place operations. 
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will first scan for plates, then targets, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.



import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
from pathlib import Path


"""CONSTANTS"""

Z_SAFE = 40 #what is the clearance distance for the robot arm to avoid collisions when moving horizontally?
Z_PICK = -25 #what is the  height for the robot claw to successfully pick up the target?
STABILITY_LIMIT = 60  #how many consecutive frames of stable detection before we "lock in" the positions and move to the next phase? (at 30fps, 60 frames is about 2 seconds)
PIXEL_TOLERANCE = 10  #object can move at most this # of pixels to be considered stationary

machine_state = "scanning plate" 

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
# MAKE SURE THAT YOU HAVE RAN calibrateCamera.py FIRST TO GENERATE THE camera_params.npz FILE
api = dType.load()
cap = cv2.VideoCapture(0)
# Resolve data files relative to this script's directory
_HERE = Path(__file__).resolve().parent
H_matrix = np.load(_HERE / "HomographyMatrix.npy")
data = np.load(_HERE / "camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


# State machine logic to control the flow of the program through the three phases: scanning for plates, scanning for targets, and executing pick/place operations.
# THIS STATE MACHINE IS TOO SIMPLE. Can you think of logics that should change the robot's sequnece of actions?
# Ex: what if the robot fails to pick up a target? should it retry? should it go back to scanning for targets in case the target was moved? what if a new plate is added during the pick/place phase?
# What if a human's hand is in sight during pick/place phase? (safety first!)

def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"
    else:
        machine_state = "scanning plate"



# ---------------------------------------------------------
# PHASE 1: DETECT Part Drop Zones (Plates)
# this script assumes a metallic circular plate as the drop zone, but you can modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=15, maxRadius=55)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        # --- AUTO-LOCK LOGIC ---
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list
  
 

# ---------------------------------------------------------
# PHASE 2: DETECT Red velcros to pick up (Red Blocks)
# this script assumes the targets to be picked up are red blocks
# be aware your target maynot be red, and they may not be rectangular! You will need to modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets. Waiting for stability... (press B to capture background, 1-9 to select)")
    stability_counter = 0
    last_count = 0

    # size thresholds (pixels area)
    MIN_OBJECT_AREA = 200
    MAX_OBJECT_AREA = 800

    # background subtraction params
    BG_FRAMES = 10
    BG_THRESHOLD = 30
    bg_captured = False
    bg_gray = None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray)

        # adaptive threshold to capture objects of differing brightness
        thresh = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # color cue (keep red detection as a strong hint)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255])))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # edge cue (use Canny to find object outlines)
        edges = cv2.Canny(clahe_img, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # background subtraction if captured
        bg_mask = None
        if bg_captured and bg_gray is not None:
            diff = cv2.absdiff(clahe_img, bg_gray)
            _, bg_mask = cv2.threshold(diff, BG_THRESHOLD, 255, cv2.THRESH_BINARY)
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # fuse cues
        combined = thresh
        combined = cv2.bitwise_or(combined, red_mask)
        combined = cv2.bitwise_or(combined, edges)
        if bg_mask is not None:
            combined = cv2.bitwise_or(combined, bg_mask)

        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_OBJECT_AREA or area > MAX_OBJECT_AREA:
                continue
            M = cv2.moments(cnt)
            if M.get("m00", 0) == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            rx, ry = pixel_to_robot(cx, cy, H_matrix)
            detected.append({"robot": (rx, ry), "pixel": (cx, cy), "area": area, "contour": cnt})

        # Draw detected objects with numbers
        for i, obj in enumerate(detected):
            cnt = obj["contour"]
            cx, cy = obj["pixel"]
            cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(display_frame, str(i + 1), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        status_text = f"BG: {'yes' if bg_captured else 'no (press B)'} | Press 1-9 to select object, E to abort"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(display_frame, f"Detected: {len(detected)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # capture a median background from several frames
            frames = []
            for _ in range(BG_FRAMES):
                r2, f2 = cap.read()
                if not r2:
                    continue
                f2 = cv2.remap(f2, map1, map2, cv2.INTER_LINEAR)
                frames.append(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
                cv2.waitKey(30)
            if len(frames) > 0:
                bg_gray = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
                bg_gray = clahe.apply(bg_gray)
                bg_captured = True
                print("Background captured for subtraction")
            continue

        if key == ord('e'):
            print("User aborted target selection.")
            return []

        # number key selection (1-9)
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(detected):
                sel = detected[idx]
                rx, ry = sel["robot"]
                print(f"User selected object #{idx+1} (area={sel['area']:.0f}) at pixel={sel['pixel']}")
                return [(rx, ry)]

        # Stability lock if no selection
        if len(detected) > 0 and len(detected) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(detected)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(detected)} targets.")
            picks = [(o["robot"][0], o["robot"][1]) for o in detected]
            return picks


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# if you are picking up rigid car parts, would you still be able to move directly to the object and to the drop zone? 
# Do you need collision avoidance? Think about if the robot gripper accidentally hits the plate or other parts on the way to the target, what would happen? How would you modify the robot's movement logic to avoid collisions?
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    cv2.VideoCapture(0)
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    # Match 1 part to 1 drop zone (uses the smaller count)
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}")

        # --- PICK SEQUENCE ---
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
        #optional alternate function call method to include a rotation of the gripper angle
        #dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, 45) 

        dobotArm.close_gripper(api)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        # --- PLACE SEQUENCE ---
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    # irl, it is ok for 1 dish to contain multiple parts
    # if len(pick_list) > len(drop_list):
    #     for i in range(len(pick_list)):
    #         pick_x, pick_y = pick_list[i]
    #         drop_x, drop_y = drop_list[0]
    #         # --- PICK SEQUENCE ---
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
    #         dobotArm.close_gripper(api)
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

    #     # --- PLACE SEQUENCE ---
    #         dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
    #         dobotArm.open_gripper(api)
    #         dobotArm.stop_pump(api)
    #         dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("\nBatch Complete.")
    return True
 

# ---------------------------------------------------------
# MAIN EXECUTION
# contains an oversimplified state machine that runs the three phases sequentially. You can modify the logic to fit your specific use case.
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

while machine_state == "scanning plate":
    drop_zone = phase_detect_plates()
    if drop_zone is not None:
        next_state()


while machine_state == "scanning target":
    pick_target = phase_detect_targets()
    if pick_target is not None:
        next_state()


while machine_state == "pick place":
    completed = phase_execute_batch(api, pick_target, drop_zone)
    if completed:
        next_state()
    else: break


cap.release()
cv2.destroyAllWindows()