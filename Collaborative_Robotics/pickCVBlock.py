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
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        # Create a display copy so drawings don't affect next frame's HSV detection
        display_frame = frame.copy()
        
        # Red Tag Logic
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    rx, ry = pixel_to_robot(cx, cy, H_matrix)
                    current_list.append((rx, ry))
                    # Draw on display_frame only
                    cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
                    
        cv2.waitKey(1)

        # --- STABILITY LOGIC ---
        if len(current_list) != 0:
            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

        # Visual Feedback
        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        
        # --- EXIT CONDITION ---
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            #cv2.waitKey(500) # Brief pause so you can see the 100%
    
            return current_list


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# if you are picking up rigid car parts, would you still be able to move directly to the object and to the drop zone? 
# Do you need collision avoidance? Think about if the robot gripper accidentally hits the plate or other parts on the way to the target, what would happen? How would you modify the robot's movement logic to avoid collisions?
# ---------------------------------------------------------
# Quick verification: map robot coords back to pixel to check if object remains after a pick
# Tuning constants
MAX_PICK_RETRIES = 3
PIXEL_CHECK_THRESHOLD = 40  # pixels


def quick_check_pick_present(pick_x, pick_y, frames=5):
    # Return True if an object (red blob) is still present near the given robot xy.
    try:
        invH = np.linalg.inv(H_matrix)
    except Exception:
        return False
    p = np.array([pick_x, pick_y, 1.0])
    px = invH @ p
    if px[2] == 0:
        return False
    u = int(round(px[0] / px[2]))
    v = int(round(px[1] / px[2]))

    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            M = cv2.moments(cnt)
            if M.get('m00', 0) == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if np.hypot(cx - u, cy - v) <= PIXEL_CHECK_THRESHOLD:
                return True
    return False


def phase_execute_batch(api, pick_list, drop_list):
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

        # --- PICK SEQUENCE with retries ---
        picked = False
        for attempt in range(1, MAX_PICK_RETRIES + 1):
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
            dobotArm.close_gripper(api)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

            # quick camera check: if object still present near pick coords -> pick failed
            still_present = quick_check_pick_present(pick_x, pick_y)
            if not still_present:
                picked = True
                break

            print(f"Pick attempt {attempt} failed for item {i+1} (object still at pick location). Retrying...")
            dobotArm.move_to_home(api)
            time.sleep(0.5)

        if not picked:
            print(f"Failed to pick item {i+1} after {MAX_PICK_RETRIES} attempts.")
            return False

        # --- PLACE SEQUENCE ---
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("\nBatch Complete.")
    return True
 

# ---------------------------------------------------------
# MAIN EXECUTION
# contains an oversimplified state machine that runs the three phases sequentially. You can modify the logic to fit your specific use case.
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

MAX_RESCAN_CYCLES = 3

running = True
print("Press ESC in the detection window to stop.")

while running:
    # PHASE 1: detect drop zones
    drop_zone = phase_detect_plates()
    if drop_zone is None:
        print("Exit requested during plate detection.")
        break

    # PHASE 2: detect targets
    pick_target = phase_detect_targets()
    if pick_target is None:
        print("Exit requested during target detection.")
        break

    # PHASE 3: execute pick/place with rescan retry logic
    completed = phase_execute_batch(api, pick_target, drop_zone)
    if not completed:
        print("Batch failed. Attempting to rescan and retry.")
        retry = 0
        while retry < MAX_RESCAN_CYCLES and not completed:
            retry += 1
            print(f"Rescan retry {retry}/{MAX_RESCAN_CYCLES}")
            drop_zone = phase_detect_plates()
            if drop_zone is None:
                print("Exit during plate detection on rescan.")
                running = False
                break
            pick_target = phase_detect_targets()
            if pick_target is None:
                print("Exit during target detection on rescan.")
                running = False
                break
            completed = phase_execute_batch(api, pick_target, drop_zone)

        if not completed:
            print("Retries exhausted or aborted; returning to scanning for next job.")
            continue

    # Batch succeeded
    print("Batch complete. Moving to HOME and continuing scanning.")
    try:
        dobotArm.move_to_home(api)
    except Exception:
        pass
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)
    time.sleep(0.5)

# Clean up
cap.release()
cv2.destroyAllWindows()