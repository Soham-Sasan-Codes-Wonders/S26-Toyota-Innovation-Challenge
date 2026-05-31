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
EXIT_KEY = 27   # ESC to stop
RESTART_KEY = 32  # SPACE to restart after a job

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
    last_positions = []
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Increased blur to reduce edge noise that throws off HoughCircles
        blurred = cv2.medianBlur(gray, 9)
        
        # Adjusted radii and accumulator threshold for better reliability
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=150, 
                                   param1=100, param2=30, minRadius=20, maxRadius=45)

        current_list = []
        current_pixels = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                # Draw the circumference and the center
                cv2.circle(display_frame, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), 3)
                
                rx, ry = pixel_to_robot(cx, cy, H_matrix)
                current_list.append((rx, ry))
                current_pixels.append((cx, cy))

        # --- AUTO-LOCK LOGIC WITH POSITIONAL STABILITY ---
        if len(current_pixels) > 0 and len(current_pixels) == len(last_positions):
            # Sort roughly by X axis to pair up corresponding circles between frames
            current_pixels.sort(key=lambda p: p[0])
            last_positions.sort(key=lambda p: p[0])
            
            is_stable = True
            for p1, p2 in zip(current_pixels, last_positions):
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist > PIXEL_TOLERANCE:
                    is_stable = False
                    break
                    
            if is_stable:
                stability_counter += 1
            else:
                stability_counter = 0
        else:
            stability_counter = 0
            
        last_positions = current_pixels.copy()

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, "SPACE: restart   ESC: exit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == EXIT_KEY:
            return None
        if key == RESTART_KEY:
            stability_counter = 0
            last_positions = []

        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} plates.")
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
        cv2.putText(display_frame, "SPACE: restart   ESC: exit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == EXIT_KEY:
            return None
        if key == RESTART_KEY:
            stability_counter = 0
            last_count = 0

        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            return current_list


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# if you are picking up rigid car parts, would you still be able to move directly to the object and to the drop zone? 
# Do you need collision avoidance? Think about if the robot gripper accidentally hits the plate or other parts on the way to the target, what would happen? How would you modify the robot's movement logic to avoid collisions?
# ---------------------------------------------------------
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

running = True
print("Press ESC in the detection window to stop. After a job completes press SPACE to restart.")

while running:
    # PHASE 1: detect drop zones
    machine_state = "scanning plate"
    drop_zone = phase_detect_plates()
    if drop_zone is None:
        print("Exit requested during plate detection.")
        break

    # PHASE 2: detect targets
    machine_state = "scanning target"
    pick_target = phase_detect_targets()
    if pick_target is None:
        print("Exit requested during target detection.")
        break

    # PHASE 3: execute pick/place
    machine_state = "pick place"
    completed = phase_execute_batch(api, pick_target, drop_zone)
    if not completed:
        print("Batch aborted.")
        break

    # Wait for user to either restart or exit
    print("\nBatch complete. Press SPACE to start next job, or ESC to exit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == EXIT_KEY:
            running = False
            break
        if key == RESTART_KEY:
            break

cap.release()
cv2.destroyAllWindows()