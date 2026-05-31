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



import argparse
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

# --- CLI and INITIALIZATION FOR CAMERA TRANSFORMATION ---
parser = argparse.ArgumentParser(description="pickCVBlock with improved detection options")
parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
parser.add_argument("--no-edges", action="store_true", help="Disable edge/Canny cue")
parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (use raw grayscale)")
parser.add_argument("--no-red", action="store_true", help="Disable red-color cue")
parser.add_argument("--no-undistort", action="store_true", help="Do not undistort frames (skip remap)")
parser.add_argument("--morph-kernel", type=int, default=5, help="Morphology kernel size (odd int)")
parser.add_argument("--morph-iterations", type=int, default=1, help="Morphology iterations for close")
parser.add_argument("--min-area", type=int, default=4, help="Minimum contour area (pixels)")
parser.add_argument("--max-area", type=int, default=10000, help="Maximum contour area (pixels)")
parser.add_argument("--min-solidity", type=float, default=0.20, help="Minimum contour solidity")
parser.add_argument("--min-extent", type=float, default=0.15, help="Minimum extent (area / bbox)")
parser.add_argument("--min-circularity", type=float, default=0.02, help="Minimum circularity")
parser.add_argument("--min-aspect-ratio", type=float, default=0.25, help="Minimum bounding box aspect ratio (w/h)")
parser.add_argument("--max-aspect-ratio", type=float, default=3.5, help="Maximum bounding box aspect ratio (w/h)")
parser.add_argument("--stability-frames", type=int, default=3, help="Frames a detection must persist to be considered stable")
parser.add_argument("--stability-lock-frames", type=int, default=30, help="Frames required to auto-lock detections")
parser.add_argument("--pixel-tolerance", type=int, default=12, help="Pixel tolerance for centroid matching")
parser.add_argument("--bg-frames", type=int, default=10, help="Frames to capture for background median")
parser.add_argument("--bg-threshold", type=int, default=40, help="Threshold for background diff mask")
parser.add_argument("--show-mask", action="store_true", help="Show combined mask window for tuning")
parser.add_argument("--debug-windows", action="store_true", help="Show individual processing windows (CLAHE, thresh, red, edges, bg)")
parser.add_argument("--debug", action="store_true", help="Print debug info to console")
parser.add_argument("--overlay-mask", action="store_true", help="Draw translucent mask overlay on camera view")
parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Alpha for mask overlay (0.0-1.0)")
parser.add_argument("--center-roi", type=float, default=0.60, help="Fraction of frame to restrict detection to center (0-1); 1.0 disables cropping")
args = parser.parse_args()

api = dType.load()

# Resolve data files relative to this script's directory
_HERE = Path(__file__).resolve().parent
H_matrix = np.load(_HERE / "HomographyMatrix.npy")
data = np.load(_HERE / "camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# open camera using requested index
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {args.camera}")

# warm-up and compute undistort maps
time.sleep(0.2)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read frame from camera during init")
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
    # tracking and background state
    tracks = []
    next_track_id = 1
    frame_idx = 0
    bg_captured = False
    bg_gray = None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if not args.no_clahe else None
    k = max(3, args.morph_kernel | 1)
    kernel_close = np.ones((k, k), np.uint8)
    open_k = max(3, int(np.sqrt(max(1, args.min_area))))
    if open_k % 2 == 0:
        open_k += 1
    if open_k > k:
        open_k = k
    kernel_open = np.ones((open_k, open_k), np.uint8)

    # center ROI
    if args.center_roi and args.center_roi > 0 and args.center_roi < 1.0:
        halfw = int(w * args.center_roi / 2.0)
        halfh = int(h * args.center_roi / 2.0)
        roi_x1 = max(0, (w // 2) - halfw)
        roi_y1 = max(0, (h // 2) - halfh)
        roi_x2 = min(w, (w // 2) + halfw)
        roi_y2 = min(h, (h // 2) + halfh)
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if not args.no_undistort and map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray) if clahe is not None else gray

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # red mask as color cue
        if not getattr(args, 'no_red', False):
            red_mask = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255])))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        else:
            red_mask = np.zeros_like(thresh)

        # edge cue
        if not args.no_edges:
            edges = cv2.Canny(clahe_img, 50, 150)
            edges = cv2.dilate(edges, kernel_close, iterations=1)
        else:
            edges = np.zeros_like(thresh)

        # background subtraction primary cue when available
        if bg_captured and bg_gray is not None:
            cur_blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
            bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
            diff = cv2.absdiff(cur_blur, bg_blur)
            _, bg_mask = cv2.threshold(diff, args.bg_threshold, 255, cv2.THRESH_BINARY)
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=args.morph_iterations)
            combined = bg_mask.copy()
        else:
            combined = thresh.copy()
            combined = cv2.bitwise_or(combined, red_mask)
            combined = cv2.bitwise_or(combined, edges)
            bg_mask = None

        # morphology: small opening to preserve tiny objects
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=args.morph_iterations)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        raw_count = len(contours)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area or area > args.max_area:
                continue
            x, y, wbox, hbox = cv2.boundingRect(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) if hull is not None else 0
            solidity = float(area / hull_area) if hull_area > 0 else 0.0
            extent = float(area) / (wbox * hbox) if (wbox * hbox) > 0 else 0.0
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            aspect = float(wbox) / hbox if hbox > 0 else float('inf')

            M = cv2.moments(cnt)
            if M.get('m00', 0) == 0:
                continue
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            # restrict to central ROI if requested
            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                continue

            # size-adaptive relaxation for small objects
            rel = min(1.0, area / max(1.0, (args.min_area * 4.0)))
            rel = max(0.5, rel)
            eff_min_solidity = args.min_solidity * rel
            eff_min_extent = args.min_extent * rel
            eff_min_circularity = args.min_circularity * rel

            if solidity < eff_min_solidity:
                continue
            if extent < eff_min_extent:
                continue
            if circularity < eff_min_circularity:
                continue
            if aspect < args.min_aspect_ratio or aspect > args.max_aspect_ratio:
                continue

            rx, ry = pixel_to_robot(cx, cy, H_matrix)
            candidates.append({'robot': (rx, ry), 'pixel': (cx, cy), 'area': area, 'contour': cnt,
                                'solidity': solidity, 'extent': extent, 'circularity': circularity,
                                'bbox': (x, y, wbox, hbox), 'aspect': aspect})

        # Tracking and per-track locks
        for cand in candidates:
            cx, cy = cand['pixel']
            best = None
            best_d = float('inf')
            for t in tracks:
                tx, ty = t['centroid']
                d = (cx - tx) ** 2 + (cy - ty) ** 2
                if d < (args.pixel_tolerance ** 2) and d < best_d:
                    best = t
                    best_d = d
            if best is not None:
                best['centroid'] = (cx, cy)
                best['stability'] += 1
                best['last_seen'] = frame_idx
                best['contour'] = cand['contour']
                best['area'] = cand['area']
                best['solidity'] = cand['solidity']
                best['extent'] = cand['extent']
                best['circularity'] = cand['circularity']
                best['bbox'] = cand['bbox']
                best['pixel'] = cand['pixel']
                best['robot'] = cand.get('robot', best.get('robot'))

                if best['stability'] >= args.stability_frames:
                    if 'lock_centroid' not in best or best.get('lock_counter', 0) == 0:
                        best['lock_centroid'] = (cx, cy)
                        best['lock_counter'] = 1
                        best['locked'] = False
                    else:
                        dx = cx - best['lock_centroid'][0]
                        dy = cy - best['lock_centroid'][1]
                        if dx * dx + dy * dy <= (args.pixel_tolerance ** 2):
                            best['lock_counter'] = best.get('lock_counter', 0) + 1
                        else:
                            best['lock_counter'] = 1
                            best['lock_centroid'] = (cx, cy)
                            best['locked'] = False

                    if not best.get('locked', False) and best.get('lock_counter', 0) >= args.stability_lock_frames:
                        best['locked'] = True
                        print(f"[LOCKED] Track {best['id']} locked at pixel {best['pixel']} robot {best.get('robot')}")
                else:
                    best['lock_counter'] = 0
                    best['locked'] = False
            else:
                tracks.append({'id': next_track_id, 'centroid': (cx, cy), 'stability': 1,
                               'last_seen': frame_idx, 'contour': cand['contour'], 'area': cand['area'],
                               'solidity': cand['solidity'], 'extent': cand['extent'], 'circularity': cand['circularity'],
                               'bbox': cand['bbox'], 'pixel': cand['pixel'], 'robot': cand.get('robot'),
                               'lock_counter': 0, 'lock_centroid': (cx, cy), 'locked': False})
                next_track_id += 1

        # prune stale tracks
        new_tracks = []
        for t in tracks:
            if frame_idx - t.get('last_seen', 0) <= max(2, args.stability_frames * 2):
                new_tracks.append(t)
        tracks = new_tracks

        stable_tracks = [t for t in tracks if t['stability'] >= args.stability_frames]

        if args.debug:
            cand_areas = [int(c['area']) for c in candidates]
            stable_areas = [int(t['area']) for t in stable_tracks]
            print(f"Raw contours: {raw_count}  Candidates: {len(candidates)}  Stable: {len(stable_tracks)}  CandidateAreas: {cand_areas}  StableAreas: {stable_areas}")

        # draw stable tracks
        for i, obj in enumerate(stable_tracks):
            cnt = obj['contour']
            cx, cy = obj['pixel']
            area = int(obj['area'])
            x, y, wbox, hbox = obj.get('bbox', cv2.boundingRect(cnt))
            lock_counter = obj.get('lock_counter', 0)
            lock_progress = int(min(100, (lock_counter / max(1, args.stability_lock_frames)) * 100))
            locked = obj.get('locked', False)
            if locked:
                contour_color = (0, 255, 0)
                rect_color = (0, 200, 0)
                text_color = (0, 200, 0)
            else:
                contour_color = (0, 165, 255)
                rect_color = (255, 0, 0)
                text_color = (0, 200, 200)
            cv2.drawContours(display_frame, [cnt], -1, contour_color, 2 if locked else 1)
            cv2.rectangle(display_frame, (x, y), (x + wbox, y + hbox), rect_color, 2 if locked else 1)
            cv2.putText(display_frame, f"{i+1}", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(display_frame, f"A:{area}", (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            if locked:
                cv2.putText(display_frame, "LOCKED", (cx + 10, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, f"Lock:{lock_progress}%", (cx + 10, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        status_text = f"BG: {'yes' if bg_captured else 'no (press B)'} | Press 1-9 to select object, E to abort"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(display_frame, f"Detected: {len(stable_tracks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if args.debug_windows:
            cv2.imshow('CLAHE' if clahe is not None else 'Gray', clahe_img)
            cv2.imshow('Thresh', thresh)
            cv2.imshow('Red', red_mask)
            cv2.imshow('Edges', edges)
            if bg_mask is not None:
                cv2.imshow('BG Mask', bg_mask)
        if args.show_mask:
            cv2.imshow('Mask', combined)
        cv2.imshow('Detection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            frames = []
            for _ in range(args.bg_frames):
                r2, f2 = cap.read()
                if not r2:
                    continue
                if not args.no_undistort and map1 is not None:
                    f2 = cv2.remap(f2, map1, map2, cv2.INTER_LINEAR)
                frames.append(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
                cv2.waitKey(30)
            if frames:
                bg_gray = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
                if clahe is not None:
                    bg_gray = clahe.apply(bg_gray)
                bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
                bg_captured = True
                tracks = []
                next_track_id = 1
                print('Background captured and tracks cleared')
            continue

        if key == ord('e'):
            print("User aborted target selection.")
            return []

        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(stable_tracks):
                sel = stable_tracks[idx]
                if not sel.get('locked', False):
                    print(f"Selection #{idx+1} ignored: object not locked yet ({sel.get('lock_counter',0)}/{args.stability_lock_frames})")
                else:
                    rx, ry = sel['robot']
                    print(f"User selected object #{idx+1} (area={sel['area']:.0f}) at pixel={sel['pixel']}")
                    return [(rx, ry)]

        # auto-return when all stable tracks are locked
        if len(stable_tracks) > 0 and all([t.get('locked', False) for t in stable_tracks]):
            picks = [(t['robot'][0], t['robot'][1]) for t in stable_tracks]
            print(f"[AUTO] Locked {len(picks)} targets: {picks}")
            return picks

        frame_idx += 1


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