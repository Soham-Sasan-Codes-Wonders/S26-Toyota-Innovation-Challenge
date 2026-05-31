"""pickCVBlocklocal.py
Local test of detection + SPACE/ESC UX (no robot calls)

Saves a lightweight, standalone copy of the detection/UX logic so you can
test with an integrated webcam without moving the robot. Uses optional
`HomographyMatrix.npy` and `camera_params.npz` if present next to the script.
"""
import cv2
import numpy as np
import time
from pathlib import Path

Z_SAFE = 40
Z_PICK = -25
STABILITY_LIMIT = 60
PIXEL_TOLERANCE = 10
EXIT_KEY = 27
RESTART_KEY = 32

cap = cv2.VideoCapture(0)
_HERE = Path(__file__).resolve().parent

# Try load homography and camera params (optional)
H_matrix = None
map1 = map2 = None
try:
    H_matrix = np.load(_HERE / "HomographyMatrix.npy")
except Exception:
    H_matrix = None

try:
    data = np.load(_HERE / "camera_params.npz")
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    # prepare undistort maps after we get a frame
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)
except Exception:
    map1 = map2 = None


def pixel_to_robot(u, v, H):
    if H is None:
        return float(u), float(v)
    p = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return float(xy[0]), float(xy[1])


def draw_overlay(frame, lines, font_scale=0.7, thickness=2):
    overlay = frame.copy()
    margin = 10
    line_h = int(28 * font_scale)
    box_h = margin*2 + line_h * len(lines)
    cv2.rectangle(overlay, (0,0), (frame.shape[1], box_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = margin + int(line_h * 0.8)
    for line in lines:
        cv2.putText(frame, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        y += line_h


def remap_if_needed(frame):
    if map1 is not None and map2 is not None:
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return frame


def wait_for_space_or_esc():
    while True:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        f = remap_if_needed(f)
        draw_overlay(f, ["Paused: Press SPACE to start/continue", "Press ESC to exit"])
        cv2.imshow("Detection", f)
        k = cv2.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            return EXIT_KEY
        if k == RESTART_KEY:
            return RESTART_KEY


def phase_detect_plates():
    stability_counter = 0
    last_positions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = remap_if_needed(frame)
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 9)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=150,
                                   param1=100, param2=30, minRadius=20, maxRadius=45)
        current_pixels = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 2, (0, 0, 255), 3)
                current_pixels.append((cx, cy))
        # stability
        if len(current_pixels) > 0 and len(current_pixels) == len(last_positions):
            current_pixels.sort(key=lambda p: p[0])
            last_positions.sort(key=lambda p: p[0])
            is_stable = True
            for p1, p2 in zip(current_pixels, last_positions):
                d = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if d > PIXEL_TOLERANCE:
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
        color = (0,255,0) if progress < 100 else (255,255,0)
        cv2.putText(display, f"LOCKING PLATES: {progress}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        draw_overlay(display, ["SPACE: restart", "ESC: exit"])
        cv2.imshow("Detection", display)
        k = cv2.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            return None
        if k == RESTART_KEY:
            stability_counter = 0
            last_positions = []
        if stability_counter >= STABILITY_LIMIT:
            mapped = [pixel_to_robot(x,y,H_matrix) for x,y in current_pixels]
            return current_pixels, mapped


def phase_detect_targets():
    stability_counter = 0
    last_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = remap_if_needed(frame)
        display = frame.copy()
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_pixels = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    current_pixels.append((cx, cy))
                    cv2.drawContours(display, [cnt], -1, (0,255,0), 2)
        # stability by count + approximate positions
        if len(current_pixels) > 0:
            if len(current_pixels) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_pixels)
        else:
            stability_counter = 0
            last_count = 0
        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0,255,0) if progress < 100 else (255,255,0)
        cv2.putText(display, f"LOCKING TARGETS: {progress}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        draw_overlay(display, ["SPACE: restart", "ESC: exit"])
        cv2.imshow("Detection", display)
        k = cv2.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            return None
        if k == RESTART_KEY:
            stability_counter = 0
            last_count = 0
        if stability_counter >= STABILITY_LIMIT:
            mapped = [pixel_to_robot(x,y,H_matrix) for x,y in current_pixels]
            return current_pixels, mapped


def main():
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    print("Local test: press SPACE in the Detection window to start, ESC to exit.")
    r = wait_for_space_or_esc()
    if r == EXIT_KEY:
        cap.release()
        cv2.destroyAllWindows()
        return
    running = True
    while running:
        print("Scanning plates...")
        plates = phase_detect_plates()
        if plates is None:
            print("Exit requested during plate detection.")
            break
        pixels_plate, mapped_plate = plates
        print(f"Detected plates (pixels): {pixels_plate}")
        print(f"Mapped plate coords (if homography present): {mapped_plate}")

        print("Scanning targets...")
        targets = phase_detect_targets()
        if targets is None:
            print("Exit requested during target detection.")
            break
        pixels_target, mapped_target = targets
        print(f"Detected targets (pixels): {pixels_target}")
        print(f"Mapped target coords (if homography present): {mapped_target}")

        # For local test we don't control robot; just show results then wait for user
        print("Batch complete. Press SPACE to run another test, ESC to exit.")
        r = wait_for_space_or_esc()
        if r == EXIT_KEY:
            running = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
