#!/usr/bin/env python3
"""Run a standalone detection demo using the webcam.

This script fuses CLAHE, adaptive threshold, Canny edges and an optional
background subtraction to detect objects by size. Objects are numbered on-screen
and you can press 1-9 to select one. The script prints the chosen object's
robot coordinates (using HomographyMatrix.npy) and exits.

If `camera_params.npz` or `HomographyMatrix.npy` are missing, the script
creates a reasonable fallback so you can still test detection without a
calibrated setup. For best results run `demo/create_sample_data.py` or
`calibrateCamera.py` on a machine with the physical camera.
"""

import argparse
import os
import time
import numpy as np
import cv2
import json


def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1.0])
    xy = H @ p
    if xy[2] == 0:
        return float(xy[0]), float(xy[1])
    xy = xy / xy[2]
    return float(xy[0]), float(xy[1])


def main():
    parser = argparse.ArgumentParser(description="Run detection demo with webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area")
    parser.add_argument("--max-area", type=int, default=10000, help="Maximum contour area")
    parser.add_argument("--min-solidity", type=float, default=0.25, help="Minimum contour solidity (area/convex_hull_area)")
    parser.add_argument("--debug", action="store_true", help="Print debug info to console")
    parser.add_argument("--show-mask", action="store_true", help="Show combined mask window for tuning")
    parser.add_argument("--debug-windows", action="store_true", help="Show individual processing windows (CLAHE, thresh, red, edges, bg)")
    parser.add_argument("--no-edges", action="store_true", help="Disable edge/Canny cue")
    parser.add_argument("--no-red", action="store_true", help="Disable red-color cue")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (use raw grayscale)")
    parser.add_argument("--no-undistort", action="store_true", help="Do not undistort frames (skip remap)")
    parser.add_argument("--morph-kernel", type=int, default=9, help="Morphology kernel size (odd int)")
    parser.add_argument("--morph-iterations", type=int, default=2, help="Morphology iterations for close")
    parser.add_argument("--overlay-mask", action="store_true", help="Draw translucent mask overlay on camera view")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Alpha for mask overlay (0.0-1.0)")
    parser.add_argument("--bg-frames", type=int, default=10, help="Frames to capture for background median")
    parser.add_argument("--bg-threshold", type=int, default=60, help="Threshold for background diff mask")
    parser.add_argument("--min-extent", type=float, default=0.35, help="Minimum extent (area / bounding box area)")
    parser.add_argument("--min-circularity", type=float, default=0.12, help="Minimum circularity (4πA / P^2)")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.25, help="Minimum bounding box aspect ratio (w/h)")
    parser.add_argument("--max-aspect-ratio", type=float, default=3.5, help="Maximum bounding box aspect ratio (w/h)")
    parser.add_argument("--stability-frames", type=int, default=3, help="Frames a detection must persist to be considered stable")
    parser.add_argument("--pixel-tolerance", type=int, default=20, help="Pixels for centroid matching across frames")
    parser.add_argument("--stability-lock-frames", type=int, default=60, help="Frames required to auto-lock detections (progress shown)")
    parser.add_argument("--center-roi", type=float, default=1.0, help="Fraction of frame to restrict detection to center (0-1); 1.0 disables cropping")
    parser.add_argument("--exclude-specular", action="store_true", help="Exclude specular highlights from mask (low saturation, high value)")
    parser.add_argument("--specular-v-threshold", type=int, default=200, help="Value (V) lower bound for specular detection 0-255")
    parser.add_argument("--specular-s-threshold", type=int, default=60, help="Saturation (S) upper bound for specular detection 0-255")
    parser.add_argument("--save-selected", type=str, default=None, help="Path to write JSON with selected object details")
    parser.add_argument("--append-selected", action="store_true", help="Append selected JSON to file instead of overwriting")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cam_params = os.path.join(base, 'camera_params.npz')
    H_file = os.path.join(base, 'HomographyMatrix.npy')

    # Load or create fallback calibration / homography
    if os.path.exists(cam_params):
        data = np.load(cam_params)
        camera_matrix = data.get('camera_matrix')
        dist_coeffs = data.get('dist_coeffs')
    else:
        print('camera_params.npz not found — using fallback intrinsics')
        camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((5,))

    if os.path.exists(H_file):
        H = np.load(H_file)
    else:
        print('HomographyMatrix.npy not found — using identity homography')
        H = np.eye(3)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Cannot open camera index {args.camera}')
        return

    # warm-up
    time.sleep(0.2)
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame from camera')
        cap.release()
        return

    h, w = frame.shape[:2]
    # compute central ROI if requested
    if args.center_roi and args.center_roi > 0 and args.center_roi < 1.0:
        halfw = int(w * args.center_roi / 2.0)
        halfh = int(h * args.center_roi / 2.0)
        roi_x1 = max(0, (w // 2) - halfw)
        roi_y1 = max(0, (h // 2) - halfh)
        roi_x2 = min(w, (w // 2) + halfw)
        roi_y2 = min(h, (h // 2) + halfh)
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
    new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # ensure odd kernel size and at least 3
    k = max(3, args.morph_kernel | 1)
    # closing kernel (fills gaps)
    kernel_close = np.ones((k, k), np.uint8)
    # opening kernel sized relative to min area to avoid removing very small objects
    open_k = max(3, int(np.sqrt(max(1, args.min_area))))
    if open_k % 2 == 0:
        open_k += 1
    if open_k > k:
        open_k = k
    kernel_open = np.ones((open_k, open_k), np.uint8)

    bg_captured = False
    bg_gray = None
    # simple short-term tracker to enforce stability across frames
    tracks = []
    next_track_id = 1
    frame_idx = 0

    print('\nDetection running — press B to capture background (recommended), 1-9 to select an object, E to exit')

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # undistort unless disabled
        if not args.no_undistort and map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray) if not args.no_clahe else gray

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # compute HSV once (used by red and specular removal)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # red mask as color cue (optional)
        if not args.no_red:
            red_mask = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255])))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        else:
            red_mask = np.zeros_like(thresh)

        # edge cue (optional)
        if not args.no_edges:
            edges = cv2.Canny(clahe_img, 50, 150)
            edges = cv2.dilate(edges, kernel_close, iterations=1)
        else:
            edges = np.zeros_like(thresh)

        # If a background was captured, use background difference as the primary cue
        if bg_captured and bg_gray is not None:
            cur = clahe_img.copy()
            cur_blur = cv2.GaussianBlur(cur, (5, 5), 0)
            bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
            diff = cv2.absdiff(cur_blur, bg_blur)
            _, bg_mask = cv2.threshold(diff, args.bg_threshold, 255, cv2.THRESH_BINARY)
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=args.morph_iterations)
            combined = bg_mask.copy()
        else:
            # fuse cues (start from threshold and merge other masks)
            combined = thresh.copy()
            if red_mask is not None:
                combined = cv2.bitwise_or(combined, red_mask)
            if edges is not None:
                combined = cv2.bitwise_or(combined, edges)
            bg_mask = None

        # remove specular highlights (bright, low-saturation) if requested
        if args.exclude_specular:
            spec_mask = cv2.inRange(hsv, (0, 0, args.specular_v_threshold), (179, args.specular_s_threshold, 255))
            spec_mask = cv2.morphologyEx(spec_mask, cv2.MORPH_CLOSE, kernel_open, iterations=1)
            combined = cv2.bitwise_and(combined, cv2.bitwise_not(spec_mask))

        # remove small speckles then close gaps (use smaller open kernel to preserve tiny objects)
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

            # compute centroid early so we can apply ROI and skin checks
            M = cv2.moments(cnt)
            if M.get('m00', 0) == 0:
                continue
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            # skip detections outside requested central ROI
            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                continue

            # size-adaptive relaxation: for very small contours relax shape thresholds
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

            # (no skin/hand exclusion here — use dedicated hand detector later)

            rx, ry = pixel_to_robot(cx, cy, H)
            candidates.append({'robot': (rx, ry), 'pixel': (cx, cy), 'area': area, 'contour': cnt,
                                'solidity': solidity, 'extent': extent, 'circularity': circularity,
                                'bbox': (x, y, wbox, hbox), 'aspect': aspect})

        # Simple centroid-based tracking to require stability across frames
        # Match candidates to existing tracks by proximity and maintain per-track locks
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
                # update track properties
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

                # Per-track lock: only start counting after short-term stability
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
                # create new track (include robot coords)
                tracks.append({'id': next_track_id, 'centroid': (cx, cy), 'stability': 1,
                               'last_seen': frame_idx, 'contour': cand['contour'], 'area': cand['area'],
                               'solidity': cand['solidity'], 'extent': cand['extent'], 'circularity': cand['circularity'],
                               'bbox': cand['bbox'], 'pixel': cand['pixel'], 'robot': cand.get('robot'),
                               'lock_counter': 0, 'lock_centroid': (cx, cy), 'locked': False})
                next_track_id += 1

        # prune stale tracks (not seen for a short while)
        new_tracks = []
        for t in tracks:
            if frame_idx - t.get('last_seen', 0) <= max(2, args.stability_frames * 2):
                new_tracks.append(t)
        tracks = new_tracks

        # stable tracks to display
        stable_tracks = [t for t in tracks if t['stability'] >= args.stability_frames]

        if args.debug:
            cand_areas = [int(c['area']) for c in candidates]
            cand_solids = [f"{c['solidity']:.2f}" for c in candidates]
            stable_areas = [int(t['area']) for t in stable_tracks]
            stable_solids = [f"{t['solidity']:.2f}" for t in stable_tracks]
            print(f"Raw contours: {raw_count}  Candidates: {len(candidates)}  Stable: {len(stable_tracks)}  CandidateAreas: {cand_areas}  StableAreas: {stable_areas}  StableSolidity: {stable_solids}")

        for i, obj in enumerate(stable_tracks):
            cnt = obj['contour']
            cx, cy = obj['pixel']
            area = int(obj['area'])
            solidity = obj.get('solidity', 0.0)
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
            cv2.drawContours(display, [cnt], -1, contour_color, 2 if locked else 1)
            cv2.rectangle(display, (x, y), (x + wbox, y + hbox), rect_color, 2 if locked else 1)
            cv2.putText(display, f"{i+1}", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(display, f"A:{area}", (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display, f"S:{solidity:.2f}", (cx + 10, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            if locked:
                cv2.putText(display, "LOCKED", (cx + 10, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Lock:{lock_progress}%", (cx + 10, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        status = f"BG: {'yes' if bg_captured else 'no (press B)'}  Detected: {len(stable_tracks)}  (E=exit)"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # show component windows if requested
        if args.debug_windows:
            cv2.imshow('CLAHE' if not args.no_clahe else 'Gray', clahe_img)
            cv2.imshow('Thresh', thresh)
            cv2.imshow('Red', red_mask)
            cv2.imshow('Edges', edges)
            if bg_mask is not None:
                cv2.imshow('BG Mask', bg_mask)
        if args.show_mask:
            cv2.imshow('Mask', combined)
        cv2.imshow('Detection', display)
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
                # Apply CLAHE to the stored background only when CLAHE is enabled
                if not args.no_clahe:
                    bg_gray = clahe.apply(bg_gray)
                # blur background to reduce speckle noise
                bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
                bg_captured = True
                # clear existing tracks so old detections don't persist
                tracks = []
                next_track_id = 1
                print('Background captured and tracks cleared')
            continue

        if key == ord('e'):
            break

        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(stable_tracks):
                sel = stable_tracks[idx]
                if not sel.get('locked', False):
                    print(f"Selection #{idx+1} ignored: object not locked yet ({sel.get('lock_counter',0)}/{args.stability_lock_frames})")
                else:
                    rx, ry = sel['robot']
                    print(f"Selected #{idx+1}: pixel={sel['pixel']} area={sel['area']:.0f} -> robot=({rx:.1f}, {ry:.1f})")
                    # Optionally save selection to file for automation
                    if args.save_selected:
                        out = {
                            'selected_index': idx + 1,
                            'pixel': [int(sel['pixel'][0]), int(sel['pixel'][1])],
                            'robot': [float(sel['robot'][0]), float(sel['robot'][1])],
                            'area': float(sel.get('area', 0)),
                            'bbox': sel.get('bbox')
                        }
                        mode = 'a' if args.append_selected else 'w'
                        try:
                            with open(args.save_selected, mode, encoding='utf8') as f:
                                if args.append_selected:
                                    f.write(json.dumps(out) + '\n')
                                else:
                                    json.dump(out, f, indent=2)
                            print(f"Saved selection to {args.save_selected}")
                        except Exception as e:
                            print(f"Failed to save selection: {e}")

                    # For testing we just print the robot coordinate; collaborator can copy this into pick script
                    break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
