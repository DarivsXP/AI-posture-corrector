# calibrate_posture.py
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from collections import deque
from posture_utils import get_pose_landmarks, calculate_angles, mp_pose

# --- Calibration Parameters ---
STABILITY_WINDOW = 30
STABILITY_THRESHOLD = 2.5
STABLE_FRAMES_REQUIRED = 75 
SAVE_PATH = "posture_baseline.json"

def is_point_in_ellipse(point, center, axes):
    px, py = point
    cx, cy = center
    ax, ay = axes
    if ax == 0 or ay == 0: return False
    return ((px - cx) ** 2 / ax ** 2) + ((py - cy) ** 2 / ay ** 2) <= 1

def run_calibration(save_path=SAVE_PATH):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Camera not detected."); return

    print("--- Posture Calibration (3-Angle) ---")
    print("1. Turn your chair 45-90 degrees (side profile).")
    print("2. Fit your hip, shoulder, and ear inside the oval.")
    print("3. Sit in YOUR BEST, straightest posture.")
    print("4. Hold still for ~3 seconds.")
    print("Press ESC to abort.")

    # Deques for all 3 angles
    angle_history = {
        "torso_recline": deque(maxlen=STABILITY_WINDOW),
        "neck_protraction": deque(maxlen=STABILITY_WINDOW),
        "back_curve": deque(maxlen=STABILITY_WINDOW)
    }
    
    stable_frames_count = 0
    calibration_data = [] 
    is_in_position = False 
    
    jitter_text = ""

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()
            h, w, _ = frame.shape
            
            # --- 1. Draw Guide Oval ---
            oval_center = (w // 2, h // 2)
            oval_axes = (int(w * 0.25), int(h * 0.45)) 
            cv2.ellipse(display, oval_center, oval_axes, 0, 0, 360, (150, 150, 150), 2)
            
            status_text = "GET IN POSITION"
            status_color = (0, 255, 255) # Yellow

            if results.pose_landmarks:
                landmarks_3d, landmarks_2d, visible = get_pose_landmarks(results, w, h)
                
                if visible:
                    # --- 2. Draw User Landmarks ---
                    cv2.line(display, landmarks_2d['hip_px'], landmarks_2d['shoulder_px'], (0,0,255), 2)
                    cv2.line(display, landmarks_2d['shoulder_px'], landmarks_2d['ear_px'], (0,0,255), 2)
                    cv2.circle(display, landmarks_2d['hip_px'], 5, (0,0,255), -1)
                    cv2.circle(display, landmarks_2d['shoulder_px'], 5, (0,0,255), -1)
                    cv2.circle(display, landmarks_2d['ear_px'], 5, (0,0,255), -1)

                    # --- 3. Check Position ---
                    hip_in = is_point_in_ellipse(landmarks_2d['hip_px'], oval_center, oval_axes)
                    shoulder_in = is_point_in_ellipse(landmarks_2d['shoulder_px'], oval_center, oval_axes)
                    ear_in = is_point_in_ellipse(landmarks_2d['ear_px'], oval_center, oval_axes)

                    if hip_in and shoulder_in and ear_in:
                        is_in_position = True
                    else:
                        is_in_position = False; stable_frames_count = 0; calibration_data = []

                    # --- 4. Check Stability ---
                    if is_in_position:
                        angles = calculate_angles(landmarks_3d)
                        for key in angle_history:
                            angle_history[key].append(angles[key])
                        
                        if len(angle_history["torso_recline"]) == STABILITY_WINDOW:
                            # Check jitter for all 3 angles
                            std_devs = {key: np.std(angle_history[key]) for key in angle_history}
                            is_stable = all(std_devs[key] < STABILITY_THRESHOLD for key in angle_history)
                            
                            jitter_text = f"Jitter: T:{std_devs['torso_recline']:.1f} N:{std_devs['neck_protraction']:.1f} B:{std_devs['back_curve']:.1f}"

                            if is_stable:
                                status_text = "HOLDING STILL..."
                                status_color = (0, 255, 0)
                                stable_frames_count += 1
                                calibration_data.append(angles)
                            else:
                                status_text = "HOLD STILL (Jitter too high)"
                                status_color = (0, 165, 255)
                                stable_frames_count = 0; calibration_data = []
                else:
                    status_text = "NOT VISIBLE"; is_in_position = False
            else:
                status_text = "NO PERSON"; is_in_position = False

            # --- Display Status ---
            cv2.putText(display, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            progress = (stable_frames_count / STABLE_FRAMES_REQUIRED) * 100
            cv2.rectangle(display, (10, 60), (int(w * 0.4), 85), (80, 80, 80), -1)
            cv2.rectangle(display, (10, 60), (int((w * 0.4) * (progress / 100)), 85), (0, 255, 0), -1)
            cv2.putText(display, f"{progress:.0f}%", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if is_in_position:
                cv2.putText(display, jitter_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: print("Calibration aborted."); break
            if stable_frames_count >= STABLE_FRAMES_REQUIRED: print("Stability achieved!"); break

    cap.release(); cv2.destroyAllWindows()
    if len(calibration_data) < STABLE_FRAMES_REQUIRED:
        print("Calibration failed."); return

    # --- Save Baseline for all 3 angles ---
    baseline = {key: np.mean([d[key] for d in calibration_data]) for key in angle_history}

    with open(save_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print("--- Calibration Complete ---")
    print(f"Baseline saved to: {os.path.abspath(save_path)}")
    print(f"Baseline: {baseline}")

if __name__ == "__main__":
    run_calibration()