# calibrate_posture.py
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from collections import deque
from posture_utils import get_pose_landmarks, calculate_angles # Import new functions

mp_pose = mp.solutions.pose

# --- Stability Parameters ---
STABILITY_WINDOW = 30      # frames (e.g., 1 second @ 30fps)
STABILITY_THRESHOLD = 0.5  # degrees (std dev must be below this)
STABLE_FRAMES_REQUIRED = 150 # frames (e.g., 5 seconds @ 30fps)
SAVE_PATH = "posture_baseline.json"

def run_calibration(save_path=SAVE_PATH):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Camera not detected. Check webcam connection.")
        return

    print("Sit straight in your BEST posture. Hold still.")
    print("Waiting for posture to stabilize...")
    print("Press ESC to abort.")

    # Deques to store recent angles for stability check
    neck_angles = deque(maxlen=STABILITY_WINDOW)
    torso_angles = deque(maxlen=STABILITY_WINDOW)
    
    stable_frames_count = 0
    calibration_data = [] # Store angles once stable

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()
            h, w, _ = frame.shape
            
            status_text = "HOLDING STILL..."
            status_color = (0, 255, 0) # Green

            if results.pose_landmarks:
                landmarks_2d, visible = get_pose_landmarks(results, w, h)
                
                if visible:
                    # --- Draw key points ---
                    cv2.line(display, landmarks_2d['hip_mid_px'], landmarks_2d['shoulder_mid_px'], (0, 255, 0), 2)
                    cv2.line(display, landmarks_2d['shoulder_mid_px'], landmarks_2d['ear_mid_px'], (0, 255, 0), 2)
                    cv2.circle(display, landmarks_2d['hip_mid_px'], 5, (0, 0, 255), -1)
                    cv2.circle(display, landmarks_2d['shoulder_mid_px'], 5, (0, 0, 255), -1)
                    cv2.circle(display, landmarks_2d['ear_mid_px'], 5, (0, 0, 255), -1)

                    angles = calculate_angles(landmarks_2d)
                    neck_angles.append(angles['neck'])
                    torso_angles.append(angles['torso'])
                    
                    # Check for stability
                    if len(neck_angles) == STABILITY_WINDOW:
                        neck_std_dev = np.std(neck_angles)
                        torso_std_dev = np.std(torso_angles)
                        
                        if neck_std_dev < STABILITY_THRESHOLD and torso_std_dev < STABILITY_THRESHOLD:
                            stable_frames_count += 1
                            # Store the stable angles
                            calibration_data.append(angles)
                        else:
                            stable_frames_count = 0 # Reset if moved
                            calibration_data = [] # Clear data
                            status_text = "HOLD STILL"
                            status_color = (0, 165, 255) # Orange

                else:
                    status_text = "NOT VISIBLE"
                    status_color = (0, 0, 255) # Red
                    stable_frames_count = 0
            else:
                status_text = "NO PERSON"
                status_color = (0, 0, 255) # Red
                stable_frames_count = 0

            # --- Display Status ---
            progress = (stable_frames_count / STABLE_FRAMES_REQUIRED) * 100
            cv2.putText(display, f"Calibrating: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display, f"Progress: {progress:.0f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Calibration aborted.")
                cap.release()
                cv2.destroyAllWindows()
                return

            if stable_frames_count >= STABLE_FRAMES_REQUIRED:
                print("Stability achieved!")
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(calibration_data) < STABLE_FRAMES_REQUIRED:
        print("Calibration failed or was aborted. Not enough stable frames.")
        return

    # --- Save Baseline ---
    baseline_neck = np.mean([d['neck'] for d in calibration_data])
    baseline_torso = np.mean([d['torso'] for d in calibration_data])
    baseline = {"neck": baseline_neck, "torso": baseline_torso}

    with open(save_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print("--- Calibration Complete ---")
    print(f"Baseline saved to: {os.path.abspath(save_path)}")
    print(f"Baseline Neck Angle: {baseline['neck']:.2f}°")
    print(f"Baseline Torso Angle: {baseline['torso']:.2f}°")

if __name__ == "__main__":
    run_calibration()