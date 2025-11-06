# monitor_posture.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from posture_utils import get_pose_landmarks, calculate_angles, mp_pose

BASELINE_FILE = "posture_baseline.json"

# --- Scoring Parameters ---
# We now have 3 thresholds, one for each angle.
# This makes the system tunable.
ZERO_SCORE_THRESHOLD = {
    "torso_recline": 15.0,  
    "neck_protraction": 15.0,
    "back_curve": 20.0 # Make this one a bit more sensitive
}

def calculate_score(current, baseline, threshold):
    deviation = baseline - current
    if deviation <= 0: return 100 # Straighter than baseline
    score = 100 * (1 - (deviation / threshold))
    return int(max(0, min(100, score)))

def get_score_color(score):
    if score >= 90: return (0, 255, 0)  # Green
    elif score >= 70: return (0, 255, 255) # Yellow
    else: return (0, 0, 255)  # Red

def run_monitoring(baseline_path=BASELINE_FILE):
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file not found: {baseline_path}"); return

    with open(baseline_path, "r") as f:
        baseline = json.load(f)
    
    print("--- Posture Monitoring Started (3-Angle Logic) ---")
    print(f"Loaded baseline: {baseline}")
    print("Press ESC to quit.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): print("Error: Camera not detected."); return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()
            h, w, _ = frame.shape
            
            overall_score = 0
            feedback_color = (0, 0, 255)

            if results.pose_landmarks:
                landmarks_3d, landmarks_2d, visible = get_pose_landmarks(results, w, h)

                if visible:
                    angles = calculate_angles(landmarks_3d)
                    
                    # --- Calculate 3 Scores ---
                    torso_score = calculate_score(angles['torso_recline'], baseline['torso_recline'], ZERO_SCORE_THRESHOLD['torso_recline'])
                    neck_score = calculate_score(angles['neck_protraction'], baseline['neck_protraction'], ZERO_SCORE_THRESHOLD['neck_protraction'])
                    back_score = calculate_score(angles['back_curve'], baseline['back_curve'], ZERO_SCORE_THRESHOLD['back_curve'])
                    
                    # Overall score is now an average of all three
                    overall_score = int((torso_score + neck_score + back_score) / 3)
                    feedback_color = get_score_color(overall_score)
                    
                    # --- Draw Lines & Dots ---
                    cv2.line(display, landmarks_2d['hip_px'], landmarks_2d['shoulder_px'], feedback_color, 2)
                    cv2.line(display, landmarks_2d['shoulder_px'], landmarks_2d['ear_px'], feedback_color, 2)
                    
                    cv2.circle(display, landmarks_2d['hip_px'], 5, (0, 0, 255), -1)
                    cv2.circle(display, landmarks_2d['shoulder_px'], 5, (0, 0, 255), -1)
                    cv2.circle(display, landmarks_2d['ear_px'], 5, (0, 0, 255), -1)

                    # --- Display All 3 Scores ---
                    cv2.putText(display, f"Torso Score: {torso_score}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, f"Neck Score:  {neck_score}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, f"Back Score:  {back_score}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    overall_score = 0
                    feedback_color = (0, 0, 255)
            
            # --- Big Overall Score Text ---
            cv2.putText(display, f"Score: {overall_score}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, feedback_color, 3)

            cv2.imshow("Posture Monitor", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped.")

if __name__ == "__main__":
    run_monitoring()