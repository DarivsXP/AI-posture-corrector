# monitor_posture.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from posture_utils import get_pose_landmarks, calculate_angles # Import new functions

mp_pose = mp.solutions.pose
BASELINE_FILE = "posture_baseline.json"

# --- Scoring Parameters ---
# An angle deviation greater than this will result in a 0% score.
ZERO_SCORE_THRESHOLD = {
    "neck": 20.0,  # 20 degrees of deviation = 0%
    "torso": 15.0  # 15 degrees of recline deviation = 0%
}

def calculate_score(current, baseline, threshold):
    """Calculates a 0-100 score based on deviation."""
    deviation = abs(current - baseline)
    score = 100 * (1 - (deviation / threshold))
    return int(max(0, min(100, score)))

def get_score_color(score):
    """Returns BGR color based on score."""
    if score >= 90:
        return (0, 255, 0)  # Green
    elif score >= 70:
        return (0, 255, 255) # Yellow
    else:
        return (0, 0, 255)  # Red

def run_monitoring(baseline_path=BASELINE_FILE):
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file not found at {baseline_path}")
        print("Please run calibrate_posture.py first.")
        return

    with open(baseline_path, "r") as f:
        baseline = json.load(f)
    
    print("--- Posture Monitoring Started ---")
    print(f"Loaded baseline: Neck={baseline['neck']:.1f}°, Torso={baseline['torso']:.1f}°")
    print("Press ESC to quit.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Camera not detected.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()
            h, w, _ = frame.shape
            
            overall_score = 0
            feedback_color = (0, 0, 255)

            if results.pose_landmarks:
                landmarks_2d, visible = get_pose_landmarks(results, w, h)

                if visible:
                    angles = calculate_angles(landmarks_2d)
                    
                    # --- Calculate Scores ---
                    neck_score = calculate_score(angles['neck'], baseline['neck'], ZERO_SCORE_THRESHOLD['neck'])
                    torso_score = calculate_score(angles['torso'], baseline['torso'], ZERO_SCORE_THRESHOLD['torso'])
                    overall_score = int((neck_score + torso_score) / 2)
                    feedback_color = get_score_color(overall_score)
                    
                    # --- Draw Lines & Dots ---
                    cv2.line(display, landmarks_2d['hip_mid_px'], landmarks_2d['shoulder_mid_px'], feedback_color, 2)
                    cv2.line(display, landmarks_2d['shoulder_mid_px'], landmarks_2d['ear_mid_px'], feedback_color, 2)
                    cv2.circle(display, landmarks_2d['hip_mid_px'], 5, (0, 0, 255), -1)
                    cv2.circle(display, landmarks_2d['shoulder_mid_px'], 5, (0, 0, 255), -1)
                    cv2.circle(display, landmarks_2d['ear_mid_px'], 5, (0, 0, 255), -1)

                    # --- Display Angles & Scores ---
                    cv2.putText(display, f"Neck Score: {neck_score}% (Angle: {angles['neck']:.1f})", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, f"Torso Score: {torso_score}% (Angle: {angles['torso']:.1f})", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    overall_score = 0
                    feedback_color = (0, 0, 255)
            
            # --- Big Overall Score Text ---
            cv2.putText(display, f"Overall Score: {overall_score}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, feedback_color, 3)

            cv2.imshow("Posture Monitor", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped.")

if __name__ == "__main__":
    run_monitoring()