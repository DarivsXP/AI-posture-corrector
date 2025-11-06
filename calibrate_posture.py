# calibrate_posture.py
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

N_CAPTURE = 120  # number of valid frames to average for baseline

def calculate_3d_angle(a, b, c):
    """3D angle between points a-b-c in degrees."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

def run_calibration(save_path="posture_baseline.json"):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not detected. Check the webcam and try again.")
        return

    print("Position camera at ~30-45° angle, chest/shoulder height.")
    print("Sit straight in your best posture. Calibration starts in 5 seconds...")
    time.sleep(5)
    print("Calibration started. Hold still. Press ESC to abort.")

    angles = {"neck": [], "upper_back": []}
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        valid_frames = 0
        total_frames = 0
        last_print = time.time()
        while valid_frames < N_CAPTURE:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            display = frame.copy()
            mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # choose LEFT side landmarks (works for 30-45 deg)
                nose = [lm[0].x, lm[0].y, lm[0].z]
                left_ear = [lm[7].x, lm[7].y, lm[7].z]
                left_shoulder = [lm[11].x, lm[11].y, lm[11].z]
                left_hip = [lm[23].x, lm[23].y, lm[23].z]

                neck_angle = calculate_3d_angle(left_shoulder, left_ear, nose)      # shoulder - ear - nose
                upper_back_angle = calculate_3d_angle(left_hip, left_shoulder, left_ear)  # hip - shoulder - ear

                # Collect only plausible angles (filter out zeros / bad detection)
                if 10 < neck_angle < 200 and 10 < upper_back_angle < 200:
                    angles["neck"].append(neck_angle)
                    angles["upper_back"].append(upper_back_angle)
                    valid_frames += 1

            # overlay status
            cv2.putText(display, f"Valid frames: {valid_frames}/{N_CAPTURE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
            cv2.putText(display, "Calibration (press ESC to cancel)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Calibration aborted by user.")
                break

            # occasional console update
            if time.time() - last_print > 2:
                print(f"Collected {valid_frames}/{N_CAPTURE} valid frames...")
                last_print = time.time()

        cap.release()
        cv2.destroyAllWindows()

    if len(angles["neck"]) < 20:
        print("Not enough valid frames collected. Try again with clearer view and lighting.")
        return

    baseline = {k: float(np.mean(v)) for k, v in angles.items()}
    with open(save_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print("Calibration complete. Baseline saved to:", os.path.abspath(save_path))
    print("Baseline:", baseline)

if __name__ == "__main__":
    run_calibration()
