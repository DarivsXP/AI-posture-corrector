# monitor_posture.py
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# thresholds / params (tune as needed)
BAD_DEGREE_THRESHOLD = 12.0    # degrees deviation considered meaningful (per-angle)
GOOD_SCORE_THRESHOLD = 88.0    # score above which posture is considered "very good"
GOOD_FRAMES_FOR_ADAPT = 90     # number of consecutive good frames (~3 seconds at 30fps)
ADAPT_LR = 0.08                # adaptive update rate (0.08 = 8% new)
ALERT_TIME = 5.0               # seconds bad posture must persist before alert

def calculate_3d_angle(a, b, c):
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

def load_baseline(path="posture_baseline.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found. Run calibration first and create {path}")
    with open(path, "r") as f:
        baseline = json.load(f)
    # ensure floats
    return {k: float(v) for k, v in baseline.items()}

def compute_score(current, baseline):
    # weights: neck is more important than upper_back (tweakable)
    w = {"neck": 0.6, "upper_back": 0.4}
    score = 100.0
    # penalty scaled and clipped
    for key in ["neck", "upper_back"]:
        diff = abs(current[key] - baseline[key])
        clipped = min(diff, 30.0)  # cap effect of large outliers
        # convert to penalty points: larger diff → bigger penalty
        # neck has stronger penalty
        score -= clipped * (w[key] * 1.5)
    score = max(0.0, min(100.0, score))
    return score

def monitor(baseline_path="posture_baseline.json"):
    baseline = load_baseline(baseline_path)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not detected. Check webcam permissions.")
        return

    print("Starting monitor. Press ESC to quit.")
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:

        good_frames = 0
        bad_since = None
        last_adapt_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read; stopping.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                nose = [lm[0].x, lm[0].y, lm[0].z]
                left_ear = [lm[7].x, lm[7].y, lm[7].z]
                left_shoulder = [lm[11].x, lm[11].y, lm[11].z]
                left_hip = [lm[23].x, lm[23].y, lm[23].z]

                # compute 3D angles
                neck = calculate_3d_angle(left_shoulder, left_ear, nose)          # shoulder-ear-nose
                upper_back = calculate_3d_angle(left_hip, left_shoulder, left_ear)  # hip-shoulder-ear

                current = {"neck": neck, "upper_back": upper_back}
                score = compute_score(current, baseline)

                # adaptive logic
                if score >= GOOD_SCORE_THRESHOLD:
                    good_frames += 1
                else:
                    good_frames = 0

                # if very good posture for long enough, slightly adapt baseline
                if good_frames >= GOOD_FRAMES_FOR_ADAPT:
                    for k in baseline:
                        baseline[k] = baseline[k] * (1.0 - ADAPT_LR) + current[k] * ADAPT_LR
                    # save baseline
                    with open(baseline_path, "w") as f:
                        json.dump(baseline, f, indent=2)
                    print("Adaptive baseline updated:", {k: round(baseline[k],2) for k in baseline})
                    good_frames = 0

                # detect sustained bad posture
                # determine if any angle deviates more than BAD_DEGREE_THRESHOLD
                bad_now = any(abs(current[k] - baseline[k]) > BAD_DEGREE_THRESHOLD for k in baseline)
                if bad_now:
                    if bad_since is None:
                        bad_since = time.time()
                    elapsed = time.time() - bad_since
                else:
                    bad_since = None
                    elapsed = 0.0

                # overlay
                mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                color = (0,255,0) if score >= 80 else ((0,200,200) if score>=60 else (0,0,255))
                cv2.putText(display, f"Score: {score:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(display, f"Neck: {int(current['neck'])}  Back: {int(current['upper_back'])}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

                # show alert if bad posture persisted longer than ALERT_TIME
                if bad_since and elapsed > ALERT_TIME:
                    cv2.putText(display, "⚠️ Please sit up! (bad posture detected)", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                elif bad_since:
                    cv2.putText(display, f"Adjusting... {int(elapsed)}s", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

            else:
                cv2.putText(display, "No person detected. Reposition camera.", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Posture Monitor (Adaptive)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped.")

if __name__ == "__main__":
    monitor()
