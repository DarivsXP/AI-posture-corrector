# monitor_posture.py

import cv2
import mediapipe as mp
import numpy as np
import json
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# load calibration data
with open("posture_baseline.json") as f:
    baseline = json.load(f)

cap = cv2.VideoCapture(0)
bad_posture_start = None
ALERT_THRESHOLD = 15  # degrees
TIME_THRESHOLD = 5    # seconds

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_shoulder = [landmarks[11].x, landmarks[11].y]
            r_shoulder = [landmarks[12].x, landmarks[12].y]
            l_hip = [landmarks[23].x, landmarks[23].y]
            r_hip = [landmarks[24].x, landmarks[24].y]
            l_ear = [landmarks[7].x, landmarks[7].y]
            r_ear = [landmarks[8].x, landmarks[8].y]

            mid_shoulder = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
            mid_hip = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
            mid_ear = [(l_ear[0]+r_ear[0])/2, (l_ear[1]+r_ear[1])/2]

            neck_angle = calculate_angle(mid_ear, mid_shoulder, mid_hip)
            back_angle = calculate_angle(mid_shoulder, mid_hip, [mid_hip[0], mid_hip[1]+0.1])
            shoulder_alignment = np.abs(l_shoulder[1] - r_shoulder[1]) * 100

            # Compare with baseline
            deviation = {
                "neck": abs(neck_angle - baseline["neck"]),
                "back": abs(back_angle - baseline["back"]),
                "shoulder": abs(shoulder_alignment - baseline["shoulder_alignment"])
            }

            bad_posture = any(dev > ALERT_THRESHOLD for dev in deviation.values())

            if bad_posture:
                if bad_posture_start is None:
                    bad_posture_start = time.time()
                elif time.time() - bad_posture_start > TIME_THRESHOLD:
                    status = "⚠️ Bad Posture Detected!"
                    color = (0, 0, 255)
                else:
                    status = "Adjusting..."
                    color = (0, 255, 255)
            else:
                bad_posture_start = None
                status = "✅ Good Posture"
                color = (0, 255, 0)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Posture Monitor", image)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
