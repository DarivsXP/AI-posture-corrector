# calibrate_posture.py

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

def get_keypoints(results, landmark_list):
    landmarks = results.pose_landmarks.landmark
    return [(landmarks[i].x, landmarks[i].y) for i in landmark_list]

def main():
    cap = cv2.VideoCapture(0)
    print("\n🪑 Sit straight in your best posture. Calibration starts in 5 seconds...")
    time.sleep(5)
    print("📷 Capturing posture baseline...")

    angle_data = {"neck": [], "back": [], "shoulder_alignment": []}

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        frame_count = 0
        while frame_count < 100:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # key points
                l_shoulder = [landmarks[11].x, landmarks[11].y]
                r_shoulder = [landmarks[12].x, landmarks[12].y]
                l_hip = [landmarks[23].x, landmarks[23].y]
                r_hip = [landmarks[24].x, landmarks[24].y]
                l_ear = [landmarks[7].x, landmarks[7].y]
                r_ear = [landmarks[8].x, landmarks[8].y]

                mid_shoulder = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
                mid_hip = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
                mid_ear = [(l_ear[0]+r_ear[0])/2, (l_ear[1]+r_ear[1])/2]

                # compute angles
                neck_angle = calculate_angle(mid_ear, mid_shoulder, mid_hip)
                back_angle = calculate_angle(mid_shoulder, mid_hip, [mid_hip[0], mid_hip[1]+0.1])
                shoulder_alignment = np.abs(l_shoulder[1] - r_shoulder[1]) * 100

                angle_data["neck"].append(neck_angle)
                angle_data["back"].append(back_angle)
                angle_data["shoulder_alignment"].append(shoulder_alignment)

                frame_count += 1

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(image, f"Capturing... {frame_count}/100", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Calibration", image)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    baseline = {k: np.mean(v) for k, v in angle_data.items()}
    with open("posture_baseline.json", "w") as f:
        json.dump(baseline, f, indent=4)

    print("\n✅ Calibration complete! Baseline saved to posture_baseline.json")
    print("Baseline angles:", baseline)

if __name__ == "__main__":
    main()
