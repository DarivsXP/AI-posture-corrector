import cv2
import mediapipe as mp
import time
import requests
import posture_utils

# --- 1. CONFIGURATION ---
# Your API Token (Verify this is current from your 'php artisan tinker' output)
API_TOKEN = "1|Yl6UB5B00KthxdbtWVMMVQqTYDCRh0CB8GKVQoXA9e13764a"

# Your Laravel API URL
API_URL = "http://127.0.0.1:8000/api/posture-chunks"

# Data Save Interval (Seconds) - Set to 300 for 5 mins in production
SAVE_INTERVAL = 10

# --- 2. POSTURE RULES (The "Ideal" Logic) ---
# These ranges define 100% score. Outside these = penalty.
# Based on the new "Vertical Deviation" math in posture_utils.py
POSTURE_RULES = {
    # Torso: 0 is vertical. 0-20 degrees is a healthy recline.
    "torso": {"min": 0, "max": 20, "name": "Torso"},

    # Neck: 0 is vertical. 0-30 degrees is acceptable flexion.
    "neck":  {"min": 0, "max": 30, "name": "Neck"},

    # Back: 180 is straight. 140-180 is acceptable. Below 140 is a hunch.
    "back":  {"min": 140, "max": 180, "name": "Back"}
}

# --- 3. HELPER CLASSES ---

class SessionData:
    """ Tracks stats for the current 5-minute chunk """
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = []
        self.slouch_frames = 0
        self.total_frames = 0
        self.alert_count = 0
        self.start_time = time.time()

    def update(self, score, is_slouching, alert_triggered):
        self.scores.append(score)
        self.total_frames += 1
        if is_slouching: self.slouch_frames += 1
        if alert_triggered: self.alert_count += 1

    def time_to_upload(self):
        return (time.time() - self.start_time) > SAVE_INTERVAL

    def get_payload(self):
        # Calculate average score
        avg = int(sum(self.scores) / len(self.scores)) if self.scores else 100

        # Calculate duration in seconds
        duration = time.time() - self.start_time
        ratio = self.slouch_frames / self.total_frames if self.total_frames else 0
        slouch_time = int(duration * ratio)

        return {
            "score": avg,
            "slouch_duration": slouch_time,
            "alert_count": self.alert_count
        }

class FeedbackSystem:
    """ Manages the adaptive alerts (Visual -> Audio -> Notification) """
    def __init__(self):
        self.slouch_start = None
        self.status = "Good"

    def process(self, is_slouching):
        alert_now = False
        msg = "Good Posture"

        if not is_slouching:
            self.slouch_start = None
            self.status = "Good"
            return msg, False

        if self.slouch_start is None:
            self.slouch_start = time.time()

        duration = time.time() - self.slouch_start

        if duration > 15:
            self.status = "Critical"
            msg = "CRITICAL: Sit Up!"
            # Only trigger count once per "event" or every few seconds
            if int(duration) % 5 == 0:
                alert_now = True
        elif duration > 5:
            self.status = "Warning"
            msg = "Warning..."

        return msg, alert_now

def calculate_frame_score(angles):
    """ Calculates 0-100 score based on deviation from rules """
    # If angles are missing, return 0 or maintain previous
    if angles["torso"] is None: return 0

    score = 100
    penalty_weight = 2.0 # How harsh the penalty is per degree

    # 1. Torso Penalty (If leaning back too far)
    if angles["torso"] > POSTURE_RULES["torso"]["max"]:
        diff = angles["torso"] - POSTURE_RULES["torso"]["max"]
        score -= diff * penalty_weight

    # 2. Neck Penalty (If leaning forward too much)
    if angles["neck"] > POSTURE_RULES["neck"]["max"]:
        diff = angles["neck"] - POSTURE_RULES["neck"]["max"]
        score -= diff * penalty_weight

    # 3. Back Penalty (If curling forward)
    if angles["back"] < POSTURE_RULES["back"]["min"]:
        diff = POSTURE_RULES["back"]["min"] - angles["back"]
        score -= diff * penalty_weight

    return int(max(0, min(100, score)))

# --- 4. MAIN LOOP ---
def main():
    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    feedback = FeedbackSystem()
    session = SessionData()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found.")
        return

    print(f"--- Monitor Running (Upload every {SAVE_INTERVAL}s) ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        # Process Image
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference
        frame.flags.writeable = False
        results = pose.process(rgb)
        frame.flags.writeable = True

        # Recolor back to BGR for display
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        score = 100
        is_slouching = False
        alert_triggered = False
        msg = "Detecting..."

        if results.pose_landmarks:
            # A. Get Angles
            lm = results.pose_landmarks.landmark
            angles = posture_utils.get_posture_angles(lm)

            if angles["torso"] is not None:
                # B. Score
                score = calculate_frame_score(angles)
                is_slouching = score < 70

                # C. Feedback
                msg, alert_triggered = feedback.process(is_slouching)

                # D. Debug Display (So you can see what's happening)
                # This prints: T=Torso Angle, N=Neck Angle, B=Back Angle
                debug_str = f"T:{int(angles['torso'])} N:{int(angles['neck'])} B:{int(angles['back'])}"
                cv2.putText(frame, debug_str, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # E. Simple Lines (Hip-Shoulder-Ear)
                try:
                    # Hardcoded indices for Left side visualization
                    # Shoulder(11), Ear(7), Hip(23)
                    p_sh = (int(lm[11].x * w), int(lm[11].y * h))
                    p_ear = (int(lm[7].x * w), int(lm[7].y * h))
                    p_hip = (int(lm[23].x * w), int(lm[23].y * h))

                    line_color = (0, 0, 255) if is_slouching else (0, 255, 0)
                    cv2.line(frame, p_ear, p_sh, line_color, 3)
                    cv2.line(frame, p_sh, p_hip, line_color, 3)
                    cv2.circle(frame, p_ear, 5, (255, 255, 255), -1)
                except: pass

        # Status Display
        status_color = (0, 0, 255) if is_slouching else (0, 255, 0)
        cv2.putText(frame, f"SCORE: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Update Session Data
        session.update(score, is_slouching, alert_triggered)

        # Upload to Laravel
        if session.time_to_upload():
            print("[UPLOADING]...", end=" ")
            try:
                payload = session.get_payload()
                res = requests.post(API_URL, json=payload, headers={
                    "Authorization": f"Bearer {API_TOKEN}",
                    "Accept": "application/json"
                })
                if res.status_code == 201:
                    print("[OK]")
                else:
                    print(f"[FAIL] {res.status_code} - {res.text}")
            except Exception as e:
                print(f"[ERR] {e}")
            session.reset()

        cv2.imshow('Posture Monitor', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
