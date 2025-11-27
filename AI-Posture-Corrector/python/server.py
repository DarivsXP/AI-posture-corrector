from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import posture_utils
import traceback # <--- Added to print exact errors

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
IDEAL_ANGLES = {
    "torso_recline": (160, 170),
    "neck_protraction": (160, 180),
    "back_curve": (145, 160)
}
ZERO_SCORE_THRESHOLD = 12.0

# Initialize MediaPipe
# Set static_image_mode=True since we process one independent frame at a time via HTTP
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

def calculate_score(val, ideal, thresh):
    if val is None: return 100
    if val < ideal[0]: dev = ideal[0] - val
    elif val > ideal[1]: dev = val - ideal[1]
    else: return 100
    return int(max(0, 100 * (1 - (dev / thresh))))

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'image' not in request.files:
            print("[ERROR] No image file in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        # Read and decode image
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            print("[ERROR] Failed to decode image")
            return jsonify({"error": "Image decode failed"}), 400

        # Run MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            # Return a neutral response if no person is found
            return jsonify({
                "score": 0,
                "is_slouching": False,
                "status": "No Person Detected",
                "angles": {"torso": 0, "neck": 0, "back": 0}
            })

        # Calculate Angles
        angles = posture_utils.get_posture_angles(results.pose_landmarks.landmark)

        # Calculate Score
        s1 = calculate_score(angles["torso_recline"], IDEAL_ANGLES["torso_recline"], ZERO_SCORE_THRESHOLD)
        s2 = calculate_score(angles["neck_protraction"], IDEAL_ANGLES["neck_protraction"], ZERO_SCORE_THRESHOLD)
        s3 = calculate_score(angles["back_curve"], IDEAL_ANGLES["back_curve"], ZERO_SCORE_THRESHOLD)

        avg_score = int((s1 + s2 + s3) / 3)

        response = {
            "score": avg_score,
            "is_slouching": avg_score < 70,
            "angles": {
                "torso": int(angles["torso_recline"] or 0),
                "neck": int(angles["neck_protraction"] or 0),
                "back": int(angles["back_curve"] or 0)
            }
        }
        return jsonify(response)

    except Exception as e:
        # PRINT THE ACTUAL ERROR TO THE TERMINAL
        print("----------------------------------------------------")
        print("[CRASH] Server Error:")
        traceback.print_exc() # This prints the full error details
        print("----------------------------------------------------")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("--- Python AI Server Running on Port 5000 ---")
    app.run(debug=True, port=5000)
