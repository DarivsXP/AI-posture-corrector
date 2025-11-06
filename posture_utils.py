# posture_utils.py
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def calculate_3d_angle(a, b, c):
    """Calculates the 3D angle a-b-c."""
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


def get_pose_landmarks(results, w, h):
    """
    Extracts key landmarks (hip, shoulder, ear) for the most visible side.
    """
    landmarks_3d_dict = {}
    landmarks_2d_dict = {}
    
    lm = results.pose_landmarks.landmark
    
    use_left_side = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > \
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
    
    if use_left_side:
        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    else:
        shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]

    visible = (shoulder.visibility > 0.5 and 
               hip.visibility > 0.5 and 
               ear.visibility > 0.5)

    if not visible:
        return None, None, False

    # --- Store 3D coordinates (x, y, z) for angle calculation ---
    landmarks_3d_dict['shoulder'] = [shoulder.x, shoulder.y, shoulder.z]
    landmarks_3d_dict['hip'] = [hip.x, hip.y, hip.z]
    landmarks_3d_dict['ear'] = [ear.x, ear.y, ear.z]

    # --- Store 2D pixel coordinates for drawing ---
    landmarks_2d_dict['shoulder_px'] = (int(shoulder.x * w), int(shoulder.y * h))
    landmarks_2d_dict['hip_px'] = (int(hip.x * w), int(hip.y * h))
    landmarks_2d_dict['ear_px'] = (int(ear.x * w), int(ear.y * h))
    
    return landmarks_3d_dict, landmarks_2d_dict, True


def calculate_angles(landmarks_3d):
    """
    Calculates all three key posture angles.
    """
    hip = landmarks_3d['hip']
    shoulder = landmarks_3d['shoulder']
    ear = landmarks_3d['ear']

    # --- Create Vertical "Helper" Points ---
    hip_vertical = [hip[0], hip[1] - 1, hip[2]]
    shoulder_vertical = [shoulder[0], shoulder[1] - 1, shoulder[2]]

    # --- Angle 1: Torso Recline (Vertical_Line - Hip - Shoulder) ---
    torso_recline_angle = calculate_3d_angle(hip_vertical, hip, shoulder)
    
    # --- Angle 2: Neck Protraction (Vertical_Line - Shoulder - Ear) ---
    neck_protraction_angle = calculate_3d_angle(shoulder_vertical, shoulder, ear)
    
    # --- Angle 3: Back Curve (Hip - Shoulder - Ear) ---
    back_curve_angle = calculate_3d_angle(hip, shoulder, ear)

    return {
        "torso_recline": torso_recline_angle,
        "neck_protraction": neck_protraction_angle,
        "back_curve": back_curve_angle
    }