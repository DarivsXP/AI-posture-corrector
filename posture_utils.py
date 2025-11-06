# posture_utils.py
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculates the angle a-b-c.
    Works for 2D and 3D points.
    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
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
    Extracts key landmarks and calculates midpoints.
    Returns a dictionary of 2D points and a visibility flag.
    """
    landmarks_3d = results.pose_landmarks.landmark
    landmarks_2d = {}
    visible = True

    # Get key 3D landmarks
    left_shoulder_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ear_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_EAR.value]

    # Check visibility
    key_landmarks_visible = (
        left_shoulder_3d.visibility > 0.5 and right_shoulder_3d.visibility > 0.5 and
        left_hip_3d.visibility > 0.5 and right_hip_3d.visibility > 0.5 and
        left_ear_3d.visibility > 0.5 and right_ear_3d.visibility > 0.5
    )
    
    if not key_landmarks_visible:
        return None, False

    # --- Calculate Midpoints in 2D (for angle calculation) ---
    # We use 2D points for screen-relative angles
    shoulder_mid_x = (left_shoulder_3d.x + right_shoulder_3d.x) / 2
    shoulder_mid_y = (left_shoulder_3d.y + right_shoulder_3d.y) / 2

    hip_mid_x = (left_hip_3d.x + right_hip_3d.x) / 2
    hip_mid_y = (left_hip_3d.y + right_hip_3d.y) / 2

    ear_mid_x = (left_ear_3d.x + right_ear_3d.x) / 2
    ear_mid_y = (left_ear_3d.y + right_ear_3d.y) / 2

    # Store 2D points (normalized 0.0-1.0)
    landmarks_2d['shoulder_mid'] = [shoulder_mid_x, shoulder_mid_y]
    landmarks_2d['hip_mid'] = [hip_mid_x, hip_mid_y]
    landmarks_2d['ear_mid'] = [ear_mid_x, ear_mid_y]

    # --- Store pixel coordinates for drawing ---
    landmarks_2d['shoulder_mid_px'] = (int(shoulder_mid_x * w), int(shoulder_mid_y * h))
    landmarks_2d['hip_mid_px'] = (int(hip_mid_x * w), int(hip_mid_y * h))
    landmarks_2d['ear_mid_px'] = (int(ear_mid_x * w), int(ear_mid_y * h))
    
    return landmarks_2d, True


def calculate_angles(landmarks_2d):
    """
    Calculates the new torso and neck angles.
    """
    hip_mid = landmarks_2d['hip_mid']
    shoulder_mid = landmarks_2d['shoulder_mid']
    ear_mid = landmarks_2d['ear_mid']

    # --- Torso Angle ---
    # Angle of the torso line relative to the vertical.
    # We create a new point 'hip_vertical' directly below 'hip_mid'
    hip_vertical = [hip_mid[0], hip_mid[1] + 1.0] 
    # Angle: hip_vertical - hip_mid - shoulder_mid
    torso_angle = calculate_angle(hip_vertical, hip_mid, shoulder_mid)
    
    # --- Neck Angle ---
    # Angle of the neck line relative to the torso line.
    # Angle: hip_mid - shoulder_mid - ear_mid
    neck_angle = calculate_angle(hip_mid, shoulder_mid, ear_mid)

    return {"torso": torso_angle, "neck": neck_angle}