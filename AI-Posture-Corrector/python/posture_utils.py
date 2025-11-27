import numpy as np
import mediapipe as mp

def get_landmark(landmarks, part_name):
    """ Retrieves (x, y, z) coordinates if visibility > 0.5 """
    try:
        lm = landmarks[part_name.value]
        # Visibility check: If the camera can't see it well, ignore it
        if lm.visibility < 0.5: return None
        return np.array([lm.x, lm.y, lm.z])
    except:
        return None

def calculate_angle_between_vectors(v1, v2):
    """ Calculates angle in degrees between two vectors """
    # Normalize vectors to unit length
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    # Dot product gives cosine of angle
    dot_prod = np.dot(v1_u, v2_u)

    # Clamp to handle floating point errors slightly outside [-1, 1]
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    return np.degrees(np.arccos(dot_prod))

def get_posture_angles(landmarks):
    """
    Calculates the 3 critical angles for posture analysis.
    Returns None for angles if key landmarks are hidden.
    """
    mp_pose = mp.solutions.pose.PoseLandmark

    # 1. Get Coordinates (Try Left first, then Right)
    # Try Left Side
    hip = get_landmark(landmarks, mp_pose.LEFT_HIP)
    shoulder = get_landmark(landmarks, mp_pose.LEFT_SHOULDER)
    ear = get_landmark(landmarks, mp_pose.LEFT_EAR)

    # If Left side is missing any point, try Right Side
    if hip is None or shoulder is None or ear is None:
        hip = get_landmark(landmarks, mp_pose.RIGHT_HIP)
        shoulder = get_landmark(landmarks, mp_pose.RIGHT_SHOULDER)
        ear = get_landmark(landmarks, mp_pose.RIGHT_EAR)

    # If we STILL can't see the key points, we can't judge posture
    if hip is None or shoulder is None or ear is None:
        return { "torso_recline": None, "neck_protraction": None, "back_curve": None }

    # 2. Define Vectors
    # In MediaPipe, Y coordinates increase downwards (0 is top, 1 is bottom).
    # So, a vector pointing UP is (0, -1, 0).
    vec_vertical = np.array([0, -1, 0])

    # Vector: Torso (Hip -> Shoulder)
    vec_torso = shoulder - hip

    # Vector: Neck (Shoulder -> Ear)
    vec_neck = ear - shoulder

    # Vector: Back (Shoulder -> Ear vs Shoulder -> Hip)
    # For the 3-point curve, we calculate the angle at the shoulder.
    vec_shoulder_to_ear = ear - shoulder
    vec_shoulder_to_hip = hip - shoulder

    # 3. Calculate Angles

    # A. Torso Recline: Angle between Vertical and Torso Vector
    # 0 deg = Perfectly Straight Up.
    # 10-20 deg = Normal Recline.
    # > 20 deg = Slouching/Leaning back too far.
    angle_torso = calculate_angle_between_vectors(vec_vertical, vec_torso)

    # B. Neck Protraction: Angle between Vertical and Neck Vector
    # 0 deg = Head directly on top of shoulders.
    # > 25 deg = Text Neck (Head forward).
    angle_neck = calculate_angle_between_vectors(vec_vertical, vec_neck)

    # C. Back Curve: Angle between Neck and Torso
    # 180 deg = Perfectly Straight Line (Military posture).
    # < 145 deg = Rounding the shoulders forward.
    angle_back = calculate_angle_between_vectors(vec_shoulder_to_ear, vec_shoulder_to_hip)

    return {
        "torso_recline": angle_torso,
        "neck_protraction": angle_neck,
        "back_curve": angle_back
    }
