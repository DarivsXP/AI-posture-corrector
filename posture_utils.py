import numpy as np

def get_landmark(landmarks, part_name):
    """
    Retrieves the 3D coordinates (x, y, z) for a specific landmark.
    """
    try:
        # Get the landmark using its name (e.g., 'LEFT_SHOULDER')
        landmark = landmarks[part_name.value]
        
        # Return coordinates, but only if the landmark is reasonably visible
        if landmark.visibility < 0.5:
            return None
            
        return [landmark.x, landmark.y, landmark.z]
    except Exception as e:
        print(f"Error getting landmark {part_name}: {e}")
        return None

def calculate_angle_3d(a, b, c):
    """
    Calculates the 3D angle between three points (A, B, C),
    where B is the vertex.
    """
    try:
        # Create vectors B->A and B->C
        v1 = np.array(a) - np.array(b)
        v2 = np.array(c) - np.array(b)

        # Calculate the angle using the dot product formula
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # To avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return None

        # Clamp the cosine value to [-1, 1] to avoid math domain errors
        cosine_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    except Exception as e:
        print(f"Error calculating 3D angle: {e}")
        return None

def get_posture_angles(landmarks):
    """
    Calculates the three key posture angles from the MediaPipe landmarks.
    """
    try:
        # Define MediaPipe landmark constants
        mp_pose = __import__("mediapipe").solutions.pose.PoseLandmark

        # 1. Get key landmarks (averaging left and right for stability)
        # We use 'visibility' to prioritize the more visible landmark
        
        left_ear = get_landmark(landmarks, mp_pose.LEFT_EAR)
        right_ear = get_landmark(landmarks, mp_pose.RIGHT_EAR)
        
        left_shoulder = get_landmark(landmarks, mp_pose.LEFT_SHOULDER)
        right_shoulder = get_landmark(landmarks, mp_pose.RIGHT_SHOULDER)
        
        left_hip = get_landmark(landmarks, mp_pose.LEFT_HIP)
        right_hip = get_landmark(landmarks, mp_pose.RIGHT_HIP)

        # Simple logic to pick the most visible side or average them
        # For simplicity here, we'll prefer the left side if visible
        ear = left_ear if left_ear else right_ear
        shoulder = left_shoulder if left_shoulder else right_shoulder
        hip = left_hip if left_hip else right_hip

        # If any key landmark is not visible, we can't calculate
        if not ear or not shoulder or not hip:
            return {
                "back_curve": None,
                "torso_recline": None,
                "neck_protraction": None
            }

        # 2. Define the theoretical "vertical" line for recline angles
        # This vector points straight up from the hip
        vertical_vector_point = [hip[0], hip[1] + 1.0, hip[2]] # A point 1 unit "up" from the hip
        
        # 3. Calculate the three angles
        
        # Angle 1: Back Curve Angle (Hip - Shoulder - Ear)
        back_curve = calculate_angle_3d(hip, shoulder, ear)
        
        # Angle 2: Torso Recline Angle (Vertical - Hip - Shoulder)
        torso_recline = calculate_angle_3d(vertical_vector_point, hip, shoulder)
        
        # Angle 3: Neck Protraction Angle (Vertical - Shoulder - Ear)
        # For this, the vertical line must start from the shoulder
        vertical_vector_point_shoulder = [shoulder[0], shoulder[1] + 1.0, shoulder[2]]
        neck_protraction = calculate_angle_3d(vertical_vector_point_shoulder, shoulder, ear)

        return {
            "back_curve": back_curve,
            "torso_recline": torso_recline,
            "neck_protraction": neck_protraction
        }

    except Exception as e:
        print(f"Error in get_posture_angles: {e}")
        return {
            "back_curve": None,
            "torso_recline": None,
            "neck_protraction": None
        }