import cv2
import mediapipe as mp
import time
import posture_utils  # <-- Import your helper file

# --- Adaptive Feedback Timer Logic ---
class SlouchTimer:
    def __init__(self, alert_threshold_1=5.0, alert_threshold_2=15.0):
        self.start_time = None
        self.state = "ideal"  # "ideal", "initial_slouch", "sustained_slouch"
        
        self.ALERT_THRESHOLD_1 = alert_threshold_1 # 5 seconds
        self.ALERT_THRESHOLD_2 = alert_threshold_2 # 15 seconds
        
        self.alert_message = ""

    def update(self, is_slouching):
        current_time = time.time()

        if is_slouching:
            if self.state == "ideal":
                # User just started slouching
                self.start_time = current_time
                self.state = "initial_slouch"
                self.alert_message = "Poor Posture Detected"
            
            elif self.state == "initial_slouch":
                # User is still slouching, check if timer 1 is up
                if (current_time - self.start_time) > self.ALERT_THRESHOLD_1:
                    self.state = "sustained_slouch"
                    self.alert_message = "ALERT: Please Sit Up!"
            
            elif self.state == "sustained_slouch":
                # User is in chronic slouch, check if timer 2 is up
                if (current_time - self.start_time) > self.ALERT_THRESHOLD_2:
                    self.alert_message = "ALERT: Chronic Slouch! (Audible)"
                    # In a real app, you would trigger the sound here
            
        else:
            # User is in ideal posture
            self.start_time = None
            self.state = "ideal"
            self.alert_message = "Good Posture!"

        return self.alert_message

# --- Main Application ---
def run_monitor():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1 # Use 1 for a good balance of speed and accuracy
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize Slouch Timer
    slouch_timer = SlouchTimer()

    # Start Webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a natural, selfie-view
        image = cv2.flip(image, 1)

        # To improve performance, mark the image as not writeable
        image.flags.set_writeable(False)
        
        # Recolor image from BGR (OpenCV) to RGB (MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get landmarks
        results = pose.process(image_rgb)

        # Recolor back to BGR to draw on it
        image.flags.set_writeable(True)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Get the original BGR

        # Check if landmarks are detected
        if results.pose_landmarks:
            # Draw the landmarks on the image (optional)
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # --- This is the core logic ---
            # 1. Get angles from our util
            angles = posture_utils.get_posture_angles(results.pose_landmarks.landmark)
            
            # 2. Define "Ideal Ranges" (from Chapter 2)
            # These are just examples, use your final values
            IDEAL_BACK_CURVE = (170, 180) 
            IDEAL_TORSO_RECLINE = (80, 100) # (Vertical-Hip-Shoulder)
            IDEAL_NECK_PROTRACTION = (80, 90) # (Vertical-Shoulder-Ear)

            # 3. Check if user is slouching
            is_slouching = False
            if angles["back_curve"] and angles["back_curve"] < IDEAL_BACK_CURVE[0]:
                is_slouching = True
            if angles["torso_recline"] and angles["torso_recline"] < IDEAL_TORSO_RECLINE[0]:
                is_slouching = True
            # Add other angle checks as needed...

            # 4. Update the adaptive feedback timer
            alert_message = slouch_timer.update(is_slouching)
            
            # 5. Calculate Score (simple example)
            # A real score would be a weighted average of all angles
            score = 100 if not is_slouching else 20
            
            # --- Display Info on Screen ---
            # Put the Alert Message
            cv2.putText(image_bgr, f"STATUS: {alert_message}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Put the Score
            cv2.putText(image_bgr, f"SCORE: {score}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
            # Put the Angle Data
            cv2.putText(image_bgr, f"Back Curve: {int(angles.get('back_curve', 0))}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image_bgr, f"Torso Recline: {int(angles.get('torso_recline', 0))}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image_bgr, f"Neck Protraction: {int(angles.get('neck_protraction', 0))}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the image
        cv2.imshow('AI Posture Monitor', image_bgr)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# --- Run the application ---
if __name__ == "__main__":
    run_monitor()