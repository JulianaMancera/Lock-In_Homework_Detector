import cv2
import mediapipe as mp
import numpy as np
import time

# initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# initialize YOLO
yolo_loaded = False
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    yolo_loaded = True
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"YOLO not loaded: {e}")

# camera setup    
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)


# Variables for tracking distraction
distraction_threshold = 5  # seconds without focus to consider distracted
focus_score = 100
last_focus_time = time.time()
distracted = False

# Eye landmarks
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def get_eye_box(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 5
    return (x_min - padding, y_min - padding, 
            x_max - x_min + 2*padding, y_max - y_min + 2*padding)

def is_looking_at_screen(landmarks, frame_shape):
    h, w = frame_shape[:2]
    
    # Key points for head pose
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[175]
    
    # Convert to pixel coordinates
    nose = (int(nose_tip.x * w), int(nose_tip.y * h))
    left_eye_pos = (int(left_eye.x * w), int(left_eye.y * h))
    right_eye_pos = (int(right_eye.x * w), int(right_eye.y * h))
    chin_pos = (int(chin.x * w), int(chin.y * h))
    
    # Calculate head orientation
    eye_center_x = (left_eye_pos[0] + right_eye_pos[0]) / 2
    eye_center_y = (left_eye_pos[1] + right_eye_pos[1]) / 2
    
    # Horizontal turn (yaw) - nose position relative to eye center
    face_width = abs(right_eye_pos[0] - left_eye_pos[0])
    nose_offset = abs(nose[0] - eye_center_x)
    horizontal_ratio = nose_offset / face_width if face_width > 0 else 0
    
    # Vertical tilt (pitch) - eye-nose vs nose-chin distance
    eye_nose_dist = abs(nose[1] - eye_center_y)
    nose_chin_dist = abs(chin_pos[1] - nose[1])
    vertical_ratio = eye_nose_dist / nose_chin_dist if nose_chin_dist > 0 else 0
    
    # Check if looking at screen (adjusted thresholds)
    looking_horizontally = horizontal_ratio < 0.25  # less strict
    looking_vertically = 0.3 < vertical_ratio < 0.8  # normal range
    
    reason = ""
    if not looking_horizontally:
        reason = "Head turned away"
    elif not looking_vertically:
        reason = "Head tilted away"
    
    return looking_horizontally and looking_vertically, reason


print("Starting Lock-In Homework Detector. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    focused = True
    distraction_reason = ""

    # face detection and gaze analysis
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Check if looking at screen
            looking_at_screen, gaze_reason = is_looking_at_screen(face_landmarks.landmark, frame.shape)
            
            if not looking_at_screen:
                focused = False
                distraction_reason = gaze_reason

            # Draw eye boxes
            left_eye_box = get_eye_box(face_landmarks.landmark, LEFT_EYE, frame.shape)
            right_eye_box = get_eye_box(face_landmarks.landmark, RIGHT_EYE, frame.shape)

            # raw bounding box around face and eyes
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))

            # Color changes based on focus/distraction
            color = (0, 255, 0) if focused else (0, 0, 255)

            # Face box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)

           # Draw eye rectangles
            cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), 
                         (left_eye_box[0] + left_eye_box[2], left_eye_box[1] + left_eye_box[3]), color, 2)
            cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), 
                         (right_eye_box[0] + right_eye_box[2], right_eye_box[1] + right_eye_box[3]), color, 2)
    else:
        focused = False
        distraction_reason = "No face detected"

    # Focus scoring
    current_time = time.time()
    if focused:
        last_focus_time = current_time
        focus_score = min(100, focus_score + 1.5)  # recover faster
        if distracted:
            print(f"Back to focus! Score: {int(focus_score)}")
            distracted = False
    else:
        # Start decreasing score immediately when distracted
        time_distracted = current_time - last_focus_time
        if time_distracted > 1:  # faster detection
            focus_score = max(0, focus_score - 2.0)  # decrease faster
            if not distracted:
                print(f"DISTRACTED: {distraction_reason}")
                distracted = True

    # Status text
    status_text = f"{'Locked in' if focused else f'Distracted: {distraction_reason}'} | Score: {int(focus_score)}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if focused else (0, 0, 255), 2)

    cv2.imshow('Lock-In Homework Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
