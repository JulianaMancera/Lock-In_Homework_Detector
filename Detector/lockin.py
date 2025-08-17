import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize YOLO for object detection
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
    print(f"Warning: Failed to load YOLO model. Object detection disabled. Error: {e}")

cap = cv2.VideoCapture(1)  # change to 0 if external cam is not detected

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Tracking distraction
distraction_threshold = 5  # seconds without focus to consider distracted
last_focus_time = time.time()
distracted = False
focus_score = 100
score_decay_rate = 0.5
score_recovery_rate = 1

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_gaze_on_screen(landmarks, frame_shape):
    h, w = frame_shape[:2]
    left_eye_inner = landmarks[133]  
    right_eye_inner = landmarks[362]  
    nose = landmarks[1]  

    left_eye_pos = (int(left_eye_inner.x * w), int(left_eye_inner.y * h))
    right_eye_pos = (int(right_eye_inner.x * w), int(right_eye_inner.y * h))
    nose_pos = (int(nose.x * w), int(nose.y * h))

    head_angle = calculate_angle(left_eye_pos, nose_pos, right_eye_pos)
    return head_angle < 45  # allow up to 45° head tilt

def detect_objects(frame):
    if not yolo_loaded:
        return False, []

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    distraction_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["cell phone", "book", "cup"]:
                distraction_detected = True
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h, classes[class_id]))
    return distraction_detected, boxes

print("Starting Lock-In Homework Detector. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    focused = False
    distraction_reason = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if is_gaze_on_screen(face_landmarks.landmark, frame.shape):
                focused = True
            else:
                distraction_reason = "Looking away from screen"

            # Box box
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))

            # Color changes based on focus/distraction
            color = (0, 255, 0) if focused else (0, 0, 255)

            # Face box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)

            # Eyes
            left_eye = (int(face_landmarks.landmark[153].x * w), int(face_landmarks.landmark[133].y * h))
            right_eye = (int(face_landmarks.landmark[262].x * w), int(face_landmarks.landmark[362].y * h))
            cv2.rectangle(frame, (left_eye[0]-20, left_eye[1]-20), (left_eye[0]+10, left_eye[1]+10), color, 2)
            cv2.rectangle(frame, (right_eye[0]-20, right_eye[1]-20), (right_eye[0]+10, right_eye[1]+10), color, 2)

    # Object distraction detection
    distraction_obj, object_boxes = detect_objects(frame)
    if distraction_obj:
        focused = False
        distraction_reason = "Holding distracting object"
        # Draw red boxes for objects
        for (x, y, w, h, label) in object_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Focus scoring 
    if focused:
        last_focus_time = time.time()
        focus_score = min(100, focus_score + score_recovery_rate)
        if distracted:
            print(f"Locked in! Focus score: {int(focus_score)}")
            distracted = False
    else:
        if time.time() - last_focus_time > distraction_threshold:
            if not distracted:
                print(f"Student distracted! Reason: {distraction_reason}. Score: {int(focus_score)}")
                distracted = True
        focus_score = max(0, focus_score - score_decay_rate)

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
