import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")

def play_alert():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

# Constants and thresholds
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 0.06
YAWN_CONSEC_FRAMES = 10
ALERT_COOLDOWN = 5  # seconds between alerts

# Variables to keep track of state
eye_close_counter = 0
yawn_counter = 0
last_alert_time = 0
alert_log = []

# Mediapipe face mesh init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Eye landmark indices (for EAR calculation)
eye_indices_left = [362, 385, 387, 263, 373, 380]
eye_indices_right = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices):
    coords = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    horizontal = np.linalg.norm(coords[0] - coords[3])
    vertical1 = np.linalg.norm(coords[1] - coords[5])
    vertical2 = np.linalg.norm(coords[2] - coords[4])
    ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0
    return ear

def is_yawning(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    dist = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
    return dist > YAWN_THRESH

# Open webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    new_frame_time = time.time()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        ear_left = eye_aspect_ratio(landmarks, eye_indices_left)
        ear_right = eye_aspect_ratio(landmarks, eye_indices_right)
        ear = (ear_left + ear_right) / 2.0

        yawn_detected = is_yawning(landmarks)

        # Eye closure counter & alert
        if ear < EYE_AR_THRESH:
            eye_close_counter += 1
        else:
            eye_close_counter = 0

        # Yawning counter & alert
        if yawn_detected:
            yawn_counter += 1
        else:
            yawn_counter = 0

        # Current time for cooldown checking
        current_time = time.time()

        # Check and trigger alert for eye closure
        if eye_close_counter >= EYE_AR_CONSEC_FRAMES:
            if current_time - last_alert_time > ALERT_COOLDOWN:
                play_alert()
                last_alert_time = current_time
                alert_log.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Eye Closure Alert")
            cv2.putText(frame, "DROWSINESS ALERT: Eyes Closed!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check and trigger alert for yawning
        if yawn_counter >= YAWN_CONSEC_FRAMES:
            if current_time - last_alert_time > ALERT_COOLDOWN:
                play_alert()
                last_alert_time = current_time
                alert_log.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Yawn Alert")
            cv2.putText(frame, "DROWSINESS ALERT: Yawning!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Face detected status
        cv2.putText(frame, "Face Detected", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Drowsiness level bar (0 to max counter)
        bar_length = min(eye_close_counter * 5, 100)
        cv2.rectangle(frame, (30, 80), (30 + bar_length, 100), (0, 0, 255), -1)
        cv2.putText(frame, "Drowsiness Level", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        eye_close_counter = 0
        yawn_counter = 0
        cv2.putText(frame, "No Face Detected", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Calculate and show FPS
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Optionally print alert logs after program ends
print("\nAlert Log:")
for log in alert_log:
    print(log)
