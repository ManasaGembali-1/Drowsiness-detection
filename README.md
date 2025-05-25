# Drowsiness-detection

# Driver Drowsiness Detection System

This project detects driver drowsiness by monitoring eye closure and yawning using a webcam in real-time. It leverages MediaPipe for face landmark detection, OpenCV for video processing, and Pygame for alert sounds.

---

## Features

- Detects eye closure using Eye Aspect Ratio (EAR) to identify drowsiness.
- Detects yawning based on mouth opening distance.
- Alerts driver with sound notifications.
- Displays drowsiness level and alerts on screen.
- Logs alert timestamps for reference.

---

## Requirements

- Python 3.7 or higher (recommended Python 3.10)
- OpenCV
- MediaPipe
- Pygame
- NumPy

---

## Install dependencies with:

- pip install opencv-python mediapipe pygame numpy
  IF PYTHON VERSION DONOT SUPPORT THEN,
- py -3.10 -m venv drow_env
- Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
- .\drow_env\Scripts\activate
- pip install opencv-python mediapipe pygame numpy
