import cv2
import numpy as np

def load_video_frames(video_path, resize=None):
    """Load video menjadi list of frames (BGR -> RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return np.array(frames)  # shape (T, H, W, 3)
