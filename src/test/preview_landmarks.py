import numpy as np
import cv2

# ====== CONFIG ======
landmark_path = "/Users/wirahutomo/Projects/TA_IDLR_GNN/data/landmarks/train/ada/ada_00001.npz"
video_path    = "/Users/wirahutomo/Projects/TA_IDLR_GNN/data/IDLRW-DATASET/ada/train/ada_00001.mp4"

# ====== LOAD LANDMARKS ======
data = np.load(landmark_path)
print("Keys in npz:", data.files)

landmarks = data["landmarks"]  # shape: (T, N, 2) or (T, N, 3)
print("Landmarks shape:", landmarks.shape)

# ====== LOAD VIDEO ======
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video frames: {frame_count}, FPS: {fps}")

# ====== PLAYBACK LOOP ======
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(landmarks):
        break

    # overlay landmarks for current frame
    for point in landmarks[frame_idx]:
        x, y = int(point[0]), int(point[1])  # only take first 2 dims
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # display
    cv2.imshow("Landmarks Preview", frame)
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == ord('q'):  # press q to quit
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
