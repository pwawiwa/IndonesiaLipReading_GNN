# src/preprocess.py

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/IDLRW-DATASET"   # Root dataset folder
OUTPUT_DIR = "preprocessed"       # Output folder for .pt files
TARGET_FRAMES = 25                # Pad/truncate sequences
LANDMARK_MODE = "lip"             # Options: "lip", "mouth", "full"

# Landmark indices for different modes (MediaPipe 468 points)
LIP_LANDMARKS = list(range(61, 81))
MOUTH_LANDMARKS = list(range(48, 68))
FULL_FACE_LANDMARKS = list(range(468))

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
print("MediaPipe FaceMesh initialized")

# -----------------------------
# Helper functions
# -----------------------------
def get_landmarks(frame):
    """Extract landmarks from a single frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark

    if LANDMARK_MODE == "lip":
        idxs = LIP_LANDMARKS
    elif LANDMARK_MODE == "mouth":
        idxs = MOUTH_LANDMARKS
    else:
        idxs = FULL_FACE_LANDMARKS

    coords = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in idxs], dtype=np.float32)
    coords -= coords.mean(axis=0, keepdims=True)  # normalize
    return coords

def process_video(video_path):
    """Extract landmark sequences from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_landmarks(frame)
        if landmarks is not None:
            frames.append(landmarks)

    cap.release()
    if len(frames) == 0:
        return None

    # Pad/truncate to TARGET_FRAMES
    seq = np.zeros((TARGET_FRAMES, frames[0].shape[0], 3), dtype=np.float32)
    for i, f in enumerate(frames[:TARGET_FRAMES]):
        seq[i] = f

    return seq

def get_label_from_path(video_path):
    """Infer word label from grandparent folder."""
    return os.path.basename(os.path.dirname(os.path.dirname(video_path)))

def process_split(split):
    """Process all videos in a dataset split (train/val/test)."""
    print(f"\nProcessing split: {split}")

    # Find all .mp4 files recursively
    pattern = os.path.join(DATA_DIR, "*", split, "*.mp4")
    video_files = sorted(glob.glob(pattern))
    print(f"Found {len(video_files)} videos in split '{split}'")

    # Prepare output folder
    output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    for video_path in tqdm(video_files, desc=f"Processing {split}"):
        seq = process_video(video_path)
        if seq is None:
            print(f"  Skipping {os.path.basename(video_path)} (no landmarks detected)")
            continue

        label = get_label_from_path(video_path)
        # Save as .pt
        save_path = os.path.join(output_dir, os.path.basename(video_path).replace(".mp4", ".pt"))
        torch.save({"nodes": seq, "word": label}, save_path)

# -----------------------------
# Main
# -----------------------------
def main():
    for split in ["train", "val", "test"]:
        process_split(split)
    print("\nPreprocessing complete! Saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
