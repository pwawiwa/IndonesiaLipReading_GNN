import torch
import cv2
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
pt_file = "preprocessed/train/ada_00001.pt"  # Path to your .pt file
output_video = "landmark_viz.mp4"
frame_size = (640, 480)  # Width, Height
fps = 25
node_color = (0, 0, 255)  # Red
node_radius = 3
node_thickness = -1  # Filled circles

# -----------------------------
# Allow numpy arrays to load safely
# -----------------------------
torch.serialization.add_safe_globals([np.ndarray])

# -----------------------------
# Load .pt file
# -----------------------------
data = torch.load(pt_file)

nodes = data["nodes"]  # (num_frames, num_nodes, 3)
word_label = data["word"]
print(f"Word label: {word_label}")
print(f"Landmark tensor shape: {nodes.shape}")

num_frames, num_nodes, _ = nodes.shape
width, height = frame_size

# -----------------------------
# Video writer setup
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

# -----------------------------
# Draw landmarks frame by frame
# -----------------------------
for frame_idx in range(num_frames):
    lm = nodes[frame_idx]  # (num_nodes, 3)

    # Scale landmarks to fit frame
    x = ((lm[:, 0] - lm[:, 0].min()) / (lm[:, 0].ptp() + 1e-6) * (width * 0.5) + width * 0.25).astype(int)
    y = ((-lm[:, 1] - lm[:, 1].min()) / (lm[:, 1].ptp() + 1e-6) * (height * 0.5) + height * 0.25).astype(int)

    # Create blank frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw each landmark
    for xi, yi in zip(x, y):
        cv2.circle(frame, (xi, yi), node_radius, node_color, node_thickness)

    # Optional: draw frame number
    cv2.putText(frame, f"Frame {frame_idx+1}/{num_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(frame)

out.release()
print(f"Video saved to {output_video}")
