"""
Visualize sample mouth extraction data showing landmarks with speech mask ON and OFF.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.mediapipe_nodes import get_partition_nodes, build_partition_adjacency


def visualize_landmarks_frame(
    landmarks: np.ndarray,
    adjacency: torch.Tensor,
    speech_mask_value: float,
    frame_idx: int,
    video_id: str,
    output_path: Path
):
    """
    Visualize landmarks for a single frame.
    
    Args:
        landmarks: (n_nodes, 2) array of normalized coordinates [0, 1]
        adjacency: (n_nodes, n_nodes) adjacency matrix
        speech_mask_value: 1.0 if speech ON, 0.0 if speech OFF
        frame_idx: Frame index
        video_id: Video identifier
        output_path: Path to save image
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Title
    mask_status = "SPEECH ON" if speech_mask_value > 0.5 else "SPEECH OFF"
    ax.set_title(
        f'Video: {video_id} | Frame {frame_idx} | {mask_status}',
        fontsize=14, fontweight='bold'
    )
    
    # Invert y-axis (MediaPipe coordinates: y=0 at top)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Normalized X coordinate')
    ax.set_ylabel('Normalized Y coordinate')
    
    # Draw edges
    n_nodes = len(landmarks)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0.5:
                ax.plot(
                    [landmarks[i, 0], landmarks[j, 0]],
                    [landmarks[i, 1], landmarks[j, 1]],
                    'b-', alpha=0.2, linewidth=0.5
                )
    
    # Draw nodes with color based on speech mask
    if speech_mask_value > 0.5:
        # Speech ON: green nodes
        ax.scatter(
            landmarks[:, 0], landmarks[:, 1],
            c='green', s=30, alpha=0.8,
            edgecolors='darkgreen', linewidths=0.5,
            label='Landmarks (Speech ON)'
        )
    else:
        # Speech OFF: red nodes
        ax.scatter(
            landmarks[:, 0], landmarks[:, 1],
            c='red', s=30, alpha=0.8,
            edgecolors='darkred', linewidths=0.5,
            label='Landmarks (Speech OFF)'
        )
    
    ax.legend()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_sample_visualizations():
    """Create visualizations for sample_mouth.pt"""
    sample_file = Path("data/extracted/mouth/sample_mouth.pt")
    
    if not sample_file.exists():
        print(f"Error: {sample_file} not found")
        return
    
    print(f"Loading {sample_file}...")
    data = torch.load(sample_file, map_location='cpu', weights_only=False)
    
    partition = data.get('partition', 'mouth')
    adjacency = data.get('adjacency')
    
    if adjacency is None:
        print("Error: Adjacency matrix not found in data")
        return
    
    output_dir = Path("data/extracted/mouth/previews/sample_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualizations...")
    print("="*80)
    
    for vid_id, vid_data in data['videos'].items():
        print(f"\nProcessing video: {vid_id}")
        landmarks = vid_data['landmarks'].numpy()  # (frames, nodes, 2)
        speech_mask = vid_data['speech_mask'].numpy()  # (frames,)
        
        n_frames = landmarks.shape[0]
        print(f"  Total frames: {n_frames}")
        print(f"  Speech ON frames: {int(speech_mask.sum())}")
        print(f"  Speech OFF frames: {int((1 - speech_mask).sum())}")
        
        # Find first frame with speech ON and first with speech OFF
        speech_on_frames = np.where(speech_mask > 0.5)[0]
        speech_off_frames = np.where(speech_mask <= 0.5)[0]
        
        # Visualize first speech ON frame
        if len(speech_on_frames) > 0:
            frame_idx = speech_on_frames[0]
            output_path = output_dir / f"{vid_id}_frame{frame_idx:03d}_speechON.png"
            visualize_landmarks_frame(
                landmarks[frame_idx],
                adjacency,
                1.0,
                frame_idx,
                vid_id,
                output_path
            )
        
        # Visualize first speech OFF frame
        if len(speech_off_frames) > 0:
            frame_idx = speech_off_frames[0]
            output_path = output_dir / f"{vid_id}_frame{frame_idx:03d}_speechOFF.png"
            visualize_landmarks_frame(
                landmarks[frame_idx],
                adjacency,
                0.0,
                frame_idx,
                vid_id,
                output_path
            )
        
        # Also visualize a few more frames (every 5th frame)
        for frame_idx in range(0, n_frames, 5):
            if frame_idx not in [speech_on_frames[0] if len(speech_on_frames) > 0 else -1,
                                speech_off_frames[0] if len(speech_off_frames) > 0 else -1]:
                mask_val = speech_mask[frame_idx]
                mask_str = "ON" if mask_val > 0.5 else "OFF"
                output_path = output_dir / f"{vid_id}_frame{frame_idx:03d}_speech{mask_str}.png"
                visualize_landmarks_frame(
                    landmarks[frame_idx],
                    adjacency,
                    mask_val,
                    frame_idx,
                    vid_id,
                    output_path
                )
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"  Total images created: {len(list(output_dir.glob('*.png')))}")


if __name__ == '__main__':
    create_sample_visualizations()

