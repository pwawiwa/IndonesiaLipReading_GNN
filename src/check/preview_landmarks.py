"""
Preview landmark connections from video file or processed dataset (processed_v3)
Shows only landmarks and their connections (no real image background)

Updated for processed_v3 with full face landmarks:
- All 468 MediaPipe landmarks (full face)
- Uses MediaPipe's default FACEMESH_TESSELATION connections (best topology)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import argparse
import sys

# Add parent directory to path to import facemesh_extractor
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.facemesh_extractor import FaceMeshExtractor


def visualize_landmark_connections(landmarks: torch.Tensor, edge_index: torch.Tensor, 
                                   frame_idx: int = 0, ax=None, title: str = ""):
    """
    Visualize landmark connections for a single frame
    Optimized for comprehensive bottom face landmarks (processed_v3)
    
    Args:
        landmarks: [T, N, 3] tensor of landmarks
        edge_index: [2, E] tensor of edge connections
        frame_idx: Which frame to visualize
        ax: Matplotlib axis (if None, creates new figure)
        title: Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Get landmarks for this frame
    frame_landmarks = landmarks[frame_idx].numpy()  # [N, 3]
    
    # Extract x, y coordinates (ignore z for 2D visualization)
    x = frame_landmarks[:, 0]
    y = frame_landmarks[:, 1]
    
    # Normalize coordinates to fit in [0, 1] range for better visualization
    # MediaPipe landmarks are typically in [0, 1] range already
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range == 0:
        x_range = 0.1
    if y_range == 0:
        y_range = 0.1
    
    x_padding = x_range * 0.1
    y_padding = y_range * 0.1
    
    # Draw edges (connections) - use thinner lines for better visibility with many landmarks
    edge_index_np = edge_index.numpy()
    for i in range(edge_index_np.shape[1]):
        src_idx = edge_index_np[0, i]
        dst_idx = edge_index_np[1, i]
        
        # Only draw if both nodes are valid
        if src_idx < len(x) and dst_idx < len(x):
            ax.plot([x[src_idx], x[dst_idx]], 
                   [y[src_idx], y[dst_idx]], 
                   'b-', alpha=0.2, linewidth=0.3, zorder=1)
    
    # Draw landmarks (nodes) - smaller size for better visibility with many landmarks
    ax.scatter(x, y, c='red', s=20, alpha=0.7, zorder=5, edgecolors='darkred', linewidths=0.5)
    
    # Label key landmarks (first 20 for reference, but with smaller font)
    num_labels = min(20, len(x))
    for i in range(num_labels):
        ax.annotate(str(i), (x[i], y[i]), fontsize=4, alpha=0.5, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.3, edgecolor='none'))
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title if title else f'Landmark Connections (Frame {frame_idx})')
    ax.grid(True, alpha=0.2)
    
    return ax


def visualize_sequence(landmarks: torch.Tensor, edge_index: torch.Tensor,
                      num_frames: int = 5, video_id: str = "", label: str = ""):
    """
    Visualize landmark connections across multiple frames
    
    Args:
        landmarks: [T, N, 3] tensor of landmarks
        edge_index: [2, E] tensor of edge connections
        num_frames: Number of frames to visualize
        video_id: Video identifier
        label: Label/word for this sample
    """
    T = landmarks.shape[0]
    
    # Select frames to visualize
    if T <= num_frames:
        frame_indices = list(range(T))
    else:
        # Select evenly spaced frames
        frame_indices = [int(i * (T - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    # Create subplot grid
    cols = min(3, len(frame_indices))
    rows = (len(frame_indices) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, frame_idx in enumerate(frame_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        title = f'Frame {frame_idx}/{T-1}'
        visualize_landmark_connections(landmarks, edge_index, frame_idx, ax, title)
    
    # Remove empty subplots
    for idx in range(len(frame_indices), rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            fig.delaxes(axes[row, col])
        else:
            fig.delaxes(axes[col])
    
    fig.suptitle(f'Landmark Connections: {label} ({video_id})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def preview_from_video(video_path: str, output_dir: Path, num_frames: int = 5):
    """
    Preview landmark connections directly from a video file
    Uses FaceMeshExtractor to extract landmarks with the same configuration
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for saved images
        num_frames: Number of frames to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    print(f"📹 Processing video: {video_path}")
    print(f"   Using FaceMeshExtractor with comprehensive bottom face landmarks")
    
    # Initialize extractor (uses same ROI_INDICES and EDGE_PAIRS_ORIGINAL)
    extractor = FaceMeshExtractor(num_workers=1)
    
    # Extract landmarks from video
    print(f"   Extracting landmarks...")
    result = extractor.extract_video(video_path, word=video_path.stem)
    
    if result is None:
        print(f"❌ Error: Failed to extract landmarks from video")
        return
    
    landmarks = result['landmarks']  # [T, N, 3]
    edge_index = result['edge_index']  # [2, E]
    video_id = result['video_id']
    T = result['num_frames']
    N = landmarks.shape[1]
    
    print(f"\n📊 Video: {video_id}")
    print(f"   Sequence length: {T} frames")
    print(f"   Landmarks per frame: {N} (full face - all 468 landmarks)")
    print(f"   Edge connections: {edge_index.shape[1]} (MediaPipe FACEMESH_TESSELATION)")
    
    # Visualize sequence
    fig = visualize_sequence(landmarks, edge_index, num_frames, video_id, video_path.stem)
    
    # Save figure
    output_path = output_dir / f"landmarks_{video_id}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"✅ Preview complete! Saved to {output_path}")
    print(f"{'='*60}\n")


def preview_from_pt(pt_path: str, output_dir: Path, num_samples: int = 5, 
                   num_frames: int = 5, sample_indices: Optional[List[int]] = None):
    """
    Preview landmark connections from processed .pt file
    
    Args:
        pt_path: Path to .pt file
        output_dir: Output directory for saved images
        num_samples: Number of samples to visualize
        num_frames: Number of frames per sample to visualize
        sample_indices: Specific sample indices to visualize (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Loading processed data: {pt_path}")
    samples = torch.load(pt_path)
    print(f"   Loaded {len(samples)} samples")
    
    if sample_indices is None:
        # Select random samples
        indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    else:
        indices = sample_indices
    
    for idx in indices:
        sample = samples[idx]
        landmarks = sample['landmarks']  # [T, N, 3]
        label = sample['label']
        video_id = sample.get('video_id', f'sample_{idx}')
        
        # Get edge_index if available
        if 'edge_index' in sample:
            edge_index = sample['edge_index']  # [2, E]
        else:
            # Fallback: create simple k-NN edges
            N = landmarks.shape[1]
            edges = []
            k = 5
            for i in range(N):
                for j in range(max(0, i - k), min(N, i + k + 1)):
                    if i != j:
                        edges.append([i, j])
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            else:
                # Self-loops as fallback
                edge_index = torch.arange(N).repeat(2, 1)
        
        T, N, _ = landmarks.shape
        
        print(f"\n📊 Sample {idx}: {label} ({video_id})")
        print(f"   Sequence length: {T} frames")
        print(f"   Landmarks per frame: {N} (full face - all 468 landmarks)")
        print(f"   Edge connections: {edge_index.shape[1]} (MediaPipe FACEMESH_TESSELATION)")
        
        # Visualize sequence
        fig = visualize_sequence(landmarks, edge_index, num_frames, video_id, label)
        
        # Save figure
        output_path = output_dir / f"landmarks_{video_id}_{idx}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"✅ Preview complete! Saved {len(indices)} samples to {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Preview landmark connections from video or processed dataset (processed_v3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview from a video file (direct extraction)
  python src/check/preview_landmarks.py --video_path data/IDLRW-DATASET/word/train/video.mp4
  
  # Preview from processed_v3 train set
  python src/check/preview_landmarks.py --pt_path data/processed_v3/train.pt
  
  # Preview from specific split
  python src/check/preview_landmarks.py --pt_path data/processed_v3/val.pt
  
  # Preview specific samples from .pt file
  python src/check/preview_landmarks.py --pt_path data/processed_v3/train.pt --sample_indices 0 1 2 3 4
        """
    )
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to video file (if provided, extracts landmarks directly)')
    parser.add_argument('--pt_path', type=str, default='data/processed_v3/train.pt',
                       help='Path to processed .pt file (default: data/processed_v3/train.pt)')
    parser.add_argument('--output_dir', type=str, default='outputs/landmark_previews',
                       help='Output directory for preview images')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (for .pt files)')
    parser.add_argument('--num_frames', type=int, default=5,
                       help='Number of frames per sample to visualize')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                       help='Specific sample indices to visualize (optional, for .pt files)')
    
    args = parser.parse_args()
    
    # If video_path is provided, preview from video directly
    if args.video_path:
        preview_from_video(
            video_path=args.video_path,
            output_dir=Path(args.output_dir),
            num_frames=args.num_frames
        )
    else:
        # Otherwise, preview from .pt file
        preview_from_pt(
            pt_path=args.pt_path,
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples,
            num_frames=args.num_frames,
            sample_indices=args.sample_indices
        )


if __name__ == "__main__":
    main()

