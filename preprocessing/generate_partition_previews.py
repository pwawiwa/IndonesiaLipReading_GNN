#!/usr/bin/env python3
"""
Generate preview images for each partition showing:
1. Graph structure (nodes + adjacency connections)
2. Speech mask on/off states on sample frames
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
import sys
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.mediapipe_nodes import (
    get_partition_nodes, 
    get_lips_nodes,
    get_mouth_area_nodes,
    build_partition_adjacency
)


def load_extracted_data(pt_file: str) -> Dict:
    """Load extracted .pt file."""
    return torch.load(pt_file, weights_only=False)


def visualize_graph_structure(
    adjacency: torch.Tensor,
    landmarks_sample: torch.Tensor,
    partition: str,
    output_path: str
):
    """
    Visualize graph structure: nodes and connections.
    For mouth partition, also shows regions (lips, jaw, cheeks) with different colors.
    
    Args:
        adjacency: Adjacency matrix (n_nodes, n_nodes)
        landmarks_sample: Sample landmark coordinates (n_nodes, 2) normalized [0,1]
        partition: Partition name
        output_path: Path to save image
    """
    n_nodes = adjacency.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Convert normalized coords to display coordinates (assume square image)
    # MediaPipe landmarks are normalized [0, 1]
    coords = landmarks_sample.numpy()
    
    # For mouth partition, color nodes by region
    node_colors = None
    if partition == 'mouth':
        # Get original node indices for mouth partition
        mouth_nodes_original = get_mouth_area_nodes()
        lips_nodes_original = get_lips_nodes()
        
        # Build node mapping (original -> new index)
        node_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(mouth_nodes_original)}
        
        # Classify nodes: lips, cheeks, jaw
        cheeks_original = [
            50, 118, 119, 100, 101, 36, 203, 205, 206, 216,  # Left cheek
            280, 347, 348, 330, 329, 266, 423, 425, 426, 436,  # Right cheek
        ]
        jaw_original = [
            152, 377, 400, 378, 379, 365, 397, 288, 361, 323,  # Chin center
            58, 172, 136, 150, 149, 176, 148, 152, 454,  # Jaw line
            18, 200, 199, 175, 169, 170, 140, 135, 138, 171,  # Additional jaw
            204, 208, 364, 367, 369, 394, 395, 396, 430,  # More jaw nodes
        ]
        
        node_colors = []
        for orig_idx in mouth_nodes_original:
            if orig_idx in lips_nodes_original:
                node_colors.append('red')  # Lips = red
            elif orig_idx in cheeks_original:
                node_colors.append('blue')  # Cheeks = blue
            elif orig_idx in jaw_original:
                node_colors.append('green')  # Jaw = green
            else:
                node_colors.append('gray')  # Other = gray
    
    # Draw edges (connections)
    edge_count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0.5:  # Edge exists
                ax.plot([coords[i, 0], coords[j, 0]], 
                       [coords[i, 1], coords[j, 1]], 
                       'b-', alpha=0.3, linewidth=0.5)
                edge_count += 1
    
    # Draw nodes with region colors for mouth partition
    if node_colors:
        # Group nodes by color
        color_groups = {}
        for i, color in enumerate(node_colors):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(i)
        
        # Draw each color group
        for color, indices in color_groups.items():
            group_coords = coords[indices]
            ax.scatter(group_coords[:, 0], group_coords[:, 1], 
                      c=color, s=50, alpha=0.8, edgecolors='black', linewidths=1,
                      label=f'{color.capitalize()} region')
    else:
        # Default: all nodes same color
        ax.scatter(coords[:, 0], coords[:, 1], 
                  c='red', s=50, alpha=0.8, edgecolors='black', linewidths=1)
    
    # Add node indices for small partitions
    # IMPORTANT: Show ORIGINAL MediaPipe indices (0-467), not remapped indices (0, 1, 2, ... N-1)
    if n_nodes <= 150:
        # Get original MediaPipe node indices
        from preprocessing.mediapipe_nodes import get_partition_nodes
        original_nodes = get_partition_nodes(partition)
        
        for i in range(n_nodes):
            # Show original MediaPipe index, not remapped index
            original_idx = original_nodes[i]
            ax.annotate(str(original_idx), (coords[i, 0], coords[i, 1]), 
                       fontsize=5, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)  # Invert Y axis (image coordinates)
    ax.set_aspect('equal')
    
    title = f'{partition.upper()} Partition Graph Structure\n'
    title += f'{n_nodes} nodes, {edge_count} edges'
    if partition == 'mouth':
        title += '\n(Red=Lips, Blue=Cheeks, Green=Jaw)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Normalized X coordinate', fontsize=10)
    ax.set_ylabel('Normalized Y coordinate', fontsize=10)
    ax.grid(True, alpha=0.3)
    if node_colors:
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Graph structure saved: {output_path}")


def visualize_speech_mask_states(
    video_path: str,
    landmarks: torch.Tensor,
    speech_mask: torch.Tensor,
    adjacency: torch.Tensor,
    partition: str,
    output_path: str,
    num_frames: int = 4
):
    """
    Visualize sample frames with speech mask on/off states.
    
    Args:
        video_path: Path to video file
        landmarks: Landmark coordinates (frames, n_nodes, 2) normalized
        speech_mask: Speech mask (frames,)
        adjacency: Adjacency matrix (n_nodes, n_nodes)
        partition: Partition name
        output_path: Path to save image
        num_frames: Number of frames to show
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠ Could not open video: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Find frames with speech on and off
    speech_frames = torch.where(speech_mask > 0.5)[0].tolist()
    non_speech_frames = torch.where(speech_mask < 0.5)[0].tolist()
    
    # Select sample frames
    frames_to_show = []
    if speech_frames:
        speech_samples = np.linspace(0, len(speech_frames) - 1, num_frames // 2, dtype=int)
        frames_to_show.extend([speech_frames[i] for i in speech_samples])
    if non_speech_frames:
        non_speech_samples = np.linspace(0, len(non_speech_frames) - 1, num_frames // 2, dtype=int)
        frames_to_show.extend([non_speech_frames[i] for i in non_speech_samples])
    
    # Limit to available frames
    frames_to_show = [f for f in frames_to_show if f < landmarks.shape[0]][:num_frames]
    
    if not frames_to_show:
        print(f"  ⚠ No valid frames found")
        cap.release()
        return
    
    # Create figure
    cols = 2
    rows = (len(frames_to_show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Color scheme
    partition_colors = {
        'lips': (0, 255, 0),      # Green
        'mouth': (255, 165, 0),   # Orange
        'full': (255, 0, 255)     # Magenta
    }
    node_color = partition_colors.get(partition, (0, 255, 0))
    
    for plot_idx, frame_idx in enumerate(frames_to_show):
        if plot_idx >= len(axes):
            break
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Get landmarks for this frame
        frame_landmarks = landmarks[frame_idx].numpy()
        is_speech = speech_mask[frame_idx].item() > 0.5
        
        # Convert normalized coords to pixel coords
        pixel_coords = frame_landmarks.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * width
        pixel_coords[:, 1] = pixel_coords[:, 1] * height
        pixel_coords = pixel_coords.astype(np.int32)
        
        # Create overlay
        overlay = frame.copy()
        
        # Highlight speech frames with yellow tint
        if is_speech:
            overlay = cv2.addWeighted(overlay, 0.7, 
                                   np.full_like(overlay, (0, 255, 255)), 0.3, 0)
            mask_status = "SPEECH ON"
            status_color = (0, 255, 0)  # Green
        else:
            mask_status = "SPEECH OFF"
            status_color = (128, 128, 128)  # Gray
        
        # Draw edges (connections)
        n_nodes = len(pixel_coords)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency[i, j] > 0.5:  # Edge exists
                    pt1 = tuple(pixel_coords[i])
                    pt2 = tuple(pixel_coords[j])
                    # Check if points are valid
                    if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                        0 <= pt2[0] < width and 0 <= pt2[1] < height):
                        cv2.line(overlay, pt1, pt2, (100, 100, 255), 1)
        
        # Draw nodes
        for node_idx, coord in enumerate(pixel_coords):
            x, y = coord
            if 0 <= x < width and 0 <= y < height:
                # Draw node circle
                cv2.circle(overlay, (x, y), 3, node_color, -1)
                cv2.circle(overlay, (x, y), 3, (255, 255, 255), 1)
        
        # Convert BGR to RGB for matplotlib
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Plot
        ax = axes[plot_idx]
        ax.imshow(overlay_rgb)
        ax.set_title(f"Frame {frame_idx} - {mask_status}", 
                    fontsize=12, fontweight='bold',
                    color='green' if is_speech else 'gray')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(frames_to_show), len(axes)):
        axes[idx].axis('off')
    
    cap.release()
    
    plt.suptitle(f'{partition.upper()} Partition - Speech Mask Visualization', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Speech mask visualization saved: {output_path}")


def get_real_facemesh_coords():
    """
    Get real MediaPipe FaceMesh coordinates by processing a sample face image.
    Uses MediaPipe to detect landmarks on a generated face template.
    
    Returns:
        coords: (468, 2) numpy array with normalized [0,1] coordinates, or None if detection fails
    """
    try:
        import mediapipe as mp
        
        # Create a standard face image that MediaPipe can process
        # Use a neutral face template
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Draw a simple face structure to help MediaPipe detect
        center = (256, 256)
        cv2.ellipse(img, center, (180, 220), 0, 0, 360, (240, 240, 240), -1)
        cv2.ellipse(img, (200, 200), (25, 35), 0, 0, 360, (200, 200, 200), -1)  # Left eye
        cv2.ellipse(img, (312, 200), (25, 35), 0, 0, 360, (200, 200, 200), -1)  # Right eye
        cv2.ellipse(img, (256, 250), (15, 40), 0, 0, 360, (220, 220, 220), -1)  # Nose
        cv2.ellipse(img, (256, 320), (50, 20), 0, 0, 360, (200, 200, 200), -1)  # Mouth
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3
        )
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        coords = np.zeros((468, 2), dtype=np.float32)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx < 468:
                    coords[idx, 0] = landmark.x
                    coords[idx, 1] = landmark.y
            face_mesh.close()
            return coords
        
        face_mesh.close()
        return None
    except Exception as e:
        print(f"Warning: Could not get real FaceMesh coordinates: {e}")
        return None


def visualize_partition_from_nodes(partition: str, output_path: str):
    """
    Generate preview visualization from node definitions using real MediaPipe coordinates.
    Shows node layout and regions for mouth partition with real positions relative to full face.
    """
    from preprocessing.mediapipe_nodes import build_partition_adjacency
    
    # Get nodes and build adjacency
    nodes = get_partition_nodes(partition)
    adjacency, node_mapping = build_partition_adjacency(partition)
    n_nodes = len(nodes)
    
    # Get real MediaPipe FaceMesh coordinates
    real_coords = get_real_facemesh_coords()
    
    if real_coords is not None:
        # Use real MediaPipe coordinates - map partition nodes to their real positions
        coords = np.zeros((n_nodes, 2), dtype=np.float32)
        for new_idx, orig_idx in enumerate(nodes):
            if orig_idx < 468:
                # Use the real MediaPipe landmark position
                coords[new_idx, 0] = float(real_coords[orig_idx, 0])
                coords[new_idx, 1] = float(real_coords[orig_idx, 1])
            else:
                # Fallback for nodes outside 468 range (shouldn't happen)
                coords[new_idx] = [0.5, 0.5]
        print(f"  Using real MediaPipe FaceMesh coordinates")
    else:
        # Fallback: Create approximate coordinates based on known MediaPipe layout
        print(f"  Warning: Using approximate coordinates (MediaPipe detection failed)")
        coords = np.zeros((n_nodes, 2), dtype=np.float32)
        
        # Approximate MediaPipe face layout for mouth region
        # These are rough estimates based on typical MediaPipe face mesh structure
        for new_idx, orig_idx in enumerate(nodes):
            # Rough estimates - mouth region is typically around x=0.4-0.6, y=0.5-0.7
            if partition == 'mouth':
                # Distribute nodes in mouth region
                angle = (new_idx / n_nodes) * 2 * np.pi
                x = 0.5 + 0.2 * np.cos(angle)
                y = 0.6 + 0.15 * np.sin(angle)
                coords[new_idx] = [x, y]
            else:
                coords[new_idx] = [0.5, 0.5]
    
    # Convert to tensor for visualization
    landmarks_tensor = torch.tensor(coords, dtype=torch.float32)
    
    # Use existing visualization function
    visualize_graph_structure(adjacency, landmarks_tensor, partition, output_path)


def visualize_partition_on_video_frame(
    video_path: str,
    frame_idx: int,
    partition: str,
    output_path: str
):
    """
    Visualize partition nodes on a real video frame.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        partition: Partition name
        output_path: Path to save image
    """
    try:
        import mediapipe as mp
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ⚠ Could not open video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_idx >= total_frames:
            print(f"  ⚠ Frame {frame_idx} exceeds total frames {total_frames}")
            cap.release()
            return False
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"  ⚠ Could not read frame {frame_idx}")
            return False
        
        # Initialize MediaPipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        face_mesh.close()
        
        if not results.multi_face_landmarks:
            print(f"  ⚠ No face detected in frame {frame_idx}")
            return False
        
        # Get partition nodes and adjacency
        nodes = get_partition_nodes(partition)
        adjacency, node_mapping = build_partition_adjacency(partition)
        
        # Get landmarks for partition nodes
        face_landmarks = results.multi_face_landmarks[0]
        pixel_coords = []
        for orig_idx in nodes:
            if orig_idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[orig_idx]
                x_pixel = int(landmark.x * width)
                y_pixel = int(landmark.y * height)
                pixel_coords.append((x_pixel, y_pixel))
            else:
                pixel_coords.append((0, 0))
        
        pixel_coords = np.array(pixel_coords)
        
        # Create overlay
        overlay = frame.copy()
        
        # Color scheme for mouth partition
        if partition == 'mouth':
            lips_nodes_original = get_lips_nodes()
            cheeks_original = [
                50, 118, 119, 100, 101, 36, 203, 205, 206, 216,
                280, 347, 348, 330, 329, 266, 423, 425, 426, 436,
            ]
            jaw_original = [
                152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
                58, 172, 136, 150, 149, 176, 148, 152, 454,
                18, 200, 199, 175, 169, 170, 140, 135, 138, 171,
                204, 208, 364, 367, 369, 394, 395, 396, 430,
            ]
        else:
            lips_nodes_original = []
            cheeks_original = []
            jaw_original = []
        
        # Draw edges
        n_nodes = len(pixel_coords)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency[i, j] > 0.5:
                    pt1 = tuple(pixel_coords[i])
                    pt2 = tuple(pixel_coords[j])
                    if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                        0 <= pt2[0] < width and 0 <= pt2[1] < height):
                        cv2.line(overlay, pt1, pt2, (100, 100, 255), 1, cv2.LINE_AA)
        
        # Draw nodes with region colors
        for new_idx, orig_idx in enumerate(nodes):
            x, y = pixel_coords[new_idx]
            if 0 <= x < width and 0 <= y < height:
                if partition == 'mouth':
                    if orig_idx in lips_nodes_original:
                        color = (0, 0, 255)  # Red (BGR)
                    elif orig_idx in cheeks_original:
                        color = (255, 0, 0)  # Blue (BGR)
                    elif orig_idx in jaw_original:
                        color = (0, 255, 0)  # Green (BGR)
                    else:
                        color = (128, 128, 128)  # Gray
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.circle(overlay, (x, y), 4, color, -1)
                cv2.circle(overlay, (x, y), 4, (255, 255, 255), 1)
        
        # Convert BGR to RGB for matplotlib
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(overlay_rgb)
        ax.set_title(f'{partition.upper()} Partition on Video Frame\n'
                    f'Video: {Path(video_path).name}, Frame: {frame_idx}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Error visualizing frame: {e}")
        return False


def find_videos_from_dataset(dataset_root: str, split: str = "train", max_videos: int = 5):
    """
    Find video files from the dataset.
    
    Args:
        dataset_root: Root directory of IDLRW-DATASET
        split: Split to search (train, val, test)
        max_videos: Maximum number of videos to return
        
    Returns:
        List of video paths
    """
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        return []
    
    video_files = []
    
    # Search in word directories
    for word_dir in sorted(dataset_path.iterdir()):
        if not word_dir.is_dir():
            continue
        
        split_dir = word_dir / split
        if split_dir.exists():
            videos = list(split_dir.glob("*.mp4"))
            video_files.extend(videos)
            
            if len(video_files) >= max_videos:
                break
    
    return video_files[:max_videos]


def generate_preview_for_partition(
    partition: str,
    extracted_dir: str = "data/extracted",
    output_dir: str = None,
    split: str = "train",
    num_samples: int = 2,
    dataset_root: str = "data/IDLRW-DATASET",
    num_video_previews: int = 3
):
    """
    Generate preview images for a partition.
    
    Args:
        partition: Partition name (lips, mouth, full)
        extracted_dir: Directory with extracted .pt files
        output_dir: Output directory for previews (None = use extracted_dir)
        split: Split to use (train, val, test)
        num_samples: Number of sample videos to visualize
    """
    print("="*80)
    print(f"GENERATING PREVIEWS FOR: {partition.upper()} PARTITION")
    print("="*80)
    
    extracted_path = Path(extracted_dir) / partition / f"{partition}_{split}.pt"
    
    # Setup output directory - always use data/extracted/partition/previews
    if output_dir is None:
        output_dir = Path(extracted_dir) / partition / "previews"
    else:
        output_dir = Path(output_dir)
        if not str(output_dir).endswith('previews'):
            output_dir = output_dir / partition / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    if not extracted_path.exists():
        print(f"⚠ Extracted file not found: {extracted_path}")
        print(f"  Generating preview from node definitions with real MediaPipe coordinates...")
        
        # Generate preview from node definitions using real MediaPipe positions
        graph_output = output_dir / f"{partition}_graph_structure.png"
        visualize_partition_from_nodes(partition, str(graph_output))
        
        # Generate previews from actual videos
        print(f"\n[2/2] Generating previews from {num_video_previews} videos...")
        video_files = find_videos_from_dataset(dataset_root, split, max_videos=num_video_previews)
        
        if video_files:
            for idx, video_path in enumerate(video_files, 1):
                # Extract frame from middle of video
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_idx = total_frames // 2 if total_frames > 0 else 0
                    cap.release()
                    
                    video_name = Path(video_path).stem
                    video_output = output_dir / f"{partition}_video{idx}_{video_name}_frame{frame_idx}.png"
                    
                    print(f"  Processing video {idx}/{len(video_files)}: {Path(video_path).name} (frame {frame_idx})...")
                    if visualize_partition_on_video_frame(str(video_path), frame_idx, partition, str(video_output)):
                        print(f"  ✓ Saved: {video_output.name}")
                    else:
                        print(f"  ⚠ Failed to process: {Path(video_path).name}")
        else:
            print(f"  ⚠ No videos found in {dataset_root}")
        
        print("\n" + "="*80)
        print(f"✓ PREVIEW GENERATION COMPLETE FOR {partition.upper()}")
        print("="*80)
        print(f"\nPreview images saved to: {output_dir}")
        print(f"  - {partition}_graph_structure.png")
        if video_files:
            print(f"  - {partition}_video*_*.png ({len(video_files)} video previews)")
        return
    
    # Load data
    print(f"\nLoading: {extracted_path}")
    data = load_extracted_data(str(extracted_path))
    
    adjacency = data['adjacency']
    videos = data['videos']
    n_nodes = data.get('n_nodes', adjacency.shape[0])
    
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {int((adjacency.sum() - adjacency.trace()).item()) // 2}")
    print(f"  Videos: {len(videos)}")
    
    # Setup output directory (already done above if file didn't exist)
    if output_dir is None:
        output_dir = Path(extracted_dir) / partition / "previews"
    else:
        output_dir = Path(output_dir)
        if not str(output_dir).endswith('previews'):
            output_dir = output_dir / partition / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # 1. Visualize graph structure
    print("\n[1/2] Generating graph structure visualization...")
    
    # Get sample landmarks (use first video, first frame)
    sample_video_id = list(videos.keys())[0]
    sample_video = videos[sample_video_id]
    sample_landmarks = sample_video['landmarks'][0]  # First frame
    
    graph_output = output_dir / f"{partition}_graph_structure.png"
    visualize_graph_structure(adjacency, sample_landmarks, partition, str(graph_output))
    
    # 2. Visualize speech mask states
    print("\n[2/2] Generating speech mask visualization...")
    
    # Sample videos
    video_ids = list(videos.keys())
    sample_ids = np.random.choice(video_ids, min(num_samples, len(video_ids)), replace=False)
    
    for idx, video_id in enumerate(sample_ids, 1):
        video_data = videos[video_id]
        video_path = video_data.get('video_path')
        
        if not video_path or not Path(video_path).exists():
            print(f"  ⚠ Video not found: {video_path}")
            continue
        
        landmarks = video_data['landmarks']
        speech_mask = video_data['speech_mask']
        
        mask_output = output_dir / f"{partition}_speech_mask_sample{idx}_{video_id}.png"
        visualize_speech_mask_states(
            video_path, landmarks, speech_mask, adjacency,
            partition, str(mask_output)
        )
    
    print("\n" + "="*80)
    print(f"✓ PREVIEW GENERATION COMPLETE FOR {partition.upper()}")
    print("="*80)
    print(f"\nPreview images saved to: {output_dir}")
    print(f"  - {partition}_graph_structure.png")
    print(f"  - {partition}_speech_mask_sample*.png")


def main():
    parser = argparse.ArgumentParser(description='Generate partition preview images')
    parser.add_argument('--partition', type=str, choices=['lips', 'mouth', 'full'],
                       default=None, help='Partition to preview (default: all)')
    parser.add_argument('--extracted-dir', type=str, default='data/extracted',
                       help='Directory with extracted .pt files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: extracted_dir/partition/previews)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Split to use')
    parser.add_argument('--num-samples', type=int, default=2,
                       help='Number of sample videos for speech mask visualization')
    parser.add_argument('--num-video-previews', type=int, default=3,
                       help='Number of video previews to generate (from dataset)')
    parser.add_argument('--dataset-root', type=str, default='data/IDLRW-DATASET',
                       help='Root directory of IDLRW-DATASET')
    
    args = parser.parse_args()
    
    partitions = [args.partition] if args.partition else ['lips', 'mouth', 'full']
    
    for partition in partitions:
        try:
            generate_preview_for_partition(
                partition=partition,
                extracted_dir=args.extracted_dir,
                output_dir=args.output_dir,
                split=args.split,
                num_samples=args.num_samples,
                dataset_root=args.dataset_root,
                num_video_previews=args.num_video_previews
            )
        except Exception as e:
            print(f"\n✗ Error processing {partition}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL PREVIEWS GENERATED")
    print("="*80)


if __name__ == '__main__':
    main()

