#!/usr/bin/env python3
"""
Preview mouth nodes and connections from a video frame.

Usage:
    python scripts/preview_mouth_frame.py --video <video_path> --frame <frame_idx>
    python scripts/preview_mouth_frame.py --video <video_path> --frame 0 --output data/mouth_preview.png
"""
import argparse
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.mediapipe_nodes import get_mouth_area_nodes, build_partition_adjacency, get_partition_nodes


def get_extended_mouth_area_nodes(landmarks_dict: dict = None) -> list:
    """
    Get extended mouth area nodes with more bottom/chin region nodes for preview.
    Excludes nose and upper face nodes - only includes bottom face (mouth, chin, jaw).
    If landmarks_dict is provided, filters nodes to only include those below the mouth.
    
    Args:
        landmarks_dict: Optional dictionary of landmarks to filter by Y coordinate
    
    Returns:
        List of node indices including extended bottom region (no nose/upper face)
    """
    # Start with lips nodes only (exclude nose_bottom from mouth_area)
    from preprocessing.mediapipe_nodes import get_lips_nodes
    nodes = get_lips_nodes()
    
    # Add lower cheek nodes (but exclude upper cheek/nose area)
    # Based on reference: focus on lower cheeks only
    lower_cheeks = [
        # Lower cheek area (not upper)
        100, 101, 118, 119,  # Left lower cheek
        329, 330, 347, 348,  # Right lower cheek
    ]
    
    # Lower face / chin / jaw nodes from reference image
    # These are MediaPipe landmarks in the chin and lower jaw area
    # Based on reference: nodes in lower face/chin/jawline region
    lower_face_candidates = [
        # Chin center and surrounding (from reference)
        18, 200, 199, 175, 169, 170, 140, 176, 148, 152,
        # Lower jaw line (from reference)
        58, 172, 136, 150, 149, 176, 148, 152,
        # More chin and lower jaw (from reference - only bottom region)
        135, 138, 171, 204, 208, 364, 365, 367, 369,
        377, 378, 379, 394, 395, 396, 397, 400, 430,
        # Additional lower face mesh points (chin/jaw only)
        288, 323, 454,
    ]
    
    # Nodes to EXCLUDE (nose and upper face from reference)
    exclude_nodes = [
        # Nose bottom (explicitly exclude)
        2, 98, 327,
        # Upper face/nose area nodes (from reference)
        50, 93, 123, 137, 147, 177, 205, 213, 215, 266, 280,
        352, 356, 360, 361, 366, 376, 401, 411, 423, 425, 426, 427, 485, 487,
        # Upper cheek nodes
        36, 203, 206, 216,  # Left upper
        266, 423, 425, 426, 436,  # Right upper (some overlap)
    ]
    
    # Combine lips + lower cheeks + lower face
    all_candidates = list(set(nodes + lower_cheeks + lower_face_candidates))
    
    # Remove excluded nodes (nose and upper face)
    all_candidates = [n for n in all_candidates if n not in exclude_nodes]
    
    # Initialize with filtered candidates
    lower_face_nodes = all_candidates.copy()
    
    # If landmarks are provided, filter to only include nodes below mouth
    if landmarks_dict is not None:
        # Get average Y coordinate of lip nodes (higher Y = lower on face)
        lip_y_coords = []
        for node_idx in nodes:
            if node_idx in landmarks_dict:
                lip_y_coords.append(landmarks_dict[node_idx]['y'])
        
        if lip_y_coords:
            # Find nodes with Y coordinate at or below mouth level
            lip_avg_y = sum(lip_y_coords) / len(lip_y_coords)
            # Threshold: include nodes at mouth level or below (no upper face)
            threshold_y = lip_avg_y - 0.02  # Allow slightly above for cheeks
            
            filtered_lower = []
            for node_idx in all_candidates:
                if node_idx in landmarks_dict:
                    # Only include if at or below mouth level (exclude nose/upper)
                    if landmarks_dict[node_idx]['y'] >= threshold_y:
                        filtered_lower.append(node_idx)
            
            # Only use filtered if we found nodes, otherwise use all candidates
            if filtered_lower:
                lower_face_nodes = filtered_lower
    
    # Final cleanup: remove any remaining excluded nodes
    lower_face_nodes = [n for n in lower_face_nodes if n not in exclude_nodes]
    
    # Combine and remove duplicates, then sort
    all_nodes = list(set(lower_face_nodes))
    all_nodes.sort()
    
    return all_nodes


def extract_frame_landmarks(video_path: str, frame_idx: int) -> tuple:
    """
    Extract landmarks from a specific frame.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        
    Returns:
        Tuple of (frame_image, landmarks_dict, width, height)
    """
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"Frame index {frame_idx} exceeds total frames {total_frames}")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frame {frame_idx}")
    
    cap.release()
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in frame {frame_idx}")
    
    # Get first face landmarks
    face_landmarks = results.multi_face_landmarks[0]
    
    # Extract all 468 landmarks (normalized coordinates)
    landmarks_dict = {}
    for idx, landmark in enumerate(face_landmarks.landmark):
        landmarks_dict[idx] = {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        }
    
    return frame, landmarks_dict, width, height


def visualize_mouth_nodes_and_connections(
    frame: np.ndarray,
    landmarks_dict: dict,
    width: int,
    height: int,
    output_path: str,
    frame_idx: int = 0
):
    """
    Visualize mouth nodes and their connections on a frame.
    
    Args:
        frame: Frame image (BGR format)
        landmarks_dict: Dictionary of all landmarks {idx: {'x': float, 'y': float, 'z': float}}
        width: Video width
        height: Video height
        output_path: Path to save output image
        frame_idx: Frame index (for display in title)
    """
    # Get extended mouth area nodes (with more bottom region)
    # Pass landmarks_dict to filter nodes to only those below mouth
    mouth_nodes = get_extended_mouth_area_nodes(landmarks_dict)
    
    # Build adjacency for extended nodes (prune from full tesselation)
    from preprocessing.mediapipe_nodes import build_full_adjacency
    full_adj = build_full_adjacency()
    node_set = set(mouth_nodes)
    
    # Create node mapping: original_idx -> new_idx
    node_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(mouth_nodes)}
    n_nodes = len(mouth_nodes)
    
    # Create adjacency matrix for extended nodes
    adjacency = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    for orig_i in mouth_nodes:
        for orig_j in mouth_nodes:
            if full_adj[orig_i, orig_j] > 0:
                new_i = node_mapping[orig_i]
                new_j = node_mapping[orig_j]
                adjacency[new_i, new_j] = 1.0
    
    # Create reverse mapping: new_idx -> original MediaPipe idx
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    # Convert frame to RGB for display
    frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = frame_rgb.copy()
    
    # Get mouth node coordinates in pixel space
    mouth_coords = []
    for new_idx in range(len(mouth_nodes)):
        orig_idx = reverse_mapping[new_idx]
        if orig_idx in landmarks_dict:
            x_norm = landmarks_dict[orig_idx]['x']
            y_norm = landmarks_dict[orig_idx]['y']
            # Convert normalized to pixel coordinates
            x_pixel = int(x_norm * width)
            y_pixel = int(y_norm * height)
            mouth_coords.append((x_pixel, y_pixel))
        else:
            mouth_coords.append((0, 0))
    
    mouth_coords = np.array(mouth_coords)
    
    # Draw edges (connections)
    n_nodes = len(mouth_coords)
    edge_count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0.5:  # Edge exists
                pt1 = tuple(mouth_coords[i])
                pt2 = tuple(mouth_coords[j])
                # Check if points are valid
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    cv2.line(overlay, pt1, pt2, (100, 100, 255), 1)
                    edge_count += 1
    
    # Draw nodes
    node_color = (255, 165, 0)  # Orange for mouth
    for node_idx, coord in enumerate(mouth_coords):
        x, y = coord
        if 0 <= x < width and 0 <= y < height:
            # Draw node circle
            cv2.circle(overlay, (x, y), 4, node_color, -1)
            cv2.circle(overlay, (x, y), 4, (255, 255, 255), 1)
            
            # Add node index label (for smaller partitions, show indices)
            if n_nodes <= 100:
                cv2.putText(overlay, str(node_idx), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(overlay)
    ax.set_title(f'Mouth Area Nodes and Connections\n'
                f'{n_nodes} nodes, {edge_count} edges, Frame {frame_idx}',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Preview saved: {output_path}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {edge_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Preview mouth nodes and connections from a video frame'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to video file'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help='Frame index to extract (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (default: data/mouth_preview_frame{frame_idx}.png)'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return 1
    
    # Set output path
    if args.output is None:
        output_path = Path('data') / f'mouth_preview_frame{args.frame}.png'
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract frame and landmarks
        print(f"Extracting frame {args.frame} from: {video_path}")
        frame, landmarks_dict, width, height = extract_frame_landmarks(
            str(video_path), args.frame
        )
        
        # Visualize
        print(f"Visualizing mouth nodes and connections...")
        visualize_mouth_nodes_and_connections(
            frame, landmarks_dict, width, height, str(output_path), args.frame
        )
        
        print(f"\n✓ Success! Preview saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

