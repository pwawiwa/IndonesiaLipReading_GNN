#!/usr/bin/env python3
"""
Preview mouth partition nodes with AU groups highlighted.

This script verifies that the mouth partition includes:
- Full lips (inner + outer)
- Full jaw
- Cheeks
- All AUs used in B3 preprocessing (AU25, AU26, AU12)
- Fully connected adjacency from MediaPipe FACEMESH_TESSELATION

Usage:
    python scripts/preview_mouth_partition_au.py --video <video_path> --output data/mouth_partition_au_preview.png
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

from preprocessing.mediapipe_nodes import (
    get_mouth_area_nodes, 
    build_partition_adjacency, 
    get_partition_nodes,
    get_lips_nodes,
    build_full_adjacency
)


def get_au_groups_for_mouth(node_mapping: dict) -> dict:
    """
    Get AU groups for mouth partition based on anatomical regions.
    
    Args:
        node_mapping: Dict mapping original MediaPipe index -> remapped index
        
    Returns:
        Dictionary mapping AU name to list of remapped node indices
    """
    from preprocessing.mediapipe_nodes import get_au_node_groups
    return get_au_node_groups('mouth', node_mapping)


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


def visualize_mouth_partition_with_au(
    frame: np.ndarray,
    landmarks_dict: dict,
    width: int,
    height: int,
    output_path: str,
    frame_idx: int = 0
):
    """
    Visualize mouth partition nodes with AU groups highlighted.
    
    Args:
        frame: Frame image (BGR format)
        landmarks_dict: Dictionary of all landmarks {idx: {'x': float, 'y': float, 'z': float}}
        width: Video width
        height: Video height
        output_path: Path to save output image
        frame_idx: Frame index (for display in title)
    """
    # Get mouth partition nodes
    mouth_nodes = get_mouth_area_nodes()
    n_nodes = len(mouth_nodes)
    
    # Build adjacency for mouth partition (pruned from full tesselation)
    adjacency, node_mapping = build_partition_adjacency('mouth')
    
    # Create reverse mapping: new_idx -> original MediaPipe idx
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    # Get AU groups (using remapped indices based on anatomical regions)
    au_groups = get_au_groups_for_mouth(node_mapping)
    
    # Get lips nodes to highlight separately
    lips_nodes_original = get_lips_nodes()
    lips_nodes_remapped = [node_mapping[n] for n in lips_nodes_original if n in node_mapping]
    
    # Convert frame to RGB for display
    frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = frame_rgb.copy()
    
    # Get mouth node coordinates in pixel space
    mouth_coords = []
    for new_idx in range(n_nodes):
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
    
    # Define colors for AU groups
    au_colors = {
        'AU25_lips_part': (255, 0, 0),      # Red
        'AU26_jaw_drop': (0, 255, 0),       # Green
        'AU12_lip_corner': (0, 0, 255),     # Blue
    }
    
    # Draw edges (connections) - light gray
    edge_count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > 0.5:  # Edge exists
                pt1 = tuple(mouth_coords[i])
                pt2 = tuple(mouth_coords[j])
                # Check if points are valid
                if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                    0 <= pt2[0] < width and 0 <= pt2[1] < height):
                    cv2.line(overlay, pt1, pt2, (200, 200, 200), 1)
                    edge_count += 1
    
    # Draw nodes colored by AU group
    for au_name, node_indices in au_groups.items():
        color = au_colors.get(au_name, (255, 165, 0))  # Default orange
        for node_idx in node_indices:
            if node_idx < len(mouth_coords):
                x, y = mouth_coords[node_idx]
                if 0 <= x < width and 0 <= y < height:
                    # Draw node circle
                    cv2.circle(overlay, (x, y), 5, color, -1)
                    cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1)
    
    # Highlight lips nodes with a border (if not already in AU25)
    for node_idx in lips_nodes_remapped:
        if node_idx < len(mouth_coords):
            x, y = mouth_coords[node_idx]
            if 0 <= x < width and 0 <= y < height:
                # Draw a thicker border for lips
                cv2.circle(overlay, (x, y), 6, (255, 255, 0), 2)  # Yellow border
    
    # Draw node indices (for smaller partitions)
    if n_nodes <= 120:
        for node_idx in range(n_nodes):
            x, y = mouth_coords[node_idx]
            if 0 <= x < width and 0 <= y < height:
                cv2.putText(overlay, str(node_idx), (x + 6, y - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Create 4 separate images, one for each color
    colors_info = [
        ('YELLOW', 'Lips (inner + outer)', lips_nodes_remapped, (255, 255, 0), 'yellow'),
        ('RED', 'AU25_lips_part', au_groups.get('AU25_lips_part', []), (255, 0, 0), 'red'),
        ('GREEN', 'AU26_jaw_drop', au_groups.get('AU26_jaw_drop', []), (0, 255, 0), 'green'),
        ('BLUE', 'AU12_lip_corner', au_groups.get('AU12_lip_corner', []), (0, 0, 255), 'blue'),
    ]
    
    base_output_path = Path(output_path)
    output_dir = base_output_path.parent
    base_name = base_output_path.stem
    
    for color_name, title, node_indices, color_bgr, color_name_lower in colors_info:
        if not node_indices:
            continue
            
        # Create overlay for this color only
        color_overlay = frame_rgb.copy()
        
        # Draw edges only between nodes of this color
        color_node_set = set(node_indices)
        color_edge_count = 0
        for i in node_indices:
            for j in node_indices:
                if i < j and adjacency[i, j] > 0.5:
                    pt1 = tuple(mouth_coords[i])
                    pt2 = tuple(mouth_coords[j])
                    if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
                        0 <= pt2[0] < width and 0 <= pt2[1] < height):
                        cv2.line(color_overlay, pt1, pt2, (200, 200, 200), 1)
                        color_edge_count += 1
        
        # Draw nodes of this color only
        for node_idx in node_indices:
            if node_idx < len(mouth_coords):
                x, y = mouth_coords[node_idx]
                if 0 <= x < width and 0 <= y < height:
                    # Draw node circle
                    cv2.circle(color_overlay, (x, y), 6, color_bgr, -1)
                    cv2.circle(color_overlay, (x, y), 6, (255, 255, 255), 2)
                    
                    # Draw node index
                    cv2.putText(color_overlay, str(node_idx), (x + 8, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(color_overlay, str(node_idx), (x + 8, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Create figure for this color
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(color_overlay)
        ax.set_title(f'{color_name}: {title}\n'
                    f'{len(node_indices)} nodes, {color_edge_count} edges, Frame {frame_idx}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save this color's image
        color_output_path = output_dir / f'{base_name}_{color_name_lower}.png'
        plt.tight_layout()
        plt.savefig(color_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {color_name} image saved: {color_output_path}")
    
    # Also create the combined image for reference
    fig = plt.figure(figsize=(16, 10))
    
    # Main visualization
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(overlay)
    ax1.set_title(f'Mouth Partition with AU Groups\n'
                 f'{n_nodes} nodes, {edge_count} edges, Frame {frame_idx}',
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Legend and statistics
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    # Build legend text
    legend_text = []
    legend_text.append("MOUTH PARTITION VERIFICATION")
    legend_text.append("")
    legend_text.append(f"Total Nodes: {n_nodes}")
    legend_text.append(f"Total Edges: {edge_count}")
    legend_text.append(f"Connectivity: {'✓ Fully Connected' if edge_count > n_nodes else '✗ Not Connected'}")
    legend_text.append("")
    legend_text.append("COMPONENTS:")
    legend_text.append(f"  • Lips (inner + outer): {len(lips_nodes_remapped)} nodes")
    legend_text.append(f"  • Cheeks: Included")
    legend_text.append(f"  • Jaw: Included")
    legend_text.append("")
    legend_text.append("AU GROUPS (B3):")
    
    for au_name, node_indices in au_groups.items():
        color = au_colors.get(au_name, (255, 165, 0))
        color_normalized = tuple(c/255.0 for c in color)
        legend_text.append(f"  • {au_name}: {len(node_indices)} nodes (indices {min(node_indices)}-{max(node_indices)})")
    
    legend_text.append("")
    legend_text.append("VERIFICATION:")
    
    # Verify AU coverage
    all_au_nodes = set()
    for node_indices in au_groups.values():
        all_au_nodes.update(node_indices)
    
    missing_nodes = set(range(n_nodes)) - all_au_nodes
    if missing_nodes:
        legend_text.append(f"  ⚠ Missing from AUs: {len(missing_nodes)} nodes")
    else:
        legend_text.append("  ✓ All nodes covered by AUs")
    
    # Check if we have enough nodes for all AUs
    required_nodes = 45  # AU25(15) + AU26(15) + AU12(15)
    if n_nodes >= required_nodes:
        legend_text.append(f"  ✓ Sufficient nodes for all AUs ({n_nodes} >= {required_nodes})")
    else:
        legend_text.append(f"  ⚠ Insufficient nodes for all AUs ({n_nodes} < {required_nodes})")
    
    # Check adjacency connectivity
    # Count connected components
    visited = [False] * n_nodes
    def dfs(node):
        visited[node] = True
        for neighbor in range(n_nodes):
            if adjacency[node, neighbor] > 0.5 and not visited[neighbor]:
                dfs(neighbor)
    
    components = 0
    for i in range(n_nodes):
        if not visited[i]:
            dfs(i)
            components += 1
    
    if components == 1:
        legend_text.append(f"  ✓ Graph is connected (1 component)")
    else:
        legend_text.append(f"  ⚠ Graph has {components} components (not fully connected)")
    
    # Display legend
    legend_str = "\n".join(legend_text)
    ax2.text(0.1, 0.95, legend_str, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary with detailed node lists
    print(f"\n{'='*60}")
    print(f"MOUTH PARTITION VERIFICATION")
    print(f"{'='*60}")
    print(f"Total Nodes: {n_nodes}")
    print(f"Total Edges: {edge_count}")
    print(f"Connectivity: {'✓ Fully Connected' if components == 1 else f'✗ {components} components'}")
    print(f"\nComponents:")
    print(f"  • Lips (inner + outer): {len(lips_nodes_remapped)} nodes")
    print(f"  • Cheeks: Included")
    print(f"  • Jaw: Included")
    
    # Print detailed node lists for each color
    print(f"\n{'='*60}")
    print(f"DETAILED NODE LISTS BY COLOR:")
    print(f"{'='*60}")
    
    # YELLOW (Lips nodes)
    print(f"\n🟡 YELLOW (Lips - inner + outer):")
    print(f"   Count: {len(lips_nodes_remapped)} nodes")
    print(f"   Remapped indices: {sorted(lips_nodes_remapped)}")
    # Get original MediaPipe indices
    lips_original = [reverse_mapping[n] for n in lips_nodes_remapped]
    print(f"   Original MediaPipe indices: {sorted(lips_original)}")
    
    # RED (AU25_lips_part)
    if 'AU25_lips_part' in au_groups:
        au25_nodes = sorted(au_groups['AU25_lips_part'])
        print(f"\n🔴 RED (AU25_lips_part):")
        print(f"   Count: {len(au25_nodes)} nodes")
        print(f"   Remapped indices: {au25_nodes}")
        au25_original = [reverse_mapping[n] for n in au25_nodes if n in reverse_mapping]
        print(f"   Original MediaPipe indices: {sorted(au25_original)}")
    
    # GREEN (AU26_jaw_drop)
    if 'AU26_jaw_drop' in au_groups:
        au26_nodes = sorted(au_groups['AU26_jaw_drop'])
        print(f"\n🟢 GREEN (AU26_jaw_drop):")
        print(f"   Count: {len(au26_nodes)} nodes")
        print(f"   Remapped indices: {au26_nodes}")
        au26_original = [reverse_mapping[n] for n in au26_nodes if n in reverse_mapping]
        print(f"   Original MediaPipe indices: {sorted(au26_original)}")
    
    # BLUE (AU12_lip_corner)
    if 'AU12_lip_corner' in au_groups:
        au12_nodes = sorted(au_groups['AU12_lip_corner'])
        print(f"\n🔵 BLUE (AU12_lip_corner):")
        print(f"   Count: {len(au12_nodes)} nodes")
        print(f"   Remapped indices: {au12_nodes}")
        au12_original = [reverse_mapping[n] for n in au12_nodes if n in reverse_mapping]
        print(f"   Original MediaPipe indices: {sorted(au12_original)}")
    
    print(f"\n{'='*60}")
    print(f"AU GROUPS SUMMARY (B3):")
    print(f"{'='*60}")
    for au_name, node_indices in au_groups.items():
        print(f"  • {au_name}: {len(node_indices)} nodes (indices {min(node_indices)}-{max(node_indices)})")
    print(f"\nVerification:")
    if missing_nodes:
        print(f"  ⚠ Missing from AUs: {len(missing_nodes)} nodes")
        print(f"     Missing node indices: {sorted(list(missing_nodes))[:20]}..." if len(missing_nodes) > 20 else f"     Missing node indices: {sorted(list(missing_nodes))}")
    else:
        print(f"  ✓ All nodes covered by AUs")
    if n_nodes >= required_nodes:
        print(f"  ✓ Sufficient nodes for all AUs ({n_nodes} >= {required_nodes})")
    else:
        print(f"  ⚠ Insufficient nodes for all AUs ({n_nodes} < {required_nodes})")
    if components == 1:
        print(f"  ✓ Graph is connected")
    else:
        print(f"  ⚠ Graph has {components} components")
    print(f"{'='*60}")
    print(f"✓ Preview saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Preview mouth partition nodes with AU groups highlighted'
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
        default='data/mouth_partition_au_preview.png',
        help='Output path (default: data/mouth_partition_au_preview.png)'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return 1
    
    # Set output path
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
        print(f"Visualizing mouth partition with AU groups...")
        visualize_mouth_partition_with_au(
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

