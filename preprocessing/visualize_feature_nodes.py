"""
Visualize feature engineering node assignments (AU groups, geometric anchors, etc.)
and save preview images to data/extracted/{partition}/previews/
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.mediapipe_nodes import (
    get_partition_nodes, 
    build_partition_adjacency,
    get_au_node_groups
)
from utils import ensure_dir


def get_real_facemesh_coords():
    """
    Get real MediaPipe FaceMesh coordinates by processing a sample face image.
    Uses MediaPipe to detect landmarks on a generated face template.
    
    Returns:
        coords: (468, 2) numpy array with normalized [0,1] coordinates, or None if detection fails
    """
    try:
        import mediapipe as mp
        import cv2
        
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


def get_sample_landmarks(partition: str, n_frames: int = 25):
    """
    Get sample landmark coordinates for visualization.
    Uses real MediaPipe FaceMesh coordinates if available, otherwise falls back to synthetic.
    
    Args:
        partition: 'lips', 'mouth', or 'full'
        n_frames: Number of frames to generate
        
    Returns:
        landmarks: (n_frames, n_nodes, 2) tensor with sample coordinates
    """
    nodes = get_partition_nodes(partition)
    n_nodes = len(nodes)
    
    # Try to get real MediaPipe FaceMesh coordinates
    real_coords = get_real_facemesh_coords()
    
    landmarks = torch.zeros(n_frames, n_nodes, 2)
    
    if real_coords is not None:
        # Use real MediaPipe coordinates
        for i, orig_node_idx in enumerate(nodes):
            if orig_node_idx < 468:
                # Use the real MediaPipe landmark position
                landmarks[:, i, 0] = float(real_coords[orig_node_idx, 0])
                landmarks[:, i, 1] = float(real_coords[orig_node_idx, 1])
            else:
                # Fallback for nodes outside 468 range (shouldn't happen)
                landmarks[:, i, 0] = 0.5
                landmarks[:, i, 1] = 0.5
    else:
        # Fallback: Create a simple face-like coordinate pattern
        print("Warning: Using synthetic coordinates. Install MediaPipe for real FaceMesh visualization.")
        for i, node_idx in enumerate(nodes):
            angle = (i / n_nodes) * 2 * np.pi
            if partition == 'full':
                x = 0.5 + 0.3 * np.cos(angle)
                y = 0.5 + 0.4 * np.sin(angle)
            elif partition == 'mouth':
                x = 0.5 + 0.15 * np.cos(angle)
                y = 0.6 + 0.1 * np.sin(angle)
            else:  # lips
                x = 0.5 + 0.1 * np.cos(angle)
                y = 0.6 + 0.05 * np.sin(angle)
            landmarks[:, i, 0] = x
            landmarks[:, i, 1] = y
    
    return landmarks


def visualize_au_groups(partition: str, output_dir: Path):
    """
    Visualize AU node groups on a sample face mesh.
    
    Args:
        partition: 'lips', 'mouth', or 'full'
        output_dir: Directory to save preview images
    """
    # Get partition info
    nodes = get_partition_nodes(partition)
    adjacency, node_mapping = build_partition_adjacency(partition)
    
    # Get AU groups
    au_groups = get_au_node_groups(partition, node_mapping)
    
    # Get sample landmarks
    landmarks = get_sample_landmarks(partition, n_frames=1)
    landmarks_np = landmarks[0].numpy()  # (n_nodes, 2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    # Color map for AU groups
    au_colors = {
        'AU25_lips_part': 'red',
        'AU26_jaw_drop': 'blue',
        'AU12_lip_corner': 'green',
        'AU27_mouth_stretch': 'orange',
    }
    
    # Plot 1: All AU groups together
    ax = axes[0]
    ax.set_title(f'{partition.upper()} Partition - All AU Groups', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Image coordinates
    
    # Plot all nodes in gray
    ax.scatter(landmarks_np[:, 0], landmarks_np[:, 1], 
               c='lightgray', s=50, alpha=0.3, label='Other nodes')
    
    # Plot each AU group
    for au_name, node_indices in au_groups.items():
        if len(node_indices) > 0:
            au_nodes_coords = landmarks_np[node_indices]
            color = au_colors.get(au_name, 'purple')
            ax.scatter(au_nodes_coords[:, 0], au_nodes_coords[:, 1],
                      c=color, s=100, alpha=0.7, label=au_name, edgecolors='black', linewidths=1)
            # Add node indices
            for idx in node_indices[:10]:  # Show first 10 indices to avoid clutter
                ax.annotate(str(idx), (landmarks_np[idx, 0], landmarks_np[idx, 1]),
                           fontsize=6, ha='center', va='center')
    
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2-4: Individual AU groups
    plot_idx = 1
    for au_name, node_indices in au_groups.items():
        if len(node_indices) > 0 and plot_idx < 4:
            ax = axes[plot_idx]
            color = au_colors.get(au_name, 'purple')
            
            ax.set_title(f'{au_name}\n({len(node_indices)} nodes)', 
                        fontsize=12, fontweight='bold', color=color)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Plot all nodes in gray
            ax.scatter(landmarks_np[:, 0], landmarks_np[:, 1],
                      c='lightgray', s=30, alpha=0.2)
            
            # Highlight AU group nodes
            au_nodes_coords = landmarks_np[node_indices]
            ax.scatter(au_nodes_coords[:, 0], au_nodes_coords[:, 1],
                      c=color, s=150, alpha=0.8, edgecolors='black', linewidths=2)
            
            # Add node indices
            for idx in node_indices:
                original_node = nodes[idx]  # Get original MediaPipe index
                ax.annotate(f'{idx}\n({original_node})', 
                           (landmarks_np[idx, 0], landmarks_np[idx, 1]),
                           fontsize=7, ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # Draw adjacency edges for this group
            for i, node_i in enumerate(node_indices):
                for j, node_j in enumerate(node_indices):
                    if i < j and adjacency[node_i, node_j] > 0:
                        ax.plot([landmarks_np[node_i, 0], landmarks_np[node_j, 0]],
                               [landmarks_np[node_i, 1], landmarks_np[node_j, 1]],
                               'k-', alpha=0.2, linewidth=0.5)
            
            ax.grid(True, alpha=0.3)
            plot_idx += 1
    
    plt.tight_layout()
    output_path = output_dir / f'{partition}_au_groups.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved AU groups visualization: {output_path}")
    return output_path


def visualize_geometric_anchors(partition: str, output_dir: Path):
    """
    Visualize geometric feature anchor nodes (used in B2).
    
    Args:
        partition: 'lips', 'mouth', or 'full'
        output_dir: Directory to save preview images
    """
    nodes = get_partition_nodes(partition)
    n_nodes = len(nodes)
    
    # Get sample landmarks
    landmarks = get_sample_landmarks(partition, n_frames=1)
    landmarks_np = landmarks[0].numpy()
    
    # Geometric features use 1 anchor (reduced from 5 to 1 for memory optimization)
    # For full partition: use nose tip (MediaPipe landmark 4, remapped index 4) - stable central reference
    # For lips/mouth partitions: use node 0 (nose tip not available)
    nose_tip_mp = 4  # MediaPipe nose tip landmark
    
    if partition == 'full' and nose_tip_mp in nodes:
        # Use nose tip as anchor for full partition (more stable, central reference)
        anchor_remapped_idx = nodes.index(nose_tip_mp)
        anchor_label = "Nose Tip (MP 4)"
    else:
        # Use node 0 for lips/mouth partitions (nose tip not available)
        anchor_remapped_idx = 0
        anchor_label = f"Node 0 (MP {nodes[0]})"
    
    anchor_indices = [anchor_remapped_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    title = f'{partition.upper()} Partition - Geometric Feature Anchor Nodes (B2)\n(1 anchor: {anchor_label})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Plot all nodes in gray
    ax.scatter(landmarks_np[:, 0], landmarks_np[:, 1],
              c='lightgray', s=50, alpha=0.3, label='Other nodes')
    
    # Highlight anchor nodes
    anchor_coords = landmarks_np[anchor_indices]
    ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1],
              c='red', s=200, alpha=0.8, label='Anchor nodes (for distance features)',
              edgecolors='black', linewidths=2, marker='s')
    
    # Add labels
    for idx in anchor_indices:
        original_node = nodes[idx]
        ax.annotate(f'Anchor {idx}\n(MP {original_node})',
                   (landmarks_np[idx, 0], landmarks_np[idx, 1]),
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Draw lines from anchor to all other nodes (sample)
    for anchor_idx in anchor_indices:
        # Show connections to a sample of other nodes
        sample_indices = [i for i in range(n_nodes) if i != anchor_idx][:20]
        for other_idx in sample_indices:
            ax.plot([landmarks_np[anchor_idx, 0], landmarks_np[other_idx, 0]],
                   [landmarks_np[anchor_idx, 1], landmarks_np[other_idx, 1]],
                   'r--', alpha=0.2, linewidth=0.5)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{partition}_geometric_anchors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved geometric anchors visualization: {output_path}")
    return output_path


def visualize_all_features(partition: str, output_dir: Path):
    """
    Create a comprehensive visualization showing all feature engineering node assignments.
    
    Args:
        partition: 'lips', 'mouth', or 'full'
        output_dir: Directory to save preview images
    """
    nodes = get_partition_nodes(partition)
    adjacency, node_mapping = build_partition_adjacency(partition)
    au_groups = get_au_node_groups(partition, node_mapping)
    
    landmarks = get_sample_landmarks(partition, n_frames=1)
    landmarks_np = landmarks[0].numpy()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    # Main plot: All feature assignments
    ax_main = plt.subplot(1, 2, 1)
    ax_main.set_title(f'{partition.upper()} Partition - All Feature Engineering Nodes', 
                     fontsize=16, fontweight='bold')
    ax_main.set_aspect('equal')
    ax_main.invert_yaxis()
    
    # Color scheme
    au_colors = {
        'AU25_lips_part': 'red',
        'AU26_jaw_drop': 'blue',
        'AU12_lip_corner': 'green',
        'AU27_mouth_stretch': 'orange',
    }
    
    # Plot all nodes
    ax_main.scatter(landmarks_np[:, 0], landmarks_np[:, 1],
                   c='lightgray', s=30, alpha=0.2, label='Unassigned nodes')
    
    # Plot AU groups
    for au_name, node_indices in au_groups.items():
        if len(node_indices) > 0:
            au_coords = landmarks_np[node_indices]
            color = au_colors.get(au_name, 'purple')
            ax_main.scatter(au_coords[:, 0], au_coords[:, 1],
                          c=color, s=100, alpha=0.7, label=au_name, 
                          edgecolors='black', linewidths=1)
    
    # Plot geometric anchors
    # For full partition: use nose tip (MediaPipe landmark 4, remapped index 4)
    # For lips/mouth partitions: use node 0 (nose tip not available)
    nose_tip_mp = 4  # MediaPipe nose tip landmark
    if partition == 'full' and nose_tip_mp in nodes:
        anchor_remapped_idx = nodes.index(nose_tip_mp)
        anchor_label = "Nose Tip (MP 4)"
    else:
        anchor_remapped_idx = 0
        anchor_label = f"Node 0 (MP {nodes[0]})"
    
    anchor_indices = [anchor_remapped_idx]
    anchor_coords = landmarks_np[anchor_indices]
    ax_main.scatter(anchor_coords[:, 0], anchor_coords[:, 1],
                   c='yellow', s=150, alpha=0.9, label=f'Geometric anchor (B2): {anchor_label}',
                   edgecolors='black', linewidths=2, marker='s')
    
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(True, alpha=0.3)
    
    # Legend/Info panel
    ax_info = plt.subplot(1, 2, 2)
    ax_info.axis('off')
    
    info_text = f"Feature Engineering Node Assignments\n"
    info_text += f"Partition: {partition.upper()}\n"
    info_text += f"Total nodes: {len(nodes)}\n\n"
    
    info_text += "AU Groups (B3):\n"
    for au_name, node_indices in au_groups.items():
        if len(node_indices) > 0:
            info_text += f"  • {au_name}: {len(node_indices)} nodes\n"
    
    info_text += f"\nGeometric Features (B2):\n"
    info_text += f"  • Anchor node: {anchor_label} (remapped index {anchor_remapped_idx})\n"
    info_text += f"  • Distance features: computed from anchor to all nodes\n"
    info_text += f"  • Angle features: computed from consecutive nodes\n"
    info_text += f"  • Ratio features: removed for memory optimization\n"
    
    info_text += f"\nFeature Counts:\n"
    info_text += f"  • B0: 2 features (x, y coordinates)\n"
    info_text += f"  • B1: 3 features (vx, vy, speed) - acceleration removed\n"
    info_text += f"  • B2: 2 features (1 distance + 1 angle) - ratio removed, 1 anchor\n"
    info_text += f"  • B3: 4 features (4 AU only) - PCA and motion energy removed\n"
    info_text += f"  • Total: 11 features per node (54% reduction)\n"
    
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f'{partition}_all_feature_nodes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive feature visualization: {output_path}")
    return output_path


def main():
    """Generate all feature engineering visualizations."""
    partitions = ['lips', 'mouth', 'full']
    
    for partition in partitions:
        print(f"\n{'='*60}")
        print(f"Visualizing {partition.upper()} partition")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir = Path(f'data/extracted/{partition}/previews')
        ensure_dir(output_dir)
        
        try:
            # Visualize AU groups
            visualize_au_groups(partition, output_dir)
            
            # Visualize geometric anchors
            visualize_geometric_anchors(partition, output_dir)
            
            # Comprehensive visualization
            visualize_all_features(partition, output_dir)
            
            print(f"✓ Completed {partition} partition")
        except Exception as e:
            print(f"✗ Error visualizing {partition}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

