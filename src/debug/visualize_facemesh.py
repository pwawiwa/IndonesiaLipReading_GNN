"""
Visualize facemesh extraction to debug extraction quality
"""
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Optional, Dict, List
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.facemesh_extractor import FaceMeshExtractor
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def visualize_extracted_landmarks(video_path: str, output_dir: Path, num_frames: int = 10):
    """
    Visualize extracted landmarks from a video
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for visualizations
        num_frames: Number of frames to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    extractor = FaceMeshExtractor()
    
    # Initialize MediaPipe
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_idx = 0
    saved_frames = 0
    
    print(f"📹 Processing video: {video_path}")
    print(f"   Total frames: {total_frames}, FPS: {fps}")
    print(f"   Will save {num_frames} frames")
    
    while cap.isOpened() and saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            results = face_mesh.process(rgb_frame)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: Original with MediaPipe overlay
            axes[0].imshow(rgb_frame)
            axes[0].set_title(f'Frame {frame_idx} - MediaPipe Detection', fontsize=14)
            axes[0].axis('off')
            
            if results.multi_face_landmarks:
                # Draw MediaPipe landmarks
                for face_landmarks in results.multi_face_landmarks:
                    # Draw tesselation
                    mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Draw contours
                    mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                axes[0].imshow(rgb_frame)
                
                # Extract ROI landmarks
                face_landmarks = results.multi_face_landmarks[0]
                full_landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                ], dtype=np.float32)
                
                roi_landmarks = full_landmarks[extractor.ROI_INDICES]
                normalized = extractor.normalize_landmarks(roi_landmarks)
                
                # Right: ROI landmarks visualization
                h, w = frame.shape[:2]
                roi_2d = normalized[:, :2] * np.array([w, h])
                
                axes[1].scatter(roi_2d[:, 0], roi_2d[:, 1], c='red', s=50, alpha=0.7)
                
                # Draw edges
                for edge_pair in extractor.EDGE_PAIRS_ORIGINAL:
                    src_orig, dst_orig = edge_pair
                    if src_orig in extractor.ROI_INDICES and dst_orig in extractor.ROI_INDICES:
                        src_roi_idx = extractor.ROI_INDICES.index(src_orig)
                        dst_roi_idx = extractor.ROI_INDICES.index(dst_orig)
                        src_pos = roi_2d[src_roi_idx]
                        dst_pos = roi_2d[dst_roi_idx]
                        axes[1].plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], 
                                   'b-', alpha=0.3, linewidth=0.5)
                
                axes[1].set_title(f'Frame {frame_idx} - Extracted ROI ({len(roi_landmarks)} landmarks)', 
                                 fontsize=14)
                axes[1].invert_yaxis()
                axes[1].set_aspect('equal')
                axes[1].grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f"ROI Landmarks: {len(roi_landmarks)}\n"
                stats_text += f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]\n"
                stats_text += f"X range: [{normalized[:, 0].min():.3f}, {normalized[:, 0].max():.3f}]\n"
                stats_text += f"Y range: [{normalized[:, 1].min():.3f}, {normalized[:, 1].max():.3f}]\n"
                stats_text += f"Z range: [{normalized[:, 2].min():.3f}, {normalized[:, 2].max():.3f}]"
                
                axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                axes[1].text(0.5, 0.5, 'No face detected', 
                           transform=axes[1].transAxes,
                           ha='center', va='center', fontsize=16, color='red')
                axes[1].set_title(f'Frame {frame_idx} - No Detection', fontsize=14)
            
            plt.tight_layout()
            output_path = output_dir / f"frame_{frame_idx:06d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_frames += 1
            print(f"   ✓ Saved frame {frame_idx} ({saved_frames}/{num_frames})")
        
        frame_idx += 1
    
    cap.release()
    print(f"✅ Saved {saved_frames} visualizations to {output_dir}")


def visualize_from_pt(pt_path: str, output_dir: Path, num_samples: int = 5, sample_indices: Optional[List[int]] = None):
    """
    Visualize extracted landmarks from processed .pt file
    
    Args:
        pt_path: Path to .pt file
        output_dir: Output directory
        num_samples: Number of samples to visualize
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
    
    extractor = FaceMeshExtractor()
    
    for idx in indices:
        sample = samples[idx]
        landmarks = sample['landmarks']  # [T, N, 3]
        label = sample['label']
        video_id = sample.get('video_id', f'sample_{idx}')
        
        T, N, _ = landmarks.shape
        
        print(f"\n📊 Sample {idx}: {label} ({video_id})")
        print(f"   Sequence length: {T} frames")
        print(f"   Landmarks per frame: {N}")
        
        # Visualize first, middle, and last frames
        frame_indices = [0, T//2, T-1] if T > 2 else [0]
        
        fig, axes = plt.subplots(len(frame_indices), 2, figsize=(16, 6*len(frame_indices)))
        if len(frame_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for row, t in enumerate(frame_indices):
            frame_landmarks = landmarks[t]  # [N, 3]
            
            # Left: 2D projection
            axes[row, 0].scatter(frame_landmarks[:, 0], frame_landmarks[:, 1], 
                               c='red', s=30, alpha=0.7)
            
            # Draw edges
            edge_index = sample.get('edge_index', None)
            if edge_index is not None:
                edge_index_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
                for i in range(edge_index_np.shape[1]):
                    src, dst = edge_index_np[:, i]
                    if src < N and dst < N:
                        axes[row, 0].plot([frame_landmarks[src, 0], frame_landmarks[dst, 0]],
                                        [frame_landmarks[src, 1], frame_landmarks[dst, 1]],
                                        'b-', alpha=0.2, linewidth=0.5)
            
            axes[row, 0].set_title(f'Frame {t} - 2D Projection', fontsize=12)
            axes[row, 0].invert_yaxis()
            axes[row, 0].set_aspect('equal')
            axes[row, 0].grid(True, alpha=0.3)
            
            # Right: 3D visualization
            ax_3d = axes[row, 1]
            scatter = ax_3d.scatter(frame_landmarks[:, 0], frame_landmarks[:, 1], 
                                   frame_landmarks[:, 2], c=frame_landmarks[:, 2],
                                   cmap='viridis', s=30, alpha=0.7)
            
            # Draw edges in 3D
            if edge_index is not None:
                edge_index_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
                for i in range(min(100, edge_index_np.shape[1])):  # Limit for performance
                    src, dst = edge_index_np[:, i]
                    if src < N and dst < N:
                        ax_3d.plot([frame_landmarks[src, 0], frame_landmarks[dst, 0]],
                                 [frame_landmarks[src, 1], frame_landmarks[dst, 1]],
                                 [frame_landmarks[src, 2], frame_landmarks[dst, 2]],
                                 'b-', alpha=0.1, linewidth=0.3)
            
            ax_3d.set_title(f'Frame {t} - 3D Visualization', fontsize=12)
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            plt.colorbar(scatter, ax=ax_3d, label='Z value')
        
        plt.suptitle(f'Sample {idx}: {label} ({video_id})', fontsize=16, y=0.995)
        plt.tight_layout()
        
        output_path = output_dir / f"sample_{idx}_{label.replace('/', '_')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved visualization to {output_path}")
    
    print(f"\n✅ Saved {len(indices)} sample visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize facemesh extraction')
    parser.add_argument('--mode', type=str, choices=['video', 'pt'], required=True,
                       help='Visualization mode: video or pt')
    parser.add_argument('--input', type=str, required=True,
                       help='Input video path or .pt file path')
    parser.add_argument('--output', type=str, default='debug_outputs/facemesh_viz',
                       help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to visualize (for video mode)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (for pt mode)')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                       help='Specific sample indices to visualize (for pt mode)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'video':
        visualize_extracted_landmarks(args.input, output_dir, args.num_frames)
    elif args.mode == 'pt':
        visualize_from_pt(args.input, output_dir, args.num_samples, args.sample_indices)


if __name__ == '__main__':
    main()

