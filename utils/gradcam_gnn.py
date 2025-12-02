"""
GradCAM for Graph Neural Networks.

Visualizes which nodes (facial landmarks) are most important for model predictions.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns


class GradCAMGNN:
    """
    GradCAM (Gradient-weighted Class Activation Mapping) for GNN models.
    
    Shows which nodes contribute most to the model's decision.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM.
        
        Args:
            model: Trained GNN model
            target_layer: Layer to compute gradients from (default: last GNN layer)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer (last GNN conv layer by default)
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self):
        """Find the last GNN convolutional layer."""
        # Look for common GNN layer names
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['conv', 'gat', 'sage', 'gin', 'gcn']):
                last_conv = module
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        target_class: int,
        batch_idx: int = 0
    ) -> np.ndarray:
        """
        Generate Class Activation Map for a sample.
        
        Args:
            features: Node features (batch, frames, nodes, features)
            adjacency: Adjacency matrix (nodes, nodes)
            target_class: Target class for CAM
            batch_idx: Batch index
            
        Returns:
            cam: Node importance scores (nodes,) in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(features, adjacency)
        
        # Get score for target class
        target_score = output[batch_idx, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[batch_idx]  # (nodes, features)
        activations = self.activations[batch_idx]  # (nodes, features)
        
        # Compute weights (global average pooling over features)
        weights = torch.mean(gradients, dim=-1, keepdim=True)  # (nodes, 1)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=-1)  # (nodes,)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_cam_for_sequence(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Generate averaged CAM across temporal sequence.
        
        Args:
            features: Node features (1, frames, nodes, features)
            adjacency: Adjacency matrix (nodes, nodes)
            target_class: Target class
            
        Returns:
            cam: Averaged node importance (nodes,)
        """
        return self.generate_cam(features, adjacency, target_class, batch_idx=0)


def visualize_cam_on_face(
    video_path: str,
    landmarks: torch.Tensor,
    cam_scores: np.ndarray,
    frame_idx: int,
    word: str,
    predicted_class: str,
    confidence: float,
    output_path: str,
    colormap: str = 'jet'
):
    """
    Visualize CAM scores on a face image.
    
    Args:
        video_path: Path to video file
        landmarks: Landmark coordinates (frames, nodes, 2) normalized
        cam_scores: CAM scores per node (nodes,)
        frame_idx: Frame index to visualize
        word: Ground truth word
        predicted_class: Predicted word
        confidence: Prediction confidence
        output_path: Path to save visualization
        colormap: Matplotlib colormap name
    """
    # Read frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame {frame_idx}")
        return
    
    h, w = frame.shape[:2]
    
    # Convert landmarks to pixel coordinates
    pixel_coords = landmarks[frame_idx].numpy()
    pixel_coords[:, 0] = pixel_coords[:, 0] * w
    pixel_coords[:, 1] = pixel_coords[:, 1] * h
    pixel_coords = pixel_coords.astype(np.int32)
    
    # Create heatmap overlay
    overlay = frame.copy()
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Draw nodes with color based on importance
    for idx, (coord, score) in enumerate(zip(pixel_coords, cam_scores)):
        x, y = coord
        
        # Get color from colormap
        color_rgba = cmap(score)
        color_bgr = (
            int(color_rgba[2] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[0] * 255)
        )
        
        # Draw circle with size proportional to importance
        radius = int(3 + score * 5)
        cv2.circle(overlay, (x, y), radius, color_bgr, -1)
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)
    
    # Blend with original frame
    alpha = 0.6
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Add text information
    correct = "✓" if word == predicted_class else "✗"
    info_text = f"GT: {word} | Pred: {predicted_class} {correct} | Conf: {confidence:.2%}"
    
    cv2.putText(result, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 0, 0), 1, cv2.LINE_AA)
    
    # Add colorbar
    colorbar_height = 30
    colorbar_width = w // 3
    colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
    
    for i in range(colorbar_width):
        score_i = i / colorbar_width
        color_rgba = cmap(score_i)
        color_bgr = (
            int(color_rgba[2] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[0] * 255)
        )
        colorbar[:, i] = color_bgr
    
    # Place colorbar
    x_start = w - colorbar_width - 10
    y_start = h - colorbar_height - 10
    result[y_start:y_start + colorbar_height, x_start:x_start + colorbar_width] = colorbar
    
    cv2.putText(result, "Low", (x_start - 40, y_start + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, "High", (x_start + colorbar_width + 5, y_start + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save
    cv2.imwrite(output_path, result)


def aggregate_cam_per_class(
    cam_scores_list: List[np.ndarray],
    method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate CAM scores across multiple samples.
    
    Args:
        cam_scores_list: List of CAM scores (each is nodes,)
        method: 'mean', 'max', or 'median'
        
    Returns:
        aggregated_cam: Aggregated scores (nodes,)
    """
    cam_array = np.stack(cam_scores_list, axis=0)  # (samples, nodes)
    
    if method == 'mean':
        return np.mean(cam_array, axis=0)
    elif method == 'max':
        return np.max(cam_array, axis=0)
    elif method == 'median':
        return np.median(cam_array, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_aggregated_cam(
    aggregated_cam: np.ndarray,
    class_name: str,
    partition: str,
    output_path: str,
    top_k: int = 20
):
    """
    Visualize aggregated CAM scores as a bar plot.
    
    Args:
        aggregated_cam: Aggregated CAM scores (nodes,)
        class_name: Class name
        partition: Partition name (lips, mouth, full)
        output_path: Path to save plot
        top_k: Number of top nodes to highlight
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Top-K important nodes
    top_indices = np.argsort(aggregated_cam)[-top_k:][::-1]
    top_scores = aggregated_cam[top_indices]
    
    axes[0].barh(range(top_k), top_scores, color='steelblue')
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels([f"Node {idx}" for idx in top_indices])
    axes[0].set_xlabel('Importance Score')
    axes[0].set_title(f'Top {top_k} Most Important Nodes\n{class_name} ({partition})')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: All nodes heatmap
    n_nodes = len(aggregated_cam)
    if n_nodes <= 100:
        # Show all nodes
        im = axes[1].imshow(aggregated_cam.reshape(1, -1), aspect='auto', cmap='hot')
        axes[1].set_yticks([])
        axes[1].set_xlabel('Node Index')
        axes[1].set_title(f'All Node Importance ({n_nodes} nodes)')
        plt.colorbar(im, ax=axes[1], label='Importance')
    else:
        # Show histogram for large number of nodes
        axes[1].hist(aggregated_cam, bins=50, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Importance Score')
        axes[1].set_ylabel('Number of Nodes')
        axes[1].set_title(f'Node Importance Distribution ({n_nodes} nodes)')
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_cam_summary(
    cam_per_class: Dict[str, np.ndarray],
    output_path: str
):
    """
    Save CAM summary statistics.
    
    Args:
        cam_per_class: Dictionary mapping class names to aggregated CAM scores
        output_path: Path to save summary (.pt file)
    """
    summary = {
        'cam_per_class': cam_per_class,
        'top_nodes_per_class': {},
        'statistics': {}
    }
    
    for class_name, cam_scores in cam_per_class.items():
        # Top 20 nodes
        top_indices = np.argsort(cam_scores)[-20:][::-1]
        summary['top_nodes_per_class'][class_name] = {
            'indices': top_indices.tolist(),
            'scores': cam_scores[top_indices].tolist()
        }
        
        # Statistics
        summary['statistics'][class_name] = {
            'mean': float(np.mean(cam_scores)),
            'std': float(np.std(cam_scores)),
            'max': float(np.max(cam_scores)),
            'min': float(np.min(cam_scores))
        }
    
    torch.save(summary, output_path)

