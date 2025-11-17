"""
Model explainability tools - visualize what the model focuses on
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset import LipReadingDataset
from src.models.combined import CombinedModel
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explain model predictions through various techniques"""
    
    def __init__(self, model: nn.Module, device: torch.device, label_map: Dict[str, int]):
        """
        Args:
            model: Trained model
            device: Device
            label_map: Label to index mapping
        """
        self.model = model.to(device)
        self.device = device
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.model.eval()
    
    def gradient_based_saliency(self, data, target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute gradient-based saliency map
        
        Args:
            data: Input data (PyG Data object)
            target_class: Target class for gradient (None = use predicted class)
            
        Returns:
            Saliency scores for each node at each timestep [T, N]
        """
        data = data.to(self.device)
        x_temporal = data.x_temporal
        
        # Ensure x_temporal is on device and has gradient
        if isinstance(x_temporal, (list, tuple)):
            x_temporal = torch.stack(x_temporal, dim=0).to(self.device)
        x_temporal = x_temporal.requires_grad_(True)
        
        # Forward pass
        logits = self.model(data)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Get gradients
        if x_temporal.grad is not None:
            saliency = torch.abs(x_temporal.grad).sum(dim=-1)  # [T, N] or [batch, T, N]
            if saliency.dim() == 3:
                saliency = saliency[0]  # Remove batch dimension
            return saliency.detach().cpu().numpy()
        else:
            return np.zeros((x_temporal.shape[-3], x_temporal.shape[-2]))
    
    def occlusion_sensitivity(self, data, target_class: Optional[int] = None,
                            region_size: int = 5) -> np.ndarray:
        """
        Compute occlusion sensitivity - mask regions and see impact on prediction
        
        Args:
            data: Input data
            target_class: Target class (None = use predicted class)
            region_size: Number of nodes to occlude at once
            
        Returns:
            Sensitivity scores [T, N]
        """
        data = data.to(self.device)
        original_logits = self.model(data)
        
        if target_class is None:
            target_class = original_logits.argmax(dim=1).item()
        
        original_prob = F.softmax(original_logits, dim=1)[0, target_class].item()
        
        T, N, _ = data.x_temporal.shape if len(data.x_temporal.shape) == 3 else data.x_temporal[0].shape
        
        sensitivity = np.zeros((T, N))
        
        for t in range(T):
            for n_start in range(0, N, region_size):
                n_end = min(n_start + region_size, N)
                
                # Create occluded data
                occluded_data = self._occlude_region(data, t, n_start, n_end)
                
                # Forward pass
                occluded_logits = self.model(occluded_data)
                occluded_prob = F.softmax(occluded_logits, dim=1)[0, target_class].item()
                
                # Sensitivity = drop in probability
                sensitivity[t, n_start:n_end] = original_prob - occluded_prob
        
        return sensitivity
    
    def _occlude_region(self, data, t: int, n_start: int, n_end: int):
        """Occlude a region by setting it to zero"""
        data_copy = data.clone()
        x_temporal = data_copy.x_temporal
        
        if isinstance(x_temporal, (list, tuple)):
            x_temporal = torch.stack(x_temporal, dim=0)
        
        x_temporal[t, n_start:n_end, :] = 0.0
        data_copy.x_temporal = x_temporal
        
        return data_copy
    
    def visualize_saliency(self, data, saliency: np.ndarray, output_path: Path,
                           title: str = "Saliency Map"):
        """Visualize saliency map"""
        T, N = saliency.shape
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Temporal saliency (sum over nodes)
        temporal_saliency = saliency.sum(axis=1)
        axes[0].plot(temporal_saliency, 'b-', linewidth=2)
        axes[0].fill_between(range(T), temporal_saliency, alpha=0.3)
        axes[0].set_xlabel('Frame', fontsize=12)
        axes[0].set_ylabel('Saliency (sum over nodes)', fontsize=12)
        axes[0].set_title('Temporal Saliency', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Spatial saliency (sum over time)
        spatial_saliency = saliency.sum(axis=0)
        axes[1].bar(range(N), spatial_saliency, alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Node Index', fontsize=12)
        axes[1].set_ylabel('Saliency (sum over time)', fontsize=12)
        axes[1].set_title('Spatial Saliency', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_heatmap(self, data, saliency: np.ndarray, output_path: Path,
                          title: str = "Saliency Heatmap"):
        """Visualize saliency as heatmap"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(saliency, cmap='YlOrRd', cbar_kws={'label': 'Saliency'},
                   xticklabels=False, yticklabels=False, ax=ax)
        
        ax.set_xlabel('Node Index', fontsize=12)
        ax.set_ylabel('Frame', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze_prediction(self, data, output_dir: Path, sample_idx: int = 0):
        """
        Comprehensive analysis of a single prediction
        
        Args:
            data: Input data
            output_dir: Output directory
            sample_idx: Sample index for naming
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = data.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(data)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        true_label = self.idx_to_label.get(data.y.item(), "Unknown")
        pred_label = self.idx_to_label.get(pred_class, "Unknown")
        
        logger.info(f"📊 Analyzing prediction:")
        logger.info(f"   True: {true_label}")
        logger.info(f"   Pred: {pred_label} ({pred_prob:.2%})")
        
        # Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs[0], k=min(5, len(self.idx_to_label)))
        
        # Gradient-based saliency
        logger.info("   Computing gradient saliency...")
        saliency = self.gradient_based_saliency(data, target_class=pred_class)
        
        # Visualize
        self.visualize_saliency(data, saliency, 
                               output_dir / f"sample_{sample_idx}_saliency.png",
                               title=f"Saliency: {true_label} → {pred_label}")
        
        self.visualize_heatmap(data, saliency,
                              output_dir / f"sample_{sample_idx}_heatmap.png",
                              title=f"Saliency Heatmap: {true_label} → {pred_label}")
        
        # Prediction distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        top5_labels = [self.idx_to_label[idx.item()] for idx in top5_indices]
        top5_probs_np = top5_probs.cpu().numpy()
        
        colors = ['green' if label == true_label else 'red' if label == pred_label else 'gray'
                  for label in top5_labels]
        ax.barh(top5_labels, top5_probs_np, color=colors, alpha=0.7)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(f'Top-5 Predictions (True: {true_label}, Pred: {pred_label})', fontsize=14)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{sample_idx}_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'true_label': true_label,
            'pred_label': pred_label,
            'pred_prob': pred_prob,
            'top5': {top5_labels[i]: float(top5_probs_np[i]) for i in range(len(top5_labels))},
            'saliency_stats': {
                'mean': float(saliency.mean()),
                'std': float(saliency.std()),
                'max': float(saliency.max()),
                'temporal_peak': int(saliency.sum(axis=1).argmax()),
                'spatial_peak': int(saliency.sum(axis=0).argmax())
            }
        }


def explain_predictions(checkpoint_path: str, config: Dict, test_pt: str,
                        output_dir: Path, num_samples: int = 10,
                        label_map: Dict[str, int] = None):
    """
    Explain predictions for multiple samples
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        test_pt: Path to test.pt file
        output_dir: Output directory
        num_samples: Number of samples to analyze
        label_map: Label mapping
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Load dataset
    from dataset.dataset import LipReadingDataset
    test_dataset = LipReadingDataset(test_pt, label_map=label_map)
    if label_map is None:
        label_map = test_dataset.label_map
    
    # Load model
    logger.info(f"🤖 Loading model from {checkpoint_path}")
    model = CombinedModel(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        spatial_dim=config.get('spatial_dim', 256),
        temporal_dim=config.get('temporal_dim', 256),
        spatial_layers=config.get('spatial_layers', 3),
        temporal_layers=config.get('temporal_layers', 2),
        dropout=config.get('dropout', 0.5)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create explainer
    explainer = ModelExplainer(model, device, label_map)
    
    # Analyze samples
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    results = []
    for i, idx in enumerate(indices):
        logger.info(f"\n📊 Analyzing sample {i+1}/{len(indices)} (index {idx})")
        data = test_dataset[idx]
        result = explainer.analyze_prediction(data, output_dir, sample_idx=idx)
        results.append(result)
    
    # Save summary
    import json
    with open(output_dir / 'explanation_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Saved explanations to {output_dir}")
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model explainability analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_pt', type=str, required=True,
                       help='Path to test.pt file')
    parser.add_argument('--output', type=str, default='debug_outputs/explanations',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to analyze')
    parser.add_argument('--input_dim', type=int, default=31,
                       help='Input dimension')
    
    args = parser.parse_args()
    
    # Load label map
    from src.dataset.dataset import LipReadingDataset
    test_dataset = LipReadingDataset(args.test_pt)
    num_classes = test_dataset.num_classes
    label_map = test_dataset.label_map
    
    config = {
        'input_dim': args.input_dim,
        'num_classes': num_classes,
        'spatial_dim': 256,
        'temporal_dim': 256,
        'spatial_layers': 3,
        'temporal_layers': 2,
        'dropout': 0.5
    }
    
    explain_predictions(args.checkpoint, config, args.test_pt, args.output,
                       args.num_samples, label_map)


if __name__ == '__main__':
    main()

