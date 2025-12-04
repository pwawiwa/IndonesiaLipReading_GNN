"""
Evaluation utilities for trained models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


class Evaluator:
    """Model evaluator with metrics and visualization."""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cuda',
        class_names: List[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to use
            class_names: List of class names
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names or [str(i) for i in range(model.num_classes)]
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Evaluate model and compute metrics.
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        # Disable tqdm progress bar - only log at completion
        for features, speech_mask, adj, labels in tqdm(self.dataloader, desc="Evaluating", disable=True):
            features = features.to(self.device)
            speech_mask = speech_mask.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(features, adj, speech_mask)
            loss = criterion(outputs, labels)
            
            # Predictions
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
        avg_loss = total_loss / len(self.dataloader)
        
        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_preds,
            'labels': all_labels,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return results
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Path,
        normalize: bool = True,
        figsize: Tuple[int, int] = (20, 18)
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save figure
            normalize: Normalize by row
            figsize: Figure size
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        plt.figure(figsize=figsize)
        
        # For 100 classes, use smaller font and no labels
        if len(self.class_names) > 50:
            sns.heatmap(
                cm,
                annot=False,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                square=True,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
            )
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
        else:
            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                square=True,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
            )
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
        
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_classification_report(
        self,
        report: Dict,
        save_path: Path,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot classification report as heatmap.
        
        Args:
            report: Classification report dictionary
            save_path: Path to save figure
            figsize: Figure size
        """
        # Extract per-class metrics
        classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # Limit to first 50 classes for readability
        if len(classes) > 50:
            classes = classes[:50]
        
        metrics = ['precision', 'recall', 'f1-score']
        data = np.array([[report[c][m] for m in metrics] for c in classes])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=metrics,
            yticklabels=classes,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Score'}
        )
        plt.title('Per-Class Metrics (First 50 Classes)', fontsize=14, pad=20)
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: Path,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], label='Train', marker='o', markersize=3)
        axes[0].plot(epochs, history['val_loss'], label='Val', marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], label='Train', marker='o', markersize=3)
        axes[1].plot(epochs, history['val_acc'], label='Val', marker='s', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy History')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[2].plot(epochs, history['lr'], marker='o', markersize=3, color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_model(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: str,
    model_class,
    device: str = 'cuda'
):
    """
    Evaluate a trained model and generate all visualizations.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data
        output_dir: Output directory for results
        model_class: Model class to instantiate
        device: Device to use
    """
    from training import get_dataloader
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    
    # Load test data
    test_loader = get_dataloader(test_data_path, batch_size=32, shuffle=False)
    
    # Get class names
    dataset = test_loader.dataset
    word_to_label = dataset.word_to_label
    class_names = sorted(word_to_label.keys(), key=lambda x: word_to_label[x])
    
    # Build model
    model = model_class(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, device, class_names)
    results = evaluator.evaluate()
    
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Save results
    torch.save({
        'accuracy': results['accuracy'],
        'loss': results['loss'],
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix']
    }, output_dir / 'test_results.pt')
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        output_dir / 'confusion_matrix.png'
    )
    
    # Plot classification report
    evaluator.plot_classification_report(
        results['classification_report'],
        output_dir / 'classification_report.png'
    )
    
    print(f"Saved results to {output_dir}")

