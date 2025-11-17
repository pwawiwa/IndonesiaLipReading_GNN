"""
Enhanced evaluation with per-word accuracy and detailed metrics
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset import LipReadingDataset
from src.models.combined import CombinedModel
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class EnhancedEvaluator:
    """Enhanced evaluator with per-word metrics"""
    
    def __init__(self, model: nn.Module, dataloader, device: torch.device,
                 label_map: Dict[str, int], idx_to_label: Dict[int, str]):
        """
        Args:
            model: Trained model
            dataloader: DataLoader for evaluation
            device: Device
            label_map: Label to index mapping
            idx_to_label: Index to label mapping
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.label_map = label_map
        self.idx_to_label = idx_to_label
        self.num_classes = len(label_map)
    
    def evaluate(self) -> Dict:
        """Run comprehensive evaluation"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_logits = []
        all_video_ids = []
        all_confidences = []
        
        logger.info("🔍 Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                confidences = probs.max(dim=1)[0]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                # Get video IDs if available
                if hasattr(batch, 'video_id'):
                    if isinstance(batch.video_id, (list, tuple)):
                        all_video_ids.extend(batch.video_id)
                    else:
                        all_video_ids.extend([batch.video_id] * len(preds))
                else:
                    all_video_ids.extend([None] * len(preds))
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        all_confidences = np.array(all_confidences)
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics(all_labels, all_preds)
        
        # Per-word metrics
        per_word_metrics = self._calculate_per_word_metrics(all_labels, all_preds, all_confidences)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
        
        # Error analysis
        error_analysis = self._analyze_errors(all_labels, all_preds, all_confidences, all_video_ids)
        
        return {
            'overall': overall_metrics,
            'per_word': per_word_metrics,
            'confusion_matrix': cm.tolist(),
            'error_analysis': error_analysis,
            'all_predictions': {
                'labels': all_labels.tolist(),
                'preds': all_preds.tolist(),
                'confidences': all_confidences.tolist(),
                'video_ids': all_video_ids
            }
        }
    
    def _calculate_overall_metrics(self, labels: np.ndarray, preds: np.ndarray) -> Dict:
        """Calculate overall metrics"""
        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'precision_macro': float(precision_score(labels, preds, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(labels, preds, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(labels, preds, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(labels, preds, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        }
    
    def _calculate_per_word_metrics(self, labels: np.ndarray, preds: np.ndarray, 
                                     confidences: np.ndarray) -> Dict:
        """Calculate per-word metrics"""
        per_word = {}
        
        for class_idx in range(self.num_classes):
            word = self.idx_to_label[class_idx]
            
            # Get samples for this word
            mask = labels == class_idx
            if mask.sum() == 0:
                continue
            
            word_labels = labels[mask]
            word_preds = preds[mask]
            word_confidences = confidences[mask]
            
            # Metrics
            correct = (word_labels == word_preds).sum()
            total = len(word_labels)
            accuracy = correct / total if total > 0 else 0.0
            
            # Precision, recall, F1 for this class
            precision = precision_score(labels, preds, labels=[class_idx], average='macro', zero_division=0)
            recall = recall_score(labels, preds, labels=[class_idx], average='macro', zero_division=0)
            f1 = f1_score(labels, preds, labels=[class_idx], average='macro', zero_division=0)
            
            # Confidence statistics
            mean_confidence = float(word_confidences.mean())
            std_confidence = float(word_confidences.std())
            
            # Most common confusion
            incorrect_mask = word_labels != word_preds
            if incorrect_mask.sum() > 0:
                incorrect_preds = word_preds[incorrect_mask]
                unique, counts = np.unique(incorrect_preds, return_counts=True)
                most_confused = self.idx_to_label[int(unique[np.argmax(counts)])]
                confusion_count = int(counts.max())
            else:
                most_confused = None
                confusion_count = 0
            
            per_word[word] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'correct': int(correct),
                'total': int(total),
                'mean_confidence': mean_confidence,
                'std_confidence': std_confidence,
                'most_confused_with': most_confused,
                'confusion_count': confusion_count
            }
        
        return per_word
    
    def _analyze_errors(self, labels: np.ndarray, preds: np.ndarray,
                       confidences: np.ndarray, video_ids: List) -> Dict:
        """Analyze prediction errors"""
        errors = []
        correct = []
        
        for i in range(len(labels)):
            if labels[i] != preds[i]:
                errors.append({
                    'true_label': self.idx_to_label[int(labels[i])],
                    'pred_label': self.idx_to_label[int(preds[i])],
                    'confidence': float(confidences[i]),
                    'video_id': video_ids[i] if i < len(video_ids) else None
                })
            else:
                correct.append({
                    'label': self.idx_to_label[int(labels[i])],
                    'confidence': float(confidences[i]),
                    'video_id': video_ids[i] if i < len(video_ids) else None
                })
        
        # High confidence errors (most concerning)
        high_conf_errors = [e for e in errors if e['confidence'] > 0.7]
        
        # Low confidence correct (uncertain but right)
        low_conf_correct = [c for c in correct if c['confidence'] < 0.5]
        
        return {
            'total_errors': len(errors),
            'total_correct': len(correct),
            'error_rate': len(errors) / len(labels) if len(labels) > 0 else 0.0,
            'high_confidence_errors': len(high_conf_errors),
            'low_confidence_correct': len(low_conf_correct),
            'sample_errors': errors[:20],  # Sample of errors
            'sample_high_conf_errors': high_conf_errors[:10],
            'sample_low_conf_correct': low_conf_correct[:10]
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, output_path: Path):
        """Plot confusion matrix"""
        plt.figure(figsize=(max(12, self.num_classes * 0.5), max(10, self.num_classes * 0.5)))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Get labels
        labels = [self.idx_to_label[i] for i in range(self.num_classes)]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved confusion matrix to {output_path}")
    
    def plot_per_word_accuracy(self, per_word_metrics: Dict, output_path: Path):
        """Plot per-word accuracy"""
        words = list(per_word_metrics.keys())
        accuracies = [per_word_metrics[w]['accuracy'] for w in words]
        counts = [per_word_metrics[w]['total'] for w in words]
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        words_sorted = [words[i] for i in sorted_indices]
        accuracies_sorted = [accuracies[i] for i in sorted_indices]
        counts_sorted = [counts[i] for i in sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Accuracy bar plot
        colors = ['red' if acc < 0.3 else 'orange' if acc < 0.5 else 'green' 
                 for acc in accuracies_sorted]
        ax1.barh(words_sorted, accuracies_sorted, color=colors, alpha=0.7)
        ax1.set_xlabel('Accuracy', fontsize=12)
        ax1.set_title('Per-Word Accuracy', fontsize=14, pad=15)
        ax1.set_xlim([0, 1])
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.legend()
        
        # Sample count
        ax2.barh(words_sorted, counts_sorted, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Number of Samples', fontsize=12)
        ax2.set_title('Per-Word Sample Count', fontsize=14, pad=15)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved per-word accuracy plot to {output_path}")


def evaluate_model(checkpoint_path: str, config: Dict, test_pt: str,
                   output_dir: Path, label_map: Dict[str, int] = None):
    """
    Evaluate a model and generate comprehensive reports
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        test_pt: Path to test.pt file
        output_dir: Output directory for reports
        label_map: Label mapping (optional, will be inferred if None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Create dataloader
    logger.info("📦 Loading test dataset...")
    test_dataset = LipReadingDataset(test_pt, label_map=label_map)
    if label_map is None:
        label_map = test_dataset.label_map
    
    idx_to_label = {v: k for k, v in label_map.items()}
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
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
    
    # Evaluate
    evaluator = EnhancedEvaluator(model, test_loader, device, label_map, idx_to_label)
    results = evaluator.evaluate()
    
    # Save results
    results_json = output_dir / 'evaluation_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Saved results to {results_json}")
    
    # Generate plots
    cm = np.array(results['confusion_matrix'])
    evaluator.plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
    evaluator.plot_per_word_accuracy(results['per_word'], output_dir / 'per_word_accuracy.png')
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Overall Accuracy: {results['overall']['accuracy']:.4f}")
    logger.info(f"F1 Macro: {results['overall']['f1_macro']:.4f}")
    logger.info(f"F1 Weighted: {results['overall']['f1_weighted']:.4f}")
    logger.info(f"\nError Rate: {results['error_analysis']['error_rate']:.4f}")
    logger.info(f"High Confidence Errors: {results['error_analysis']['high_confidence_errors']}")
    logger.info(f"Low Confidence Correct: {results['error_analysis']['low_confidence_correct']}")
    
    # Worst performing words
    logger.info("\n📉 Worst Performing Words (Accuracy < 30%):")
    worst_words = [(w, m['accuracy']) for w, m in results['per_word'].items() if m['accuracy'] < 0.3]
    worst_words.sort(key=lambda x: x[1])
    for word, acc in worst_words[:10]:
        logger.info(f"  {word}: {acc:.2%} ({results['per_word'][word]['correct']}/{results['per_word'][word]['total']})")
    
    # Best performing words
    logger.info("\n📈 Best Performing Words (Accuracy > 70%):")
    best_words = [(w, m['accuracy']) for w, m in results['per_word'].items() if m['accuracy'] > 0.7]
    best_words.sort(key=lambda x: x[1], reverse=True)
    for word, acc in best_words[:10]:
        logger.info(f"  {word}: {acc:.2%} ({results['per_word'][word]['correct']}/{results['per_word'][word]['total']})")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced model evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_pt', type=str, required=True,
                       help='Path to test.pt file')
    parser.add_argument('--output', type=str, default='debug_outputs/evaluation',
                       help='Output directory')
    parser.add_argument('--input_dim', type=int, default=37,
                       help='Input dimension (default: 37 = 3 pos + 3 vel + 3 acc + 18 AU + 10 geo)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (will be inferred if not provided)')
    
    args = parser.parse_args()
    
    # Load label map from dataset
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
    
    evaluate_model(args.checkpoint, config, args.test_pt, args.output, label_map)


if __name__ == '__main__':
    main()

