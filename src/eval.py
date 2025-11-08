"""
src/eval.py
Evaluation module with comprehensive metrics
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation class
    """
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        criterion: nn.Module = None
    ) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader
            device: Device
            criterion: Loss function (optional)
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = batch.to(device)
                
                # Forward pass
                logits = model(batch)
                preds = torch.argmax(logits, dim=1)
                
                # Store predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                
                # Compute loss if criterion provided
                if criterion is not None:
                    loss = criterion(logits, batch.y.squeeze(-1))
                    total_loss += loss.item() * batch.num_graphs
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels, all_logits)
        
        # Add loss if computed
        if criterion is not None:
            metrics['loss'] = total_loss / len(dataloader.dataset)
        
        return metrics
    
    def _calculate_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        logits: np.ndarray
    ) -> Dict:
        """
        Calculate all metrics
        
        Args:
            preds: Predictions [N]
            labels: Ground truth labels [N]
            logits: Model logits [N, num_classes]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(labels, preds)
        
        # Precision, Recall, F1 - Macro
        metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        
        # Precision, Recall, F1 - Weighted
        metrics['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['per_class'] = {}
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i]
            }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # Top-k accuracy
        metrics['top3_accuracy'] = self._topk_accuracy(logits, labels, k=3)
        metrics['top5_accuracy'] = self._topk_accuracy(logits, labels, k=5)
        
        # Classification report (string)
        metrics['classification_report'] = classification_report(
            labels,
            preds,
            target_names=self.class_names,
            zero_division=0
        )
        
        return metrics
    
    def _topk_accuracy(self, logits: np.ndarray, labels: np.ndarray, k: int) -> float:
        """
        Calculate top-k accuracy
        
        Args:
            logits: Model logits [N, num_classes]
            labels: Ground truth labels [N]
            k: Top k
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = np.any(top_k_preds == labels[:, None], axis=1)
        return np.mean(correct)
    
    def print_metrics(self, metrics: Dict, title: str = "Evaluation Results"):
        """
        Pretty print metrics
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the report
        """
        print("\n" + "="*80)
        print(f"{title:^80}")
        print("="*80)
        
        # Overall metrics
        print("\n[Overall Metrics]")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        
        print("\n[Macro Average]")
        print(f"Precision:         {metrics['precision_macro']:.4f}")
        print(f"Recall:            {metrics['recall_macro']:.4f}")
        print(f"F1-Score:          {metrics['f1_macro']:.4f}")
        
        print("\n[Weighted Average]")
        print(f"Precision:         {metrics['precision_weighted']:.4f}")
        print(f"Recall:            {metrics['recall_weighted']:.4f}")
        print(f"F1-Score:          {metrics['f1_weighted']:.4f}")
        
        # Top-k accuracy
        print("\n[Top-k Accuracy]")
        print(f"Top-3:             {metrics['top3_accuracy']:.4f}")
        print(f"Top-5:             {metrics['top5_accuracy']:.4f}")
        
        # Loss if available
        if 'loss' in metrics:
            print(f"\nLoss:              {metrics['loss']:.4f}")
        
        # Classification report
        print("\n[Per-Class Metrics]")
        print(metrics['classification_report'])
        
        print("="*80 + "\n")
    
    def find_worst_and_best_cases(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        top_k: int = 5
    ) -> Tuple[List, List]:
        """
        Find worst and best prediction cases
        
        Args:
            model: Model
            dataloader: DataLoader
            device: Device
            top_k: Number of cases to return
            
        Returns:
            worst_cases: List of (video_id, true_label, pred_label, confidence)
            best_cases: List of (video_id, true_label, pred_label, confidence)
        """
        model.eval()
        
        cases = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Finding cases", leave=False):
                batch = batch.to(device)
                
                # Forward pass
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Get confidence
                confidences = probs[torch.arange(len(preds)), preds]
                
                # Store cases
                for i in range(len(preds)):
                    video_id = batch.video_id[i] if hasattr(batch, 'video_id') else f"sample_{i}"
                    true_label = batch.y[i].item()
                    pred_label = preds[i].item()
                    confidence = confidences[i].item()
                    is_correct = (true_label == pred_label)
                    
                    cases.append({
                        'video_id': video_id,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence,
                        'is_correct': is_correct
                    })
        
        # Sort by confidence
        cases.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get worst cases (incorrect with high confidence)
        worst_cases = [c for c in cases if not c['is_correct']][:top_k]
        
        # Get best cases (correct with high confidence)
        best_cases = [c for c in cases if c['is_correct']][:top_k]
        
        return worst_cases, best_cases