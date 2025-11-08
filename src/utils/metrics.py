from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import Dict

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> Dict:
    """
    Computes accuracy, and macro/weighted F1, Precision, Recall.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        labels: List of unique class labels (indices 0 to N-1)
        
    Returns:
        Dictionary of computed metrics.
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F-score, Support for all classes
    # zero_division='warn' is fine, or set to 0.0
    prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=labels, zero_division=0
    )
    prec_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=labels, zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': prec_macro,
        'recall_macro': recall_macro,
        'f1_weighted': f1_weighted,
        'precision_weighted': prec_weighted,
        'recall_weighted': recall_weighted
    }
    
    return metrics