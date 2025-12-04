"""
Utility functions for calculating class weights and analyzing class distribution.
"""
import torch
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from pathlib import Path


def calculate_class_weights(
    dataset,
    method: str = 'balanced',
    max_samples: int = None
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: Dataset instance with video_ids and word_to_label
        method: Weight calculation method
            - 'balanced': sklearn-style balanced weights (n_samples / (n_classes * count))
            - 'inverse': Inverse frequency (1 / count)
            - 'sqrt': Square root of inverse frequency (1 / sqrt(count))
            - 'log': Logarithmic inverse frequency (1 / log(1 + count))
        max_samples: Maximum number of samples to use for counting (for large datasets)
        
    Returns:
        Class weights tensor of shape (num_classes,)
    """
    # Count class frequencies
    class_counts = Counter()
    
    if max_samples is not None and len(dataset) > max_samples:
        # Sample subset for counting
        import random
        indices = random.sample(range(len(dataset)), max_samples)
        for idx in indices:
            video_id = dataset.video_ids[idx]
            word = dataset.video_to_word[video_id]
            label = dataset.word_to_label[word]
            class_counts[label] += 1
    else:
        # Count all samples
        for video_id in dataset.video_ids:
            word = dataset.video_to_word[video_id]
            label = dataset.word_to_label[word]
            class_counts[label] += 1
    
    num_classes = len(dataset.word_to_label)
    total_samples = sum(class_counts.values())
    
    # Create weight array
    weights = torch.ones(num_classes, dtype=torch.float32)
    
    for label, count in class_counts.items():
        if count == 0:
            continue
            
        if method == 'balanced':
            # sklearn-style: n_samples / (n_classes * count)
            weights[label] = total_samples / (num_classes * count)
        elif method == 'inverse':
            # Simple inverse frequency
            weights[label] = 1.0 / count
        elif method == 'sqrt':
            # Square root of inverse frequency
            weights[label] = 1.0 / np.sqrt(count)
        elif method == 'log':
            # Logarithmic inverse frequency
            weights[label] = 1.0 / np.log(1 + count)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to have mean=1.0 (optional, but helps with stability)
    weights = weights / weights.mean()
    
    return weights


def get_class_distribution(dataset, max_samples: int = None) -> Dict:
    """
    Get class distribution statistics.
    
    Args:
        dataset: Dataset instance
        max_samples: Maximum samples to analyze
        
    Returns:
        Dictionary with distribution stats
    """
    class_counts = Counter()
    
    if max_samples is not None and len(dataset) > max_samples:
        import random
        indices = random.sample(range(len(dataset)), max_samples)
        for idx in indices:
            video_id = dataset.video_ids[idx]
            word = dataset.video_to_word[video_id]
            label = dataset.word_to_label[word]
            class_counts[label] += 1
    else:
        for video_id in dataset.video_ids:
            word = dataset.video_to_word[video_id]
            label = dataset.word_to_label[word]
            class_counts[label] += 1
    
    counts = np.array([class_counts.get(i, 0) for i in range(len(dataset.word_to_label))])
    
    stats = {
        'class_counts': dict(class_counts),
        'total_samples': int(counts.sum()),
        'num_classes': len(dataset.word_to_label),
        'min_count': int(counts.min()),
        'max_count': int(counts.max()),
        'mean_count': float(counts.mean()),
        'std_count': float(counts.std()),
        'imbalance_ratio': float(counts.max() / counts.min()) if counts.min() > 0 else float('inf'),
        'zero_count_classes': int((counts == 0).sum())
    }
    
    return stats

