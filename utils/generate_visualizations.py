#!/usr/bin/env python3
"""
Generate loss and accuracy visualizations from training history.
"""
import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Generate training visualizations')
    parser.add_argument('--result-dir', type=str, required=True,
                       help='Result directory with history.pt')
    
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    
    # Load history
    history_file = result_dir / 'history.pt'
    if not history_file.exists():
        print(f"History file not found: {history_file}")
        return
    
    history = torch.load(history_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history and 'val_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = result_dir / 'loss_history.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_file}")


if __name__ == '__main__':
    main()

