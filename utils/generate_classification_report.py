#!/usr/bin/env python3
"""
Generate classification report and confusion matrix.
"""
import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from utils.model_loader import load_model_from_checkpoint
from training import get_dataloader
from training.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Generate classification report')
    parser.add_argument('--result-dir', type=str, required=True,
                       help='Result directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Config file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    device = args.device
    
    # Load config
    config = load_config(args.config)
    
    # Load checkpoint
    checkpoint_path = result_dir / 'best.pth'
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load test data
    feature_level = config['data']['feature_level']
    partition = config['data']['partition']
    feature_dir = Path(config['data']['feature_dir']) / feature_level
    test_file = feature_dir / f"{partition}_test.pt"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    test_loader = get_dataloader(str(test_file), batch_size=32, shuffle=False)
    
    # Get class names
    dataset = test_loader.dataset
    word_to_label = dataset.word_to_label
    class_names = sorted(word_to_label.keys(), key=lambda x: word_to_label[x])
    
    # Load model from checkpoint (handles config filtering automatically)
    model = load_model_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_name=config['model']['name'],
        device=device
    )
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, device, class_names)
    results = evaluator.evaluate()
    
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Save classification report
    torch.save(results['classification_report'], result_dir / 'classification_report.pt')
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        result_dir / 'confusion_matrix.png'
    )
    
    # Plot classification report
    evaluator.plot_classification_report(
        results['classification_report'],
        result_dir / 'classification_report.png'
    )
    
    print(f"Saved classification report to: {result_dir}")


if __name__ == '__main__':
    main()

