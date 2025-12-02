#!/usr/bin/env python3
"""
Generate GradCAM visualizations for node importance.
"""
import argparse
import torch
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from utils.model_loader import load_model_from_checkpoint
from training import get_dataloader
from utils.gradcam_gnn import GradCAMGNN, aggregate_cam_per_class, visualize_aggregated_cam, save_cam_summary


def main():
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations')
    parser.add_argument('--result-dir', type=str, required=True,
                       help='Result directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Config file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--partition', type=str, required=True,
                       help='Partition name')
    parser.add_argument('--feature-level', type=str, required=True,
                       help='Feature level')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples per class for GradCAM')
    
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    device = args.device
    
    # Create gradcam output directory
    gradcam_dir = result_dir / 'gradcam'
    gradcam_dir.mkdir(exist_ok=True)
    
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
    
    test_loader = get_dataloader(str(test_file), batch_size=1, shuffle=False)
    
    # Get class names
    dataset = test_loader.dataset
    word_to_label = dataset.word_to_label
    class_names = sorted(word_to_label.keys(), key=lambda x: word_to_label[x])
    label_to_word = {v: k for k, v in word_to_label.items()}
    
    # Load model from checkpoint (handles config filtering automatically)
    model = load_model_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_name=config['model']['name'],
        device=device
    )
    
    # Initialize GradCAM
    gradcam = GradCAMGNN(model)
    
    # Collect CAM scores per class
    cam_per_class = {class_name: [] for class_name in class_names}
    
    print("Generating GradCAM for samples...")
    sample_count = 0
    max_samples_per_class = args.num_samples
    
    for batch_idx, (features, speech_mask, adj, labels) in enumerate(test_loader):
        if sample_count >= len(class_names) * max_samples_per_class:
            break
        
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(features, adj, speech_mask)
            _, predicted = outputs.max(1)
            predicted_class = predicted.item()
            true_class = labels.item()
        
        class_name = label_to_word[true_class]
        
        # Only process if we haven't collected enough for this class
        if len(cam_per_class[class_name]) < max_samples_per_class:
            # Generate CAM for true class
            # Note: GradCAM doesn't use speech_mask (model forward accepts it as optional)
            try:
                cam = gradcam.generate_cam_for_sequence(
                    features,
                    adj,
                    target_class=true_class
                )
                cam_per_class[class_name].append(cam)
                sample_count += 1
            except Exception as e:
                print(f"Error generating CAM for sample {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate CAM per class
    print("Aggregating CAM scores per class...")
    aggregated_cam = {}
    for class_name, cam_list in cam_per_class.items():
        if len(cam_list) > 0:
            aggregated_cam[class_name] = aggregate_cam_per_class(cam_list, method='mean')
    
    # Visualize aggregated CAM for each class
    print("Generating visualizations...")
    for class_name, cam_scores in aggregated_cam.items():
        output_path = gradcam_dir / f"gradcam_{class_name.replace(' ', '_')}.png"
        visualize_aggregated_cam(
            cam_scores,
            class_name,
            partition,
            str(output_path),
            top_k=20
        )
    
    # Save summary
    save_cam_summary(aggregated_cam, str(gradcam_dir / 'gradcam_summary.pt'))
    
    print(f"GradCAM visualizations saved to: {gradcam_dir}")
    print(f"Generated CAM for {len(aggregated_cam)} classes")


if __name__ == '__main__':
    main()

