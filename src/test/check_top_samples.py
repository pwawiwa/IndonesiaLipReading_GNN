#!/usr/bin/env python3
"""
Simple script to check top X samples in train.pt
Usage: python check_top_samples.py --top 10
"""

import torch
import argparse
import numpy as np

def check_top_samples(pt_file, top_n=10):
    """
    Check top N samples from processed PT file
    
    Args:
        pt_file: Path to PT file (e.g., 'data/landmarks/processed/train.pt')
        top_n: Number of top samples to check
    """
    print(f"Loading {pt_file}...")
    data, slices = torch.load(pt_file, weights_only=False)
    
    total_samples = len(slices['x']) - 1
    print(f"Total samples: {total_samples}")
    print(f"Checking top {top_n} samples:")
    print("-" * 80)
    
    for i in range(min(top_n, total_samples)):
        # Get sample data
        sample_x = data.x[slices['x'][i]:slices['x'][i+1]]
        sample_y = data.y[i].item()
        
        # Reshape to frames
        frames = sample_x.view(30, 468, 3)
        
        # Count real frames (non-zero)
        real_frames = 0
        for frame_idx in range(30):
            frame = frames[frame_idx]
            if not (frame == 0).all(dim=1).all():
                real_frames += 1
        
        # Get class name if available
        class_name = f"Class_{sample_y}"
        
        print(f"Sample {i+1:3d}:")
        print(f"  Class: {class_name} (ID: {sample_y})")
        print(f"  Shape: {sample_x.shape}")
        print(f"  Zero-padded frames: {30-real_frames}")
        print(f"  First landmark: [{sample_x[0,0]:.4f}, {sample_x[0,1]:.4f}, {sample_x[0,2]:.4f}]")
        print(f"  Last real landmark: [{sample_x[(real_frames*468)-1,0]:.4f}, {sample_x[(real_frames*468)-1,1]:.4f}, {sample_x[(real_frames*468)-1,2]:.4f}]")
        print()

def check_class_distribution(pt_file, top_n=20):
    """
    Check class distribution in the dataset
    
    Args:
        pt_file: Path to PT file
        top_n: Number of top classes to show
    """
    print(f"Loading {pt_file}...")
    data, slices = torch.load(pt_file, weights_only=False)
    
    # Count classes
    class_counts = {}
    for i in range(len(slices['x']) - 1):
        class_id = data.y[i].item()
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Class distribution (top {top_n}):")
    print("-" * 50)
    for i, (class_id, count) in enumerate(sorted_classes[:top_n]):
        print(f"Class {class_id:3d}: {count:5d} samples")
    
    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Total samples: {sum(class_counts.values())}")

def main():
    parser = argparse.ArgumentParser(description='Check top samples in PT file')
    parser.add_argument('--file', default='data/landmarks/processed/train.pt',
                       help='Path to PT file')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top samples to check')
    parser.add_argument('--classes', action='store_true',
                       help='Show class distribution instead')
    parser.add_argument('--top-classes', type=int, default=20,
                       help='Number of top classes to show')
    
    args = parser.parse_args()
    
    if args.classes:
        check_class_distribution(args.file, args.top_classes)
    else:
        check_top_samples(args.file, args.top)

if __name__ == "__main__":
    main()
