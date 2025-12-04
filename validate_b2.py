#!/usr/bin/env python3
"""
Validate B2 feature computation.
Check that B2 produces exactly 2 features per node (1 distance + 1 angle).
"""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.mediapipe_nodes import get_partition_nodes


def test_b2_shapes(partition='mouth'):
    """Test that B2 produces correct shapes."""
    print("=" * 60)
    print("VALIDATING B2 FEATURE ENGINEERING")
    print("=" * 60)
    
    engineer = FeatureEngineer(feature_level='B2')
    
    # Test single partition
    if partition:
        print(f"\nTesting {partition.upper()} partition...")
        
        # Get node count
        nodes = get_partition_nodes(partition)
        n_nodes = len(nodes)
        print(f"  Nodes: {n_nodes}")
        
        # Create dummy landmarks: (frames=10, n_nodes, 2)
        frames = 10
        landmarks = torch.randn(frames, n_nodes, 2)
        meta = {'width': 640, 'height': 480}
        
        # Compute B2 features
        b2_features = engineer.compute_B2(landmarks, meta, partition)
        
        # Validate shape
        expected_shape = (frames, n_nodes, 2)  # 1 distance + 1 angle
        actual_shape = b2_features.shape
        
        print(f"  Expected shape: {expected_shape}")
        print(f"  Actual shape:   {actual_shape}")
        
        if actual_shape == expected_shape:
            print(f"  ✓ Shape validation PASSED")
        else:
            print(f"  ✗ Shape validation FAILED")
            return False
        
        # Validate feature values
        # Distance should be >= 0
        distances = b2_features[:, :, 0]
        if (distances < 0).any():
            print(f"  ✗ Distance features contain negative values")
            return False
        print(f"  ✓ Distance features are non-negative (min={distances.min():.4f}, max={distances.max():.4f})")
        
        # Angles should be in [0, π] (from acos)
        angles = b2_features[:, :, 1]
        if (angles < 0).any() or (angles > 3.1416).any():
            print(f"  ✗ Angle features out of range [0, π]")
            print(f"     Min: {angles.min():.4f}, Max: {angles.max():.4f}")
            return False
        print(f"  ✓ Angle features in valid range [0, π] (min={angles.min():.4f}, max={angles.max():.4f})")
        
        # Check for NaN or Inf
        if torch.isnan(b2_features).any() or torch.isinf(b2_features).any():
            print(f"  ✗ Features contain NaN or Inf values")
            return False
        print(f"  ✓ No NaN or Inf values")
        
        # Check anchor selection
        nose_tip_mp = 4
        
        if partition == 'full' and nose_tip_mp in nodes:
            anchor_remapped_idx = nodes.index(nose_tip_mp)
            print(f"  ✓ Anchor: Nose tip (MP 4, remapped index {anchor_remapped_idx})")
        else:
            anchor_remapped_idx = 0
            print(f"  ✓ Anchor: Node 0 (MP {nodes[0]}, remapped index {anchor_remapped_idx})")
    
    print("\n" + "=" * 60)
    print("B2 VALIDATION COMPLETE - ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate B2 features')
    parser.add_argument('--partition', type=str, default='mouth', choices=['lips', 'mouth', 'full'],
                       help='Partition to test (default: mouth)')
    args = parser.parse_args()
    
    success = test_b2_shapes(partition=args.partition)
    sys.exit(0 if success else 1)

