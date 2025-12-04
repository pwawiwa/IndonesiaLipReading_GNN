#!/usr/bin/env python3
"""
Explanation of B2 features: What they compute and how they're used in models.
"""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.mediapipe_nodes import get_partition_nodes


def explain_b2():
    """Explain what B2 computes and how it's used."""
    print("=" * 80)
    print("B2 FEATURE ENGINEERING EXPLANATION")
    print("=" * 80)
    
    partition = 'mouth'
    engineer = FeatureEngineer(feature_level='B2')
    nodes = get_partition_nodes(partition)
    n_nodes = len(nodes)
    
    print(f"\nPartition: {partition.upper()} ({n_nodes} nodes)")
    print("\n" + "-" * 80)
    print("WHAT B2 COMPUTES (2 features per node):")
    print("-" * 80)
    
    # Create example landmarks
    frames = 5
    landmarks = torch.randn(frames, n_nodes, 2)
    meta = {'width': 640, 'height': 480}
    
    # Compute B2
    b2_features = engineer.compute_B2(landmarks, meta, partition)
    
    print(f"\n1. DISTANCE FEATURE (Feature 0):")
    print(f"   - Computes: Euclidean distance from anchor node to each node")
    print(f"   - Anchor node: Node 0 (MediaPipe landmark 0)")
    print(f"   - Shape: (frames, n_nodes, 1)")
    print(f"   - Range: [0, +∞) - always non-negative")
    print(f"   - Meaning: How far each landmark is from the anchor point")
    print(f"   - Example values: min={b2_features[:,:,0].min():.4f}, max={b2_features[:,:,0].max():.4f}")
    
    print(f"\n2. ANGLE FEATURE (Feature 1):")
    print(f"   - Computes: Angle between consecutive nodes in the graph")
    print(f"   - Method: Angle between vectors (prev→current) and (current→next)")
    print(f"   - Shape: (frames, n_nodes, 1)")
    print(f"   - Range: [0, π] radians (0° to 180°)")
    print(f"   - Meaning: Local curvature/bending at each node")
    print(f"   - Example values: min={b2_features[:,:,1].min():.4f}, max={b2_features[:,:,1].max():.4f}")
    
    print("\n" + "-" * 80)
    print("HOW B2 IS USED IN MODELS:")
    print("-" * 80)
    
    print("\nFeature Concatenation (when training with B2):")
    print("  B0 (2 features): [x, y] - normalized coordinates")
    print("  B1 (3 features): [vx, vy, speed] - velocity + speed")
    print("  B2 (2 features): [distance, angle] - geometric features")
    print("  ────────────────────────────────────────────────")
    print("  TOTAL (7 features per node): [x, y, vx, vy, speed, distance, angle]")
    
    print("\nIncremental Loading:")
    print("  - If you train with B0: loads only B0 (2 features)")
    print("  - If you train with B1: loads B0 + B1 (5 features)")
    print("  - If you train with B2: loads B0 + B1 + B2 (7 features)")
    print("  - If you train with B3: loads B0 + B1 + B2 + B3 (11 features)")
    
    print("\n" + "-" * 80)
    print("WHAT INFORMATION B2 PROVIDES TO THE MODEL:")
    print("-" * 80)
    
    print("\n1. SPATIAL RELATIONSHIPS:")
    print("   - Distance feature: Captures how landmarks are positioned relative")
    print("     to a stable anchor point (Node 0). This helps the model understand")
    print("     the overall mouth shape and size.")
    print("   - Example: When mouth opens wide, distances from anchor increase")
    
    print("\n2. LOCAL CURVATURE:")
    print("   - Angle feature: Captures local bending/curvature at each landmark")
    print("   - Helps distinguish between different lip shapes:")
    print("     • Small angles (near 0°): Straight segments")
    print("     • Large angles (near 180°): Sharp turns/bends")
    print("   - Example: Rounded lips vs. stretched lips have different angle patterns")
    
    print("\n3. COMPLEMENTARY TO B0 AND B1:")
    print("   - B0 provides: Absolute positions (x, y)")
    print("   - B1 provides: Motion information (velocity, speed)")
    print("   - B2 provides: Geometric relationships (distance, angles)")
    print("   - Together: Model gets spatial + temporal + geometric information")
    
    print("\n" + "-" * 80)
    print("EXAMPLE: How B2 helps distinguish visemes:")
    print("-" * 80)
    
    print("\nScenario: Distinguishing 'O' vs 'U' sounds")
    print("  - Both might have similar B0 (position) and B1 (motion)")
    print("  - But B2 provides:")
    print("    • Different distance patterns (mouth opening size)")
    print("    • Different angle patterns (lip curvature)")
    print("  - Model can use B2 to better separate these similar visemes")
    
    print("\n" + "-" * 80)
    print("TECHNICAL DETAILS:")
    print("-" * 80)
    
    print(f"\nAnchor Selection:")
    print(f"  - For 'mouth' partition: Uses Node 0 (MediaPipe landmark 0)")
    print(f"  - For 'full' partition: Uses Nose Tip (MediaPipe landmark 4)")
    print(f"  - Reason: Stable reference point for distance computation")
    
    print(f"\nAngle Computation:")
    print(f"  - For each node i:")
    print(f"    vec1 = coords[i] - coords[i-1]  (vector from previous node)")
    print(f"    vec2 = coords[i+1] - coords[i]  (vector to next node)")
    print(f"    angle = arccos(dot(vec1, vec2) / (||vec1|| * ||vec2||))")
    print(f"  - Edge cases: First/last nodes use boundary conditions")
    
    print(f"\nMemory Optimization:")
    print(f"  - Reduced from 5 anchors to 1 anchor (80% reduction)")
    print(f"  - Ratio features removed (not meaningful for full partition)")
    print(f"  - Result: 2 features instead of original 6+ features")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("B2 adds 2 geometric features per node:")
    print("  1. Distance from anchor - captures spatial relationships")
    print("  2. Angle between consecutive nodes - captures local curvature")
    print("\nWhen concatenated with B0+B1, gives model:")
    print("  - Position (B0) + Motion (B1) + Geometry (B2) = Rich feature set")
    print("\nThis helps the model better distinguish between similar visemes")
    print("by providing complementary geometric information.")
    print("=" * 80)


if __name__ == '__main__':
    explain_b2()


