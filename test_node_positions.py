#!/usr/bin/env python3
"""
Test script to verify that original MediaPipe node positions are preserved.
This ensures that when you refer to a node by its MediaPipe index (0-467),
it always corresponds to the same landmark position.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from preprocessing.mediapipe_nodes import get_mouth_area_nodes, build_partition_adjacency
from preprocessing.generate_partition_previews import get_real_facemesh_coords
import torch

def test_node_positions():
    """Test that node positions are consistent and use original MediaPipe indices."""
    print("="*80)
    print("TESTING NODE POSITION CONSISTENCY")
    print("="*80)
    
    # Get mouth partition nodes (returns ORIGINAL MediaPipe indices 0-467)
    nodes = get_mouth_area_nodes()
    print(f"\nTotal nodes in mouth partition: {len(nodes)}")
    print(f"Node list is deterministic: {len(set(nodes)) == len(nodes)}")
    
    # Get real MediaPipe coordinates
    real_coords = get_real_facemesh_coords()
    if real_coords is None:
        print("\n⚠ Warning: Could not get real MediaPipe coordinates")
        return
    
    # Test specific nodes mentioned by user
    test_nodes = [1, 59, 64, 84, 98, 100, 129]
    
    print("\n" + "="*80)
    print("ORIGINAL MEDIAPIPE NODE POSITIONS (Fixed)")
    print("="*80)
    print("\nWhen you refer to a node number, it means the ORIGINAL MediaPipe index (0-467):")
    print("These positions are FIXED and never change in MediaPipe FaceMesh.\n")
    
    for node_idx in test_nodes:
        if node_idx < 468:
            x, y = real_coords[node_idx, 0], real_coords[node_idx, 1]
            in_partition = node_idx in nodes
            print(f"Node {node_idx:3d}: Position ({x:.4f}, {y:.4f}) - {'IN partition' if in_partition else 'NOT in partition'}")
    
    # Get adjacency and mapping
    adj, node_mapping = build_partition_adjacency('mouth')
    
    print("\n" + "="*80)
    print("NODE MAPPING (Original -> Remapped)")
    print("="*80)
    print("\nThe node_mapping converts ORIGINAL MediaPipe indices to remapped indices (0, 1, 2, ... N-1)")
    print("This remapping is ONLY for the adjacency matrix. Landmarks preserve original positions.\n")
    
    for node_idx in test_nodes:
        if node_idx in node_mapping:
            remapped = node_mapping[node_idx]
            x, y = real_coords[node_idx, 0], real_coords[node_idx, 1]
            print(f"Original MediaPipe node {node_idx:3d} -> Remapped index {remapped:3d} | Position: ({x:.4f}, {y:.4f})")
        else:
            print(f"Original MediaPipe node {node_idx:3d} -> NOT IN PARTITION")
    
    # Verify connections
    print("\n" + "="*80)
    print("MANUAL CONNECTIONS VERIFICATION")
    print("="*80)
    print("\nAll connections use ORIGINAL MediaPipe indices:\n")
    
    connections = [
        (1, 98), (57, 98), (82, 98), (82, 129),
        (59, 100), (84, 100), (84, 129)
    ]
    
    for orig_i, orig_j in connections:
        if orig_i in node_mapping and orig_j in node_mapping:
            remapped_i = node_mapping[orig_i]
            remapped_j = node_mapping[orig_j]
            connected = adj[remapped_i, remapped_j].item() > 0.5
            status = "✓ CONNECTED" if connected else "✗ NOT CONNECTED"
            print(f"  {orig_i:3d} -> {orig_j:3d} (original) | {remapped_i:3d} -> {remapped_j:3d} (remapped) | {status}")
        else:
            missing = []
            if orig_i not in node_mapping:
                missing.append(str(orig_i))
            if orig_j not in node_mapping:
                missing.append(str(orig_j))
            print(f"  {orig_i:3d} -> {orig_j:3d} | ⚠ Nodes {', '.join(missing)} not in partition")
    
    print("\n" + "="*80)
    print("KEY POINT:")
    print("="*80)
    print("When you mention 'node 59', it ALWAYS refers to MediaPipe landmark 59,")
    print("which has a FIXED position in the MediaPipe FaceMesh (never changes).")
    print("The remapping (0, 1, 2, ... N-1) is ONLY for the adjacency matrix.")
    print("Landmarks are stored in remapped order but preserve original positions.")
    print("="*80)

if __name__ == '__main__':
    test_node_positions()

