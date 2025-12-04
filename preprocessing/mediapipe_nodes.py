"""
MediaPipe FaceMesh node definitions and adjacency matrices.

MediaPipe FaceMesh has 468 landmarks. We define three partitions:
- full: all 468 nodes with default MediaPipe FACEMESH_TESSELATION connections
- mouth_area: mouth + cheeks + chin region
- lips: inner + outer lips only
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
import warnings
import sys
import os
from contextlib import redirect_stderr
from io import StringIO

# Suppress protobuf compatibility warning from MediaPipe
# This is a harmless error: protobuf 6.x removed GetPrototype but MediaPipe still works
# The error appears on stderr but doesn't affect functionality
try:
    # Suppress stderr during MediaPipe import to hide AttributeError
    with redirect_stderr(StringIO()):
        import mediapipe as mp
except Exception:
    # If suppression fails, import normally (error will appear but won't break functionality)
    import mediapipe as mp

# Get MediaPipe's default connections
mp_face_mesh = mp.solutions.face_mesh

# Use MediaPipe's default FACEMESH_TESSELATION (complete mesh)
FACEMESH_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION

# Also keep other connection sets for partial partitions
FACEMESH_LIPS = mp_face_mesh.FACEMESH_LIPS
FACEMESH_LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
FACEMESH_RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
FACEMESH_FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL
FACEMESH_CONTOURS = mp_face_mesh.FACEMESH_CONTOURS


def get_lips_nodes() -> List[int]:
    """
    Get node indices for lips partition (inner + outer lips).
    
    Returns:
        List of node indices
    """
    # Inner lips
    inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                  324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    # Outer lips
    outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                  375, 321, 405, 314, 17, 84, 181, 91, 146]
    
    # Lip corners and additional points
    corners = [61, 291, 78, 308]
    
    # Combine and remove duplicates
    all_nodes = list(set(inner_lips + outer_lips + corners))
    all_nodes.sort()
    
    return all_nodes


def get_mouth_area_nodes() -> List[int]:
    """
    Get node indices for mouth_area partition (lips + jaw + cheeks + nose).
    
    IMPORTANT: Returns ORIGINAL MediaPipe node indices (0-467), NOT remapped indices.
    These indices correspond to fixed MediaPipe FaceMesh landmark positions.
    The remapping to 0, 1, 2, ... N-1 happens in build_partition_adjacency().
    
    Selection criteria:
    1. All lips nodes (inner + outer)
    2. Jaw nodes (symmetrical - both left and right)
    3. Cheek nodes (symmetrical - both left and right)
    4. Nose nodes (for anchor)
    5. All nodes connected to the above (via BFS)
    
    Returns:
        List of ORIGINAL MediaPipe node indices (0-467) in sorted order (deterministic)
    """
    # 1. All lips nodes (inner + outer)
    lips = get_lips_nodes()
    
    # 2. Jaw nodes (symmetrical - left and right)
    # Left jaw
    left_jaw = [
        58, 172, 136, 150, 149, 176, 148,  # Left jaw line
        18, 200, 199, 175, 169, 170, 140, 135, 138, 171,  # Left lower jaw
        204, 208,  # Left jaw continuation
    ]
    # Right jaw (symmetrical)
    right_jaw = [
        288, 361, 323, 454,  # Right jaw line
        364, 367, 369, 394, 395, 396, 430,  # Right lower jaw
        377, 400, 378, 379, 365, 397,  # Right jaw continuation
    ]
    # Chin center (shared)
    chin_center = [152]
    
    jaw = left_jaw + right_jaw + chin_center
    
    # 3. Cheek nodes (symmetrical - left and right)
    # Left cheek
    left_cheek = [
        50, 118, 119, 100, 101, 36, 203, 205, 206, 216,
    ]
    # Right cheek (symmetrical)
    right_cheek = [
        280, 347, 348, 330, 329, 266, 423, 425, 426, 436,
    ]
    
    cheeks = left_cheek + right_cheek
    
    # 4. Nose node (for anchor) - only 1 node
    nose = [4]  # Nose tip only (primary anchor)
    
    # Core nodes: lips + jaw + cheeks + nose
    core_nodes = set(lips + jaw + cheeks + nose)
    
    # Build full adjacency for BFS
    full_adj = build_full_adjacency()
    
    # 5. Use BFS to find all nodes connected to core nodes (limited to mouth region)
    # Limit BFS to only explore nodes within the lower face/mouth region
    # Define candidate region: lower face nodes that could connect mouth components
    # This includes nodes in the lower half of the face (roughly Y > 0.4 in normalized coordinates)
    # We'll use a hop-limited BFS: only explore up to 2-3 hops from core nodes
    
    # Lower face region candidates (nodes that could be in mouth region)
    # These are nodes that are geometrically in the lower face area
    lower_face_candidates = set()
    
    # Add all nodes that are directly connected to core nodes (1 hop)
    for core_node in core_nodes:
        for neighbor in range(468):
            if full_adj[core_node, neighbor] > 0.5:
                lower_face_candidates.add(neighbor)
    
    # Add nodes that are 2 hops away from core nodes (still in mouth region)
    second_hop = set()
    for candidate in lower_face_candidates:
        for neighbor in range(468):
            if full_adj[candidate, neighbor] > 0.5:
                second_hop.add(neighbor)
    
    # Combine: core nodes + 1 hop + 2 hops
    all_candidates = core_nodes | lower_face_candidates | second_hop
    
    # Use BFS to find connected component within candidates
    visited = set()
    connected_component = set()
    
    def bfs_from_core(start_nodes, candidate_set):
        """BFS to find connected component from core nodes, limited to candidate set."""
        component = set()
        queue = list(start_nodes)
        visited_bfs = set(start_nodes)
        
        while queue:
            current = queue.pop(0)
            if current in candidate_set:  # Only include if in candidate set
                component.add(current)
            # Sort neighbors for deterministic order
            neighbors = sorted([n for n in range(468) 
                              if full_adj[current, n] > 0.5 and n not in visited_bfs])
            for neighbor in neighbors:
                visited_bfs.add(neighbor)
                if neighbor in candidate_set:  # Only explore if in candidate set
                    queue.append(neighbor)
        return component
    
    # Start BFS from core nodes (deterministic: sorted)
    sorted_core = sorted(core_nodes)
    connected_component = bfs_from_core(sorted_core, all_candidates)
    
    # Ensure all core nodes are included (they must be in the partition)
    connected_component.update(core_nodes)
    
    # Final BFS to ensure single connected component within candidates
    visited_final = set()
    final_component = set()
    
    def bfs_final(start_node, candidate_set):
        """Final BFS to get complete connected component within candidates."""
        component = set()
        queue = [start_node]
        visited_bfs = {start_node}
        
        while queue:
            current = queue.pop(0)
            if current in candidate_set:
                component.add(current)
            # Sort neighbors for deterministic order
            neighbors = sorted([n for n in range(468)
                              if (full_adj[current, n] > 0.5 and 
                                  n in candidate_set and
                                  n not in visited_bfs)])
            for neighbor in neighbors:
                visited_bfs.add(neighbor)
                queue.append(neighbor)
        return component
    
    # Find the largest connected component (deterministic: iterate in sorted order)
    for node in sorted(connected_component):
        if node not in visited_final:
            component = bfs_final(node, all_candidates)
            if len(component) > len(final_component):
                final_component = component
            elif len(component) == len(final_component) and len(component) > 0:
                # If same size, choose deterministically by smallest first node
                if min(component) < min(final_component):
                    final_component = component
            visited_final.update(component)
    
    # Ensure all core nodes are in final component
    final_component.update(core_nodes)
    
    # Remove all nose nodes except nose tip (4)
    # Only keep node 4 for nose anchor
    nose_nodes_to_remove = [2, 6, 8, 9, 49, 98, 168, 220, 290, 305, 327]
    for nose_node in nose_nodes_to_remove:
        if nose_node in final_component:
            final_component.remove(nose_node)
    
    # Ensure nose tip (4) is included
    if 4 not in final_component:
        final_component.add(4)
    
    # Return sorted list (deterministic)
    all_nodes = sorted(list(final_component))
    
    return all_nodes


def get_full_nodes() -> List[int]:
    """
    Get all 468 MediaPipe FaceMesh node indices.
    
    Returns:
        List of all node indices (0-467)
    """
    return list(range(468))


def get_partition_nodes(partition: str) -> List[int]:
    """
    Get node indices for specified partition.
    
    IMPORTANT: Returns ORIGINAL MediaPipe node indices (0-467), NOT remapped indices.
    These indices correspond to fixed MediaPipe FaceMesh landmark positions.
    The remapping to 0, 1, 2, ... N-1 happens in build_partition_adjacency().
    
    Args:
        partition: One of 'lips', 'mouth', 'full'
        
    Returns:
        List of ORIGINAL MediaPipe node indices (0-467) in sorted order (deterministic)
    """
    if partition == 'lips':
        return get_lips_nodes()
    elif partition in ['mouth', 'mouth_area']:
        return get_mouth_area_nodes()
    elif partition == 'full':
        return get_full_nodes()
    else:
        raise ValueError(f"Unknown partition: {partition}. Use 'lips', 'mouth', or 'full'")


def build_full_adjacency() -> torch.Tensor:
    """
    Build full adjacency matrix using MediaPipe's default FACEMESH_TESSELATION.
    
    This uses the complete face mesh topology with all connections as defined
    by MediaPipe (not just contours, but the full triangulated mesh).
    
    Returns:
        Adjacency matrix of shape (468, 468)
    """
    # Use MediaPipe's default tesselation (complete mesh)
    all_edges = FACEMESH_TESSELATION
    
    # Create adjacency matrix
    adj = torch.zeros(468, 468, dtype=torch.float32)
    
    for i, j in all_edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # Undirected graph
    
    # Add manual connections if not in MediaPipe's tesselation
    # Connection: 396 -> 200 (jaw region)
    if adj[396, 200] == 0:
        adj[396, 200] = 1.0
        adj[200, 396] = 1.0
    
    # Add self-loops
    adj = adj + torch.eye(468, dtype=torch.float32)
    
    return adj


def build_partition_adjacency(partition: str) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Build adjacency matrix for specified partition.
    
    The adjacency preserves anatomical connections by:
    1. Getting the subset of nodes for the partition
    2. Keeping only edges where both endpoints exist in the subset
    3. Remapping node indices to 0, 1, 2, ... N-1
    
    For 'full' partition: Uses MediaPipe's default FACEMESH_TESSELATION (complete mesh).
    For partial partitions: Prunes the tesselation to keep only edges within the subset.
    
    Args:
        partition: One of 'lips', 'mouth', 'full'
        
    Returns:
        Tuple of (adjacency_matrix, node_mapping)
        - adjacency_matrix: shape (N, N) where N is number of nodes in partition
        - node_mapping: Dict mapping original MediaPipe index to new index
    """
    nodes = get_partition_nodes(partition)
    n_nodes = len(nodes)
    
    # Create node mapping: original_idx -> new_idx
    node_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(nodes)}
    
    if partition == 'full':
        # For full face, use MediaPipe's default FACEMESH_TESSELATION
        return build_full_adjacency(), node_mapping
    
    # For partial faces, prune adjacency from full tesselation
    full_adj = build_full_adjacency()
    node_set = set(nodes)
    
    # Create new adjacency matrix
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    
    # Copy edges where both endpoints exist in partition
    for orig_i in nodes:
        for orig_j in nodes:
            if full_adj[orig_i, orig_j] > 0:
                new_i = node_mapping[orig_i]
                new_j = node_mapping[orig_j]
                adj[new_i, new_j] = 1.0
    
    # No manual connections needed - using clean partition with only lips, jaw, cheeks, nose
    # All connections come from MediaPipe's default tesselation
    
    return adj, node_mapping


def get_au_node_groups(partition: str, node_mapping: Dict[int, int]) -> Dict[str, List[int]]:
    """
    Get Action Unit (AU) node groups based on anatomical regions.
    
    Maps AUs to actual anatomical node groups (not just index ranges).
    
    Action Unit Definitions (FACS - Facial Action Coding System):
    - AU25 (Lips Part): Measures vertical lip separation/opening
    - AU26 (Jaw Drop): Measures jaw opening (lower jaw movement)
    - AU12 (Lip Corner Pull): Measures lip corner movement (smile)
    - AU27 (Mouth Stretch): Measures horizontal mouth opening/stretch
    
    Args:
        partition: One of 'lips', 'mouth', 'full'
        node_mapping: Dict mapping original MediaPipe index -> remapped index
        
    Returns:
        Dictionary mapping AU name to list of remapped node indices
    """
    au_groups = {}
    
    # Create reverse mapping: remapped index -> original MediaPipe index
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    if partition == 'lips':
        # For lips partition, use all lip nodes for AU25
        lips_original = get_lips_nodes()
        au_groups['AU25_lips_part'] = [
            node_mapping[n] for n in lips_original if n in node_mapping
        ]
        # AU27: Mouth stretch (use remaining nodes if any)
        all_nodes_remapped = set(node_mapping.values())
        au25_nodes = set(au_groups['AU25_lips_part'])
        remaining = sorted(list(all_nodes_remapped - au25_nodes))
        if remaining:
            au_groups['AU27_mouth_stretch'] = remaining[:min(20, len(remaining))]
    
    elif partition in ['mouth', 'mouth_area']:
        # AU25: Lips Part - measures vertical lip separation
        lips_original = get_lips_nodes()
        au_groups['AU25_lips_part'] = [
            node_mapping[n] for n in lips_original if n in node_mapping
        ]
        
        # AU26: Jaw Drop - measures jaw opening (symmetric left and right jaw nodes)
        # Use symmetric jaw nodes for proper coverage
        # Right jaw continuation after 58: 215 -> 132 -> 177 -> 93 -> 234
        jaw_chin_original = [
            # Symmetric left jaw
            172, 136, 150, 149, 176, 148,
            # Symmetric right jaw
            397, 365, 379, 378, 400, 377,
            # Chin center and lower jaw
            152, 18, 58,
            # Right jaw continuation after 58
            215, 132, 177, 93, 234,
            # Additional lower jaw nodes
            454, 200, 199, 175, 169, 170, 140, 135, 138, 171,
            208, 364, 367, 369, 394, 395, 396,
            # Additional chin/jaw nodes
            288, 361, 323,
        ]
        au_groups['AU26_jaw_drop'] = [
            node_mapping[n] for n in jaw_chin_original if n in node_mapping
        ]
        
        # AU12: Lip Corner Pull - measures lip corner movement (smile)
        lip_corners_original = [
            # Actual lip corner nodes only
            61, 78,   # Left corner (outer and inner)
            291, 308, # Right corner (outer and inner)
        ]
        au_groups['AU12_lip_corner'] = [
            node_mapping[n] for n in lip_corners_original if n in node_mapping
        ]
        
        # AU27: Mouth Stretch - measures horizontal mouth opening
        # Use horizontal lip nodes (left/right lip edges) and corners
        mouth_stretch_original = [
            # Horizontal lip edges (left side)
            61, 84, 17, 314, 405, 320, 307, 375, 321,
            # Horizontal lip edges (right side)  
            291, 308, 324, 318, 402, 317, 14, 87, 178,
            # Additional horizontal mouth nodes
            0, 13, 14, 17, 37, 39, 40, 267, 269, 270,
        ]
        au_groups['AU27_mouth_stretch'] = [
            node_mapping[n] for n in mouth_stretch_original if n in node_mapping
        ]
        
        # If AU27 doesn't have enough nodes, add remaining mouth area nodes
        if len(au_groups['AU27_mouth_stretch']) < 10:
            all_nodes_remapped = set(node_mapping.values())
            assigned_nodes = set()
            for group_nodes in au_groups.values():
                assigned_nodes.update(group_nodes)
            remaining = sorted(list(all_nodes_remapped - assigned_nodes))
            if remaining:
                au_groups['AU27_mouth_stretch'].extend(remaining[:min(15, len(remaining))])
    
    else:  # full
        # AU25: Lips Part - measures vertical lip separation (upper and lower lip nodes)
        # Uses all lip nodes to capture vertical opening
        lips_original = get_lips_nodes()
        au_groups['AU25_lips_part'] = [
            node_mapping[n] for n in lips_original if n in node_mapping
        ]
        
        # AU26: Jaw Drop - measures jaw opening (symmetric left and right jaw nodes)
        # Use symmetric jaw nodes for proper coverage
        # Left jaw: 172, 136, 150, 149, 176, 148
        # Right jaw: 397, 365, 379, 378, 400, 377
        # Chin center: 152, 18
        # Right jaw continuation after 58: 215 -> 132 -> 177 -> 93 -> 234
        jaw_chin_original = [
            # Symmetric left jaw
            172, 136, 150, 149, 176, 148,
            # Symmetric right jaw
            397, 365, 379, 378, 400, 377,
            # Chin center and lower jaw
            152, 18, 58,
            # Right jaw continuation after 58
            215, 132, 177, 93, 234,
            # Additional lower jaw nodes
            454, 200, 199, 175, 169, 170, 140, 135, 138, 171,
            208, 364, 367, 369, 394, 395, 396,
            # Additional chin/jaw nodes for completeness
            288, 361, 323,
        ]
        au_groups['AU26_jaw_drop'] = [
            node_mapping[n] for n in jaw_chin_original if n in node_mapping
        ]
        
        # AU12: Lip Corner Pull - measures lip corner movement (smile)
        # ONLY the 4 actual corner nodes
        lip_corners_original = [
            # Actual lip corner nodes only
            61, 78,   # Left corner (outer and inner)
            291, 308, # Right corner (outer and inner)
        ]
        au_groups['AU12_lip_corner'] = [
            node_mapping[n] for n in lip_corners_original if n in node_mapping
        ]
        
        # AU27: Mouth Stretch - measures horizontal mouth opening
        # Should use horizontal lip nodes (left/right lip edges) and corners
        # NOT just "remaining nodes" - this was incorrect
        # Use nodes that measure horizontal mouth width
        mouth_stretch_original = [
            # Horizontal lip edges (left side)
            61, 84, 17, 314, 405, 320, 307, 375, 321,
            # Horizontal lip edges (right side)  
            291, 308, 324, 318, 402, 317, 14, 87, 178,
            # Additional horizontal mouth nodes
            0, 13, 14, 17, 37, 39, 40, 267, 269, 270,
        ]
        # Filter to only include nodes that are in the partition
        au_groups['AU27_mouth_stretch'] = [
            node_mapping[n] for n in mouth_stretch_original if n in node_mapping
        ]
        
        # If AU27 doesn't have enough nodes, add remaining mouth area nodes
        if len(au_groups['AU27_mouth_stretch']) < 10:
            all_nodes_remapped = set(node_mapping.values())
            assigned_nodes = set()
            for group_nodes in au_groups.values():
                assigned_nodes.update(group_nodes)
            remaining = sorted(list(all_nodes_remapped - assigned_nodes))
            # Add remaining mouth-related nodes to AU27
            if remaining:
                au_groups['AU27_mouth_stretch'].extend(remaining[:min(15, len(remaining))])
    
    # Sort node indices within each group
    for au_name in au_groups:
        au_groups[au_name] = sorted(au_groups[au_name])
    
    return au_groups


def get_partition_info(partition: str) -> Dict:
    """
    Get comprehensive information about a partition.
    
    Args:
        partition: One of 'lips', 'mouth', 'full'
        
    Returns:
        Dictionary with partition info
    """
    nodes = get_partition_nodes(partition)
    adj, node_mapping = build_partition_adjacency(partition)
    
    # Count edges (excluding self-loops)
    n_edges = int((adj.sum() - adj.trace()).item()) // 2
    
    return {
        'partition': partition,
        'n_nodes': len(nodes),
        'n_edges': n_edges,
        'nodes': nodes,
        'adjacency': adj,
        'node_mapping': node_mapping,
    }

