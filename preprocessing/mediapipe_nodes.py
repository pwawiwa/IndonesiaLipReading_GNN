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
    Get node indices for mouth_area partition (lips + cheeks + chin).
    
    This function ensures full connectivity by:
    1. Starting with core mouth region nodes (lips, cheeks, jaw)
    2. Including additional lower face nodes needed for connectivity
    3. Using BFS to find the connected component containing all core nodes
    
    Returns:
        List of node indices (fully connected, focused on mouth region)
    """
    # Start with core mouth region nodes
    # Lips (inner + outer) - full coverage
    lips = get_lips_nodes()
    
    # Cheek landmarks - full coverage
    cheeks = [
        # Left cheek
        50, 118, 119, 100, 101, 36, 203, 205, 206, 216,
        # Right cheek
        280, 347, 348, 330, 329, 266, 423, 425, 426, 436,
    ]
    
    # Chin and jaw - full coverage
    chin_jaw = [
        # Chin center and surrounding
        152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
        # Jaw line (full)
        58, 172, 136, 150, 149, 176, 148, 152, 454,
        # Additional lower jaw nodes for full coverage
        18, 200, 199, 175, 169, 170, 140, 135, 138, 171,
        204, 208, 364, 367, 369, 394, 395, 396, 430,
    ]
    
    # Nose bottom (for mouth context)
    nose_bottom = [2, 98, 327]
    
    # Additional lower face mesh nodes for connectivity
    # These are intermediate nodes in the tesselation that connect mouth components
    additional_connecting = [
        # Lower face mesh nodes that connect lips to cheeks/jaw
        93, 123, 137, 147, 177, 213, 215, 352, 356, 360,
        366, 376, 401, 411, 427, 485, 487,
        # Additional nodes in lower face region
        132, 133, 134, 136, 142, 143, 144, 145, 153, 154,
        162, 163, 164, 165, 166, 167, 173, 174,
    ]
    
    # Combine all candidate nodes
    all_candidates = list(set(
        lips + cheeks + chin_jaw + nose_bottom + additional_connecting
    ))
    
    # Build full adjacency to verify connectivity
    full_adj = build_full_adjacency()
    
    # Use BFS to find the connected component containing core mouth nodes
    # Start BFS from lips nodes (most important for mouth partition)
    core_start_nodes = lips[:5]  # Start from a few lip nodes
    node_set = set(all_candidates)
    visited = set()
    connected_component = set()
    
    def bfs_from_start(start_node):
        """BFS to find connected component from start node."""
        component = set()
        queue = [start_node]
        visited_local = {start_node}
        
        while queue:
            current = queue.pop(0)
            if current in node_set:  # Only include if in our candidate set
                component.add(current)
            for neighbor in range(468):
                if (full_adj[current, neighbor] > 0.5 and 
                    neighbor not in visited_local):
                    visited_local.add(neighbor)
                    if neighbor in node_set:  # Only explore if in candidate set
                        queue.append(neighbor)
        return component
    
    # Find connected component starting from core nodes
    for start_node in core_start_nodes:
        if start_node not in visited:
            component = bfs_from_start(start_node)
            connected_component.update(component)
            visited.update(component)
    
    # Ensure all core nodes are included (add them if not in component)
    # CRITICAL: All lip nodes must be included (including inner corners 78, 308)
    core_nodes = set(lips + cheeks + chin_jaw + nose_bottom)
    connected_component.update(core_nodes)
    
    # Force all lip nodes to be in the final component (they're core)
    # This ensures inner corners 78, 308 are included even if BFS missed them
    for lip_node in lips:
        connected_component.add(lip_node)
    
    # Final BFS to ensure single connected component
    # Re-run BFS from all nodes in component to get full connectivity
    final_component = set()
    visited_final = set()
    
    def bfs_final(start_node):
        """Final BFS to get complete connected component."""
        component = set()
        queue = [start_node]
        visited_bfs = {start_node}
        
        while queue:
            current = queue.pop(0)
            component.add(current)
            for neighbor in range(468):
                if (full_adj[current, neighbor] > 0.5 and 
                    neighbor in connected_component and
                    neighbor not in visited_bfs):
                    visited_bfs.add(neighbor)
                    queue.append(neighbor)
        return component
    
    # Find the largest connected component
    for node in connected_component:
        if node not in visited_final:
            component = bfs_final(node)
            if len(component) > len(final_component):
                final_component = component
            visited_final.update(component)
    
    # CRITICAL: Force all lip nodes to be included (they're core nodes)
    # This ensures inner corners 78, 308 and all inner lip nodes are included
    for lip_node in lips:
        final_component.add(lip_node)
        # Also add any nodes directly connected to lip nodes in the full mesh
        for neighbor in range(468):
            if full_adj[lip_node, neighbor] > 0.5:
                if neighbor in all_candidates:  # Only if in our candidate set
                    final_component.add(neighbor)
    
    # Re-run BFS to ensure single connected component after adding lip nodes
    # This connects any isolated lip nodes to the main component
    visited_reconnect = set()
    all_final_nodes = set(final_component)
    
    def bfs_reconnect(start_node):
        """BFS to reconnect isolated nodes."""
        component = set()
        queue = [start_node]
        visited_bfs = {start_node}
        
        while queue:
            current = queue.pop(0)
            component.add(current)
            for neighbor in range(468):
                if (full_adj[current, neighbor] > 0.5 and 
                    neighbor in all_final_nodes and
                    neighbor not in visited_bfs):
                    visited_bfs.add(neighbor)
                    queue.append(neighbor)
        return component
    
    # Find all connected components in final_component
    all_components = []
    for node in final_component:
        if node not in visited_reconnect:
            component = bfs_reconnect(node)
            all_components.append(component)
            visited_reconnect.update(component)
    
    # Use the largest component (should include all core nodes)
    if all_components:
        final_component = max(all_components, key=len)
        # Ensure all core nodes are in final component
        for core_node in core_nodes:
            final_component.add(core_node)
        # Reconnect one more time to include all core nodes
        final_component = bfs_reconnect(list(core_nodes)[0])
    
    # Return sorted list
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
    
    Args:
        partition: One of 'lips', 'mouth', 'full'
        
    Returns:
        List of node indices
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

