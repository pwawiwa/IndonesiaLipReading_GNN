import numpy as np

def get_facemesh_topology(mouth_only=True):
    """
    Returns adjacency list for MediaPipe FaceMesh landmarks.
    If mouth_only=True, returns connections for 40 mouth landmarks only.
    If mouth_only=False, returns connections for full 468 landmarks.
    """
    if mouth_only:
        # MediaPipe FaceMesh mouth landmark indices
        MOUTH_LANDMARKS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
        
        # Create mapping from original indices to new indices (0-39)
        landmark_to_idx = {landmark: idx for idx, landmark in enumerate(MOUTH_LANDMARKS)}
        
        # Get original MediaPipe lip connections
        import mediapipe as mp
        original_connections = list(mp.solutions.face_mesh.FACEMESH_LIPS)
        
        # Convert to new indices and filter valid connections
        FACEMESH_CONNECTIONS = []
        for i, j in original_connections:
            if i in landmark_to_idx and j in landmark_to_idx:
                new_i = landmark_to_idx[i]
                new_j = landmark_to_idx[j]
                FACEMESH_CONNECTIONS.append((new_i, new_j))
        
        num_nodes = 40
    else:
        # Original full face connections (subset)
        FACEMESH_CONNECTIONS = [
            # Contour
            (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
            (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
            # Lips outer
            (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
            (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
            (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
            (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
            # Lips inner
            (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
            (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
            (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
            (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
            # Jaw
            (127, 34), (34, 139), (139, 71), (71, 68), (68, 104),
            (104, 69), (69, 67), (67, 109), (109, 10),
        ]
        num_nodes = 468

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in FACEMESH_CONNECTIONS:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # undirected

    return adj_matrix, FACEMESH_CONNECTIONS


if __name__ == "__main__":
    print("=== Mouth Only Topology ===")
    adj_matrix, edges = get_facemesh_topology(mouth_only=True)
    print("Adjacency matrix shape:", adj_matrix.shape)
    print("Number of connections:", len(edges))
    print("Example edges:", edges[:10])
    
    print("\n=== Full Face Topology ===")
    adj_matrix_full, edges_full = get_facemesh_topology(mouth_only=False)
    print("Adjacency matrix shape:", adj_matrix_full.shape)
    print("Number of connections:", len(edges_full))
    print("Example edges:", edges_full[:10])
