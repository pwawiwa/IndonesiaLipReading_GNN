import numpy as np

def get_facemesh_topology():
    """
    Returns adjacency list of MediaPipe FaceMesh (468 landmarks).
    Each edge connects two landmark indices.
    """
    # Reference: MediaPipe FaceMesh connections
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
        # Add more as needed (eye, eyebrows, etc.)
    ]

    num_nodes = 468
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in FACEMESH_CONNECTIONS:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # undirected

    return adj_matrix, FACEMESH_CONNECTIONS


if __name__ == "__main__":
    adj_matrix, edges = get_facemesh_topology()
    print("Adjacency matrix shape:", adj_matrix.shape)
    print("Example edges:", edges[:10])
