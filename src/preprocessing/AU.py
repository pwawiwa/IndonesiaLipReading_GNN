import numpy as np

def compute_au_features(landmarks: np.ndarray, au_landmarks: dict) -> dict:
    """
    Compute AU-related distances per frame
    landmarks: [N,3] array of normalized ROI landmarks
    au_landmarks: dict mapping AU names -> landmark indices
    returns: dict AU_name -> scalar feature
    """
    features = {}
    for au_name, idxs in au_landmarks.items():
        if len(idxs) == 2:
            # Use distance between two points
            p1, p2 = landmarks[idxs[0]], landmarks[idxs[1]]
            features[au_name] = np.linalg.norm(p2 - p1)
        elif len(idxs) > 2:
            # Use spread of points
            points = landmarks[idxs]
            center = points.mean(axis=0)
            features[au_name] = np.mean(np.linalg.norm(points - center, axis=1))
        elif len(idxs) == 1:
            # Single point: height/z-value as proxy
            features[au_name] = landmarks[idxs[0], 2]  # z-axis
        else:
            features[au_name] = 0.0
    return features
