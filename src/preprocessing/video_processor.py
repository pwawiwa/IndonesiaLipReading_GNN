# Video processing utilities for lip reading
import cv2
import mediapipe as mp
import torch
from torch_geometric.data import Data

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Define lip indices for MediaPipe Face Mesh
        self.lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]
        
    def extract_landmarks(self, frame):
        """Extract facial landmarks from a single frame."""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        return landmarks
        
    def create_graph(self, landmarks):
        """Create a PyG graph from landmarks."""
        if landmarks is None:
            return None
            
        # Extract lip landmarks coordinates
        nodes = []
        for idx in self.lip_indices:
            point = landmarks.landmark[idx]
            nodes.append([point.x, point.y, point.z])
            
        # Create edges (connections between landmarks)
        edges = []
        for i in range(len(self.lip_indices)):
            for j in range(i + 1, len(self.lip_indices)):
                edges.append([i, j])
                edges.append([j, i])  # Make it bidirectional
                
        # Convert to PyG Data object
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
        
    def process_video(self, video_path):
        """Process entire video and return sequence of graphs."""
        cap = cv2.VideoCapture(video_path)
        graphs = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            landmarks = self.extract_landmarks(frame)
            graph = self.create_graph(landmarks)
            if graph is not None:
                graphs.append(graph)
                
        cap.release()
        return graphs