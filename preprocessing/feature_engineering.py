"""
Feature Engineering: B0 - B3 feature sets.

Each level stores CUMULATIVE features (includes all previous levels):
- B0: Raw normalized coordinates (X, Y per node) - 2 features
- B1: B0 + Velocity + speed + acceleration - 7 features (B0: 2 + B1: 5 = 7 total)
- B2: B0 + B1 + Global geometric features - 12 features (B0: 2 + B1: 5 + B2: 5 = 12 total)
  Note: B2 features are global per frame (broadcast to all nodes): MAR, lip width, lip height, cheek puff, chin height
- B3: B0 + B1 + B2 + AU features - 11 features (B0: 2 + B1: 3 + B2: 2 + B3: 4 = 11 total)
  Note: AU features: 4 AU groups (AU25, AU26, AU12, AU27). PCA and motion energy removed.

When loading for training, just load the target level file (no concatenation needed).
Each file is self-contained with all features up to that level.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
import math
from tqdm import tqdm


class FeatureEngineer:
    """Feature engineering for landmark sequences."""
    
    def __init__(self, feature_level: str = 'B0'):
        """
        Initialize feature engineer.
        
        Args:
            feature_level: One of 'B0', 'B1', 'B2', 'B3'
        """
        self.feature_level = feature_level
        self.pca_models = {}  # For B4
    
    def compute_B0(self, landmarks: torch.Tensor, meta: Dict) -> torch.Tensor:
        """
        B0: Raw normalized coordinates.
        
        Input landmarks are already normalized to [0, 1] by resolution during extraction.
        Here we additionally center and scale by face bounding box.
        
        Args:
            landmarks: Shape (frames, n_nodes, 2)
            meta: Video metadata with width, height
            
        Returns:
            Features of shape (frames, n_nodes, 2)
        """
        # Landmarks are already resolution-normalized (X/width, Y/height)
        # Additional normalization: center by face center, scale by face size
        # VECTORIZED for efficiency
        
        frames, n_nodes, _ = landmarks.shape
        features = landmarks.clone()
        
        # Vectorized per-frame normalization
        # Compute bounding boxes for all frames at once
        x_coords = features[:, :, 0]  # (frames, n_nodes)
        y_coords = features[:, :, 1]  # (frames, n_nodes)
        
        # Create valid mask (non-zero coordinates)
        valid_mask = (x_coords != 0) | (y_coords != 0)  # (frames, n_nodes)
        has_valid = valid_mask.any(dim=1)  # (frames,)
        
        # For frames with valid points, compute min/max using masked operations
        # Replace invalid values with inf/-inf for min/max computation
        x_coords_masked = x_coords.masked_fill(~valid_mask, float('inf'))
        x_min = x_coords_masked.min(dim=1, keepdim=True)[0]  # (frames, 1)
        x_min = torch.where(has_valid.unsqueeze(1), x_min, torch.zeros_like(x_min))
        
        x_coords_masked = x_coords.masked_fill(~valid_mask, float('-inf'))
        x_max = x_coords_masked.max(dim=1, keepdim=True)[0]  # (frames, 1)
        x_max = torch.where(has_valid.unsqueeze(1), x_max, torch.zeros_like(x_max))
        
        y_coords_masked = y_coords.masked_fill(~valid_mask, float('inf'))
        y_min = y_coords_masked.min(dim=1, keepdim=True)[0]  # (frames, 1)
        y_min = torch.where(has_valid.unsqueeze(1), y_min, torch.zeros_like(y_min))
        
        y_coords_masked = y_coords.masked_fill(~valid_mask, float('-inf'))
        y_max = y_coords_masked.max(dim=1, keepdim=True)[0]  # (frames, 1)
        y_max = torch.where(has_valid.unsqueeze(1), y_max, torch.zeros_like(y_max))
        
        # Face centers and sizes (vectorized)
        x_center = (x_min + x_max) / 2  # (frames, 1)
        y_center = (y_min + y_max) / 2  # (frames, 1)
        face_width = x_max - x_min  # (frames, 1)
        face_height = y_max - y_min  # (frames, 1)
        face_size = torch.max(face_width, face_height)  # (frames, 1)
        
        # Avoid division by zero
        face_size = torch.clamp(face_size, min=1e-6)
        
        # Center and scale (vectorized)
        features[:, :, 0] = (x_coords - x_center) / face_size
        features[:, :, 1] = (y_coords - y_center) / face_size
        
        # Zero out frames with no valid points
        features[~has_valid] = 0
        
        return features
    
    def compute_velocity(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity (first derivative).
        
        Args:
            coords: Shape (frames, n_nodes, 2)
            
        Returns:
            Velocity of shape (frames, n_nodes, 2)
        """
        frames = coords.shape[0]
        velocity = torch.zeros_like(coords)
        
        if frames > 1:
            # Forward difference for first frame
            velocity[0] = coords[1] - coords[0]
            
            # Central difference for middle frames
            if frames > 2:
                velocity[1:-1] = (coords[2:] - coords[:-2]) / 2.0
            
            # Backward difference for last frame
            velocity[-1] = coords[-1] - coords[-2]
        
        return velocity
    
    def compute_acceleration(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration (second derivative).
        
        Args:
            velocity: Shape (frames, n_nodes, 2)
            
        Returns:
            Acceleration of shape (frames, n_nodes, 2)
        """
        return self.compute_velocity(velocity)
    
    def compute_speed(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute speed magnitude from velocity.
        
        Args:
            velocity: Shape (frames, n_nodes, 2)
            
        Returns:
            Speed of shape (frames, n_nodes, 1)
        """
        # Speed = ||velocity||
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        return speed
    
    def compute_B1(self, landmarks: torch.Tensor, meta: Dict) -> torch.Tensor:
        """
        B1: Returns B0 + velocity + speed + acceleration (cumulative, includes B0).
        
        Args:
            landmarks: Shape (frames, n_nodes, 2)
            meta: Video metadata
            
        Returns:
            Features of shape (frames, n_nodes, 7)  [B0: x, y] + [B1: vx, vy, speed, ax, ay]
        """
        # B0 features (needed for velocity computation)
        b0 = self.compute_B0(landmarks, meta)
        
        # Velocity
        velocity = self.compute_velocity(b0)
        
        # Speed magnitude
        speed = self.compute_speed(velocity)
        
        # Acceleration
        acceleration = self.compute_acceleration(velocity)
        
        # Return B0 + velocity + speed + acceleration (cumulative: B0 + B1)
        b1_features = torch.cat([velocity, speed, acceleration], dim=-1)  # (frames, n_nodes, 5)
        features = torch.cat([b0, b1_features], dim=-1)  # (frames, n_nodes, 7)
        
        return features
    
    def compute_B2(self, landmarks: torch.Tensor, meta: Dict, partition: str, b0_coords: Optional[torch.Tensor] = None, b1_features: Optional[torch.Tensor] = None, node_indices: Optional[Dict[str, int]] = None) -> torch.Tensor:
        """
        B2: Returns B0 + B1 + global geometric features (cumulative, includes B0+B1).
        Global features are broadcast to all nodes per frame.
        Features: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening (8 features).
        OPTIMIZED: Accepts precomputed node indices to avoid repeated lookups.
        
        Args:
            landmarks: Shape (frames, n_nodes, 2) - raw landmarks (used if b0_coords not provided)
            meta: Video metadata
            partition: 'lips', 'mouth', or 'full'
            b0_coords: Optional pre-computed B0 coordinates (faster, avoids recomputation)
            b1_features: Optional pre-computed B1 features (B0+B1, faster, avoids recomputation)
            node_indices: Optional precomputed node indices dict (faster, avoids repeated lookups)
            
        Returns:
            Features of shape (frames, n_nodes, 15)  [B0: x, y] + [B1: vx, vy, speed, ax, ay] + [B2: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening]
        """
        # OPTIMIZATION: Use existing B0+B1 if provided (much faster)
        if b1_features is not None:
            b0_b1 = b1_features  # Already has B0+B1
            b0 = b1_features[:, :, :2]  # Extract B0 for geometric computation
        else:
            # Compute B0
            if b0_coords is not None:
                b0 = b0_coords
            else:
                b0 = self.compute_B0(landmarks, meta)
            
            # Compute B1 (B0 + velocity + speed + acceleration)
            b1 = self.compute_B1(landmarks, meta)  # Returns B0+B1
            b0_b1 = b1
        
        # Global geometric features (B2 incremental) - broadcast to all nodes
        # Features: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening (8 features)
        # OPTIMIZATION: Pass precomputed node_indices to avoid repeated lookups
        geom_global = self.compute_global_geometric_features(b0, partition, node_indices=node_indices)
        
        # Broadcast global features to all nodes: (frames, 8) -> (frames, n_nodes, 8)
        frames, n_nodes, _ = b0_b1.shape
        geom_broadcast = geom_global.unsqueeze(1).expand(frames, n_nodes, -1)  # (frames, n_nodes, 8)
        
        # Return B0 + B1 + B2 (cumulative)
        features = torch.cat([b0_b1, geom_broadcast], dim=-1)  # (frames, n_nodes, 15)
        
        return features
    
    def compute_global_geometric_features(
        self,
        coords: torch.Tensor,
        partition: str,
        node_indices: Optional[Dict[str, int]] = None
    ) -> torch.Tensor:
        """
        Compute global geometric features per frame (broadcast to all nodes).
        Features: MAR, lip width, lip height, jaw height, cheek puff, lip curvature, lip corner angle, jaw opening.
        OPTIMIZED: Accepts precomputed node indices to avoid repeated lookups.
        
        Args:
            coords: Shape (frames, n_nodes, 2) - normalized coordinates
            partition: 'lips', 'mouth', or 'full'
            node_indices: Optional precomputed dict with keys: 'left_corner', 'right_corner', 'chin_center', 'lips_nodes', 'left_cheek', 'right_cheek'
                         If None, will compute on-the-fly (slower)
            
        Returns:
            Global features of shape (frames, 8) - [MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening]
        """
        frames = coords.shape[0]
        device = coords.device
        dtype = coords.dtype
        
        # Initialize output tensor (8 features: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening)
        global_features = torch.zeros(frames, 8, device=device, dtype=dtype)
        
        # Use precomputed indices if available (much faster)
        if node_indices is not None:
            left_corner_remapped = node_indices.get('left_corner')
            right_corner_remapped = node_indices.get('right_corner')
            chin_center_remapped = node_indices.get('chin_center')
            lips_nodes_remapped = node_indices.get('lips_nodes', [])
            left_cheek_remapped = node_indices.get('left_cheek', [])
            right_cheek_remapped = node_indices.get('right_cheek', [])
        else:
            # Fallback: compute on-the-fly (slower, but backward compatible)
            from preprocessing.mediapipe_nodes import get_partition_nodes, get_lips_nodes
            
            # Get node indices
            nodes = get_partition_nodes(partition)
            node_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
            
            # Get specific landmark indices (original MediaPipe indices)
            lips_nodes_orig = get_lips_nodes()
            
            # Map to remapped indices
            lips_nodes_remapped = [node_mapping[n] for n in lips_nodes_orig if n in node_mapping]
            
            # Lip corner indices (original MediaPipe)
            left_corner_outer = 61  # Left lip corner (outer)
            right_corner_outer = 291  # Right lip corner (outer)
            left_corner_inner = 78  # Left lip corner (inner)
            right_corner_inner = 308  # Right lip corner (inner)
            
            # Chin center
            chin_center_orig = 152
            
            # Cheek nodes (original MediaPipe)
            left_cheek_orig = [50, 118, 119, 100, 101, 36, 203, 205, 206, 216]
            right_cheek_orig = [280, 347, 348, 330, 329, 266, 423, 425, 426, 436]
            
            # Map to remapped indices
            left_corner_remapped = node_mapping.get(left_corner_outer, node_mapping.get(left_corner_inner, None))
            right_corner_remapped = node_mapping.get(right_corner_outer, node_mapping.get(right_corner_inner, None))
            chin_center_remapped = node_mapping.get(chin_center_orig, None)
            left_cheek_remapped = [node_mapping[n] for n in left_cheek_orig if n in node_mapping]
            right_cheek_remapped = [node_mapping[n] for n in right_cheek_orig if n in node_mapping]
        
        # Extract lip coordinates
        if lips_nodes_remapped:
            lip_coords = coords[:, lips_nodes_remapped, :]  # (frames, n_lip_nodes, 2)
            
            # Compute lip Y bounds once (used for MAR, lip_height, and jaw_height)
            lip_y_min = lip_coords[:, :, 1].min(dim=1)[0]  # (frames,) - upper lip
            lip_y_max = lip_coords[:, :, 1].max(dim=1)[0]  # (frames,) - lower lip
            lip_height = lip_y_max - lip_y_min  # (frames,)
            
            # 1. MAR (Mouth Aspect Ratio) = lip_height / lip_width
            # 2. Lip Width = distance between left and right corners
            if left_corner_remapped is not None and right_corner_remapped is not None:
                # Lip width: distance between left and right corners
                lip_width = torch.norm(
                    coords[:, right_corner_remapped, :] - coords[:, left_corner_remapped, :],
                    dim=1
                )  # (frames,)
                
                # MAR = height / width (avoid division by zero)
                mar = lip_height / (lip_width + 1e-6)  # (frames,)
                global_features[:, 0] = mar
                global_features[:, 1] = lip_width
            else:
                # Fallback: use lip bounding box
                lip_x_min = lip_coords[:, :, 0].min(dim=1)[0]
                lip_x_max = lip_coords[:, :, 0].max(dim=1)[0]
                lip_width = lip_x_max - lip_x_min
                
                mar = lip_height / (lip_width + 1e-6)
                global_features[:, 0] = mar
                global_features[:, 1] = lip_width
            
            # 3. Lip Height (already computed)
            global_features[:, 2] = lip_height
            
            # 4. Jaw Height: distance from chin center to upper lip
            if chin_center_remapped is not None:
                chin_y = coords[:, chin_center_remapped, 1]  # (frames,)
                jaw_height = chin_y - lip_y_min  # (frames,) - vertical distance
                global_features[:, 3] = jaw_height
            else:
                # Fallback: use bottom of coords
                bottom_y = coords[:, :, 1].max(dim=1)[0]  # (frames,)
                jaw_height = bottom_y - lip_y_max  # Distance from bottom to lower lip
                global_features[:, 3] = jaw_height
        
        # 5. Cheek Puff: average distance of cheek nodes from face center
        if left_cheek_remapped and right_cheek_remapped:
            # Face center: average of all nodes
            face_center = coords.mean(dim=1)  # (frames, 2)
            
            # Left cheek center
            left_cheek_coords = coords[:, left_cheek_remapped, :]  # (frames, n_left_cheek, 2)
            left_cheek_center = left_cheek_coords.mean(dim=1)  # (frames, 2)
            
            # Right cheek center
            right_cheek_coords = coords[:, right_cheek_remapped, :]  # (frames, n_right_cheek, 2)
            right_cheek_center = right_cheek_coords.mean(dim=1)  # (frames, 2)
            
            # Distance from face center
            left_dist = torch.norm(left_cheek_center - face_center, dim=1)  # (frames,)
            right_dist = torch.norm(right_cheek_center - face_center, dim=1)  # (frames,)
            
            # Average cheek puff
            cheek_puff = (left_dist + right_dist) / 2.0  # (frames,)
            global_features[:, 4] = cheek_puff
        
        # 6. Lip Curvature: approximate curvature from upper and lower lip
        if lips_nodes_remapped and len(lips_nodes_remapped) >= 3:
            lip_coords = coords[:, lips_nodes_remapped, :]  # (frames, n_lip_nodes, 2)
            
            # Compute curvature as variance of Y coordinates (higher variance = more curved)
            # For each frame, compute variance of lip Y coordinates
            lip_y = lip_coords[:, :, 1]  # (frames, n_lip_nodes)
            lip_y_mean = lip_y.mean(dim=1, keepdim=True)  # (frames, 1)
            lip_y_var = ((lip_y - lip_y_mean) ** 2).mean(dim=1)  # (frames,)
            
            # Alternative: use range (max - min) as curvature measure
            lip_y_range = lip_y.max(dim=1)[0] - lip_y.min(dim=1)[0]  # (frames,)
            
            # Combine variance and range for better curvature measure
            lip_curvature = lip_y_var * lip_y_range  # (frames,)
            global_features[:, 5] = lip_curvature
        
        # 7. Lip Corner Angle: angle between left and right corners (relative to horizontal)
        if left_corner_remapped is not None and right_corner_remapped is not None:
            left_corner_coords = coords[:, left_corner_remapped, :]  # (frames, 2)
            right_corner_coords = coords[:, right_corner_remapped, :]  # (frames, 2)
            
            # Vector from left to right corner
            corner_vector = right_corner_coords - left_corner_coords  # (frames, 2)
            
            # Angle relative to horizontal (atan2 of y/x)
            corner_angle = torch.atan2(corner_vector[:, 1], corner_vector[:, 0])  # (frames,)
            # Convert to degrees for better interpretability (optional, can keep in radians)
            corner_angle_deg = corner_angle * 180.0 / math.pi  # (frames,)
            global_features[:, 6] = corner_angle_deg
        
        # 8. Jaw Opening: distance from chin center to lower lip
        if chin_center_remapped is not None and lips_nodes_remapped:
            chin_coords = coords[:, chin_center_remapped, :]  # (frames, 2)
            lip_coords = coords[:, lips_nodes_remapped, :]  # (frames, n_lip_nodes, 2)
            lower_lip_y = lip_coords[:, :, 1].max(dim=1)[0]  # (frames,) - lower lip Y
            
            # Vertical distance from chin to lower lip
            jaw_opening = lower_lip_y - chin_coords[:, 1]  # (frames,)
            global_features[:, 7] = jaw_opening
        elif lips_nodes_remapped:
            # Fallback: use bottom of coords if chin not available
            lip_coords = coords[:, lips_nodes_remapped, :]  # (frames, n_lip_nodes, 2)
            lower_lip_y = lip_coords[:, :, 1].max(dim=1)[0]  # (frames,)
            bottom_y = coords[:, :, 1].max(dim=1)[0]  # (frames,)
            jaw_opening = bottom_y - lower_lip_y
            global_features[:, 7] = jaw_opening
        
        return global_features
    
    def compute_geometric_features(
        self,
        coords: torch.Tensor,
        partition: str,
        anchor_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute geometric features: pairwise distances, angles, ratios.
        Vectorized for performance.
        
        Args:
            coords: Shape (frames, n_nodes, 2)
            partition: 'lips', 'mouth', or 'full'
            anchor_idx: Pre-computed anchor index (optional, for optimization)
            
        Returns:
            Geometric features of shape (frames, n_nodes, n_geom_features)
        """
        frames, n_nodes, _ = coords.shape
        device = coords.device
        dtype = coords.dtype
        
        # 1. Pairwise distances to anchor nodes (vectorized)
        # Reduced to 1 anchor for memory optimization
        # OPTIMIZATION: Compute anchor index once if not provided
        if anchor_idx is None:
            from preprocessing.mediapipe_nodes import get_partition_nodes
            nodes = get_partition_nodes(partition)
            nose_tip_mp = 4  # MediaPipe nose tip landmark
            
            if partition == 'full' and nose_tip_mp in nodes:
                # Use nose tip as anchor for full partition (more stable, central reference)
                anchor_remapped_idx = nodes.index(nose_tip_mp)
            else:
                # Use node 0 for lips/mouth partitions (nose tip not available)
                anchor_remapped_idx = 0
        else:
            anchor_remapped_idx = anchor_idx
        
        # OPTIMIZED: Direct distance computation without creating large intermediate tensor
        anchor_coords = coords[:, anchor_remapped_idx:anchor_remapped_idx+1, :]  # (frames, 1, 2)
        
        # Compute distances more efficiently: (frames, n_nodes)
        # Use broadcasting: coords (frames, n_nodes, 2) - anchor_coords (frames, 1, 2)
        diff = coords - anchor_coords  # (frames, n_nodes, 2)
        distances = torch.norm(diff, dim=2, keepdim=True)  # (frames, n_nodes, 1)
        
        # 2. Angles between consecutive nodes (vectorized)
        # Compute vectors: prev->current and current->next
        # For each node i: vec1 = coords[i] - coords[i-1], vec2 = coords[i+1] - coords[i]
        # Edge cases: first node uses coords[0] as vec1, last node uses coords[-1] as vec2
        
        # Shift coordinates for vector computation
        coords_prev = torch.cat([coords[:, 0:1, :], coords[:, :-1, :]], dim=1)  # Shift right (prev node)
        coords_next = torch.cat([coords[:, 1:, :], coords[:, -1:, :]], dim=1)   # Shift left (next node)
        
        vec1 = coords - coords_prev  # (frames, n_nodes, 2)
        vec2 = coords_next - coords  # (frames, n_nodes, 2)
        
        # Compute dot products and norms (vectorized)
        dot_products = torch.sum(vec1 * vec2, dim=2)  # (frames, n_nodes)
        norm1 = torch.norm(vec1, dim=2)  # (frames, n_nodes)
        norm2 = torch.norm(vec2, dim=2)  # (frames, n_nodes)
        
        # Compute angles (vectorized)
        # Avoid division by zero
        valid_mask = (norm1 > 1e-6) & (norm2 > 1e-6)
        cos_angles = torch.zeros_like(dot_products)
        cos_angles[valid_mask] = dot_products[valid_mask] / (norm1[valid_mask] * norm2[valid_mask])
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
        angles = torch.acos(cos_angles)  # (frames, n_nodes)
        
        # Ratio removed for memory optimization (not meaningful for full partition)
        # Stack all features: distances (1), angles (1)
        # Shape: (frames, n_nodes, 2)
        geom_features = torch.cat([
            distances,                  # (frames, n_nodes, 1)
            angles.unsqueeze(2),        # (frames, n_nodes, 1)
        ], dim=2)
        
        return geom_features
    
    def compute_B3(
        self,
        landmarks: torch.Tensor,
        meta: Dict,
        partition: str,
        node_mapping: Dict,
        b0_coords: Optional[torch.Tensor] = None,
        b2_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        B3: Returns B0 + B1 + B2 + AU features (cumulative, includes B0+B1+B2).
        PCA and motion energy removed for aggressive memory optimization.
        
        Args:
            landmarks: Shape (frames, n_nodes, 2)
            meta: Video metadata
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Original MediaPipe index -> new index
            b0_coords: Optional pre-computed B0 coordinates (faster)
            b2_features: Optional pre-computed B2 features (B0+B1+B2, faster, avoids recomputation)
            
        Returns:
            Tuple of (features, additional_meta) - B0+B1+B2+B3 (11 features total)
        """
        # OPTIMIZATION: Use existing B0+B1+B2 if provided (much faster)
        if b2_features is not None:
            b0_b1_b2 = b2_features  # Already has B0+B1+B2
            b0 = b2_features[:, :, :2]  # Extract B0 for AU computation
        else:
            # Compute B0
            if b0_coords is not None:
                b0 = b0_coords
            else:
                b0 = self.compute_B0(landmarks, meta)
            
            # Compute B2 (B0+B1+B2)
            b2 = self.compute_B2(landmarks, meta, partition, b0_coords=b0)  # Returns B0+B1+B2
            b0_b1_b2 = b2
        
        # AU features only (B3 incremental)
        au_features, au_groups = self.compute_AU_features(b0, partition, node_mapping)
        
        # Return B0 + B1 + B2 + B3 (cumulative)
        features = torch.cat([b0_b1_b2, au_features], dim=-1)  # (frames, n_nodes, 11)
        
        # Additional meta
        additional_meta = {
            'au_groups': au_groups,
            'pca_n_components': 0,  # PCA removed
        }
        
        return features, additional_meta
    
    def compute_AU_features(
        self,
        coords: torch.Tensor,
        partition: str,
        node_mapping: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Action Unit inspired features.
        
        Args:
            coords: Shape (frames, n_nodes, 2)
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Original MediaPipe index -> new index
            
        Returns:
            Tuple of (AU features, AU node mapping)
        """
        frames, n_nodes, _ = coords.shape
        
        # Define AU groups based on anatomical regions (not index ranges)
        from preprocessing.mediapipe_nodes import get_au_node_groups
        au_groups = get_au_node_groups(partition, node_mapping)
        
        # Compute AU group displacements (VECTORIZED)
        # Process all frames at once
        n_au_groups = len([g for g in au_groups.values() if len(g) > 0])
        if n_au_groups == 0:
            return torch.zeros(frames, n_nodes, 1, dtype=torch.float32, device=coords.device), au_groups
        
        au_features_per_group = []
        
        for au_name, node_indices in au_groups.items():
            if len(node_indices) == 0:
                continue
            
            # Extract group coordinates for all frames: (frames, n_group_nodes, 2)
            group_coords = coords[:, node_indices, :]
            
            # Compute displacement magnitude for all frames: (frames, n_group_nodes)
            group_displacement = torch.norm(group_coords, dim=2)
            
            # Mean across group nodes: (frames,)
            group_mean = group_displacement.mean(dim=1)
            
            au_features_per_group.append(group_mean)
        
        # Stack all AU groups: (frames, n_au_groups)
        au_features_flat = torch.stack(au_features_per_group, dim=1)
        
        # Broadcast to all nodes: (frames, n_nodes, n_au_groups)
        au_features = au_features_flat.unsqueeze(1).expand(-1, n_nodes, -1)
        
        return au_features, au_groups
    
    def compute_PCA_features(
        self,
        features: torch.Tensor,
        n_components: int = 10
    ) -> Tuple[torch.Tensor, Optional[PCA]]:
        """
        Compute PCA modes from features.
        
        Args:
            features: Shape (frames, n_nodes, n_features)
            n_components: Number of PCA components
            
        Returns:
            Tuple of (PCA features, PCA model)
        """
        frames, n_nodes, n_features = features.shape
        
        # Reshape to (frames * n_nodes, n_features) for PCA
        features_flat = features.reshape(-1, n_features).numpy()
        
        # Fit PCA
        n_components = min(n_components, n_features, len(features_flat))
        if n_components > 0:
            pca = PCA(n_components=n_components)
            pca_features_flat = pca.fit_transform(features_flat)
            
            # Reshape back to (frames, n_nodes, n_components)
            pca_features = torch.from_numpy(pca_features_flat).reshape(frames, n_nodes, n_components)
            
            return pca_features, pca
        else:
            return torch.zeros(frames, n_nodes, 1), None
    
    def compute_motion_energy(
        self,
        coords: torch.Tensor,
        window_size: int = 5
    ) -> torch.Tensor:
        """
        Compute motion energy features (temporal variance, motion magnitude).
        
        Args:
            coords: Shape (frames, n_nodes, 2)
            window_size: Temporal window size for computing motion
            
        Returns:
            Motion energy features of shape (frames, n_nodes, n_energy_features)
        """
        frames, n_nodes, _ = coords.shape
        
        # Compute velocity
        velocity = torch.diff(coords, dim=0)  # (frames-1, n_nodes, 2)
        velocity = torch.cat([velocity[0:1], velocity], dim=0)  # Pad first frame
        
        # Motion magnitude
        motion_magnitude = torch.norm(velocity, dim=2)  # (frames, n_nodes)
        
        # Temporal variance (sliding window) - VECTORIZED
        # Use unfold to create sliding windows efficiently
        half_window = window_size // 2
        padded_motion = torch.nn.functional.pad(
            motion_magnitude.unsqueeze(0), 
            (0, 0, half_window, half_window), 
            mode='reflect'
        ).squeeze(0)  # (frames + 2*half_window, n_nodes)
        
        # Create sliding windows using unfold
        # unfold(0, window_size, 1) on (frames + 2*half_window, n_nodes) gives:
        # (frames + 2*half_window - window_size + 1, n_nodes, window_size)
        # For window_size=5, half_window=2: (frames + 4 - 5 + 1, n_nodes, 5) = (frames, n_nodes, 5)
        windows = padded_motion.unfold(0, window_size, 1)  # (frames, n_nodes, window_size)
        # No permute needed - unfold already gives correct shape for computing stats over window_size
        
        # Compute statistics across window dimension
        # Reduced from 3 to 2 features (removed variance) to save memory while keeping all 468 nodes
        mean_motion = windows.mean(dim=2)  # (output_frames, n_nodes)
        max_motion = windows.max(dim=2)[0]  # (output_frames, n_nodes)
        
        # unfold reduces frames: output_frames = frames + 2*half_window - window_size + 1
        # For window_size=5, half_window=2: output_frames = frames + 4 - 5 + 1 = frames
        # But to be safe, pad/trim to match original frame count
        output_frames = mean_motion.shape[0]
        if output_frames < frames:
            # Pad with edge values
            pad_before = (frames - output_frames) // 2
            pad_after = frames - output_frames - pad_before
            mean_motion = torch.nn.functional.pad(mean_motion, (0, 0, pad_before, pad_after), mode='replicate')
            max_motion = torch.nn.functional.pad(max_motion, (0, 0, pad_before, pad_after), mode='replicate')
        elif output_frames > frames:
            # Trim to match
            trim_before = (output_frames - frames) // 2
            mean_motion = mean_motion[trim_before:trim_before + frames]
            max_motion = max_motion[trim_before:trim_before + frames]

        # Stack features: (frames, n_nodes, 2) - reduced from 3 to save memory
        motion_energy = torch.stack([mean_motion, max_motion], dim=2)
        
        return motion_energy
    
    
    def compute_features(
        self,
        landmarks: torch.Tensor,
        meta: Dict,
        partition: str = 'lips',
        node_mapping: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute features based on feature level.
        
        Args:
            landmarks: Shape (frames, n_nodes, 2)
            meta: Video metadata
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Node mapping (required for B3)
            
        Returns:
            Tuple of (features, additional_meta)
        """
        if self.feature_level == 'B0':
            features = self.compute_B0(landmarks, meta)
            additional_meta = {}
        
        elif self.feature_level == 'B1':
            features = self.compute_B1(landmarks, meta)
            additional_meta = {}
        
        elif self.feature_level == 'B2':
            # B2 can accept optional b0_coords for optimization (passed via compute_features)
            features = self.compute_B2(landmarks, meta, partition)
            additional_meta = {}
        
        elif self.feature_level == 'B3':
            if node_mapping is None:
                raise ValueError("node_mapping required for B3 features")
            features, additional_meta = self.compute_B3(landmarks, meta, partition, node_mapping)
        
        else:
            raise ValueError(f"Unknown feature level: {self.feature_level}")
        
        return features, additional_meta


def process_split_features(
    extracted_data_path: str,
    feature_level: str,
    output_path: str,
    b0_features_path: Optional[str] = None,
    b1_features_path: Optional[str] = None,
    b2_features_path: Optional[str] = None
) -> None:
    """
    Process features for an entire split.
    Each feature level (B0-B3) stores CUMULATIVE features (includes all previous levels).
    
    Storage format (cumulative):
    - B0: B0 features only (2 features: x, y)
    - B1: B0 + B1 features (5 features: B0: 2 + B1: 3)
    - B2: B0 + B1 + B2 features (7 features: B0: 2 + B1: 3 + B2: 2)
    - B3: B0 + B1 + B2 + B3 features (11 features: B0: 2 + B1: 3 + B2: 2 + B3: 4)
    
    When loading for training, just load the target level file (no concatenation needed).
    
    Args:
        extracted_data_path: Path to extracted landmarks .pt file
        feature_level: 'B0', 'B1', 'B2', or 'B3'
        output_path: Path to save features .pt file
    """
    from pathlib import Path
    from utils import setup_logger, ensure_dir
    import gc
    
    logger = setup_logger('FeatureEngineering')
    logger.info(f"Processing features: {feature_level}")
    logger.info(f"Input: {extracted_data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Each feature level ({feature_level}) is computed independently and saved to unique file")
    
    # Load extracted data
    logger.info(f"Loading extracted data...")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        extracted_data = torch.load(extracted_data_path, map_location='cpu', weights_only=False)
    
    # Extract metadata
    partition = extracted_data['partition']
    split = extracted_data['split']
    node_mapping = extracted_data.get('node_mapping', {})
    adjacency = extracted_data['adjacency']
    word_to_label = extracted_data['word_to_label']
    videos = extracted_data['videos']
    
    total_videos = len(videos)
    logger.info(f"Total videos: {total_videos}")
    
    # OPTIMIZATION: Precompute node indices for B2 (avoid repeated lookups per video)
    b2_node_indices = None
    if feature_level == 'B2':
        from preprocessing.mediapipe_nodes import get_partition_nodes, get_lips_nodes
        
        nodes = get_partition_nodes(partition)
        node_mapping_dict = {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
        
        # Get specific landmark indices (original MediaPipe indices)
        lips_nodes_orig = get_lips_nodes()
        lips_nodes_remapped = [node_mapping_dict[n] for n in lips_nodes_orig if n in node_mapping_dict]
        
        # Lip corner indices (original MediaPipe)
        left_corner_outer = 61
        right_corner_outer = 291
        left_corner_inner = 78
        right_corner_inner = 308
        chin_center_orig = 152
        
        # Cheek nodes (original MediaPipe)
        left_cheek_orig = [50, 118, 119, 100, 101, 36, 203, 205, 206, 216]
        right_cheek_orig = [280, 347, 348, 330, 329, 266, 423, 425, 426, 436]
        
        # Map to remapped indices
        left_corner_remapped = node_mapping_dict.get(left_corner_outer, node_mapping_dict.get(left_corner_inner, None))
        right_corner_remapped = node_mapping_dict.get(right_corner_outer, node_mapping_dict.get(right_corner_inner, None))
        chin_center_remapped = node_mapping_dict.get(chin_center_orig, None)
        left_cheek_remapped = [node_mapping_dict[n] for n in left_cheek_orig if n in node_mapping_dict]
        right_cheek_remapped = [node_mapping_dict[n] for n in right_cheek_orig if n in node_mapping_dict]
        
        b2_node_indices = {
            'left_corner': left_corner_remapped,
            'right_corner': right_corner_remapped,
            'chin_center': chin_center_remapped,
            'lips_nodes': lips_nodes_remapped,
            'left_cheek': left_cheek_remapped,
            'right_cheek': right_cheek_remapped
        }
        logger.info(f"✓ Precomputed B2 node indices for partition '{partition}'")
    
    # Initialize feature engineer (each level computes independently from landmarks)
    fe = FeatureEngineer(feature_level=feature_level)
    
    # OPTIMIZATION: Load previous level features if available (for faster computation)
    # B1 can use B0, B2 can use B1 (which includes B0), B3 can use B2 (which includes B0+B1+B2)
    b0_features_data = None
    b1_features_data = None
    b2_features_data = None
    
    if feature_level == 'B1' and b0_features_path and Path(b0_features_path).exists():
        logger.info(f"Loading existing B0 features from {b0_features_path} for faster B1 computation...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                b0_features_data = torch.load(b0_features_path, map_location='cpu')
            logger.info(f"✓ Loaded B0 features for {len(b0_features_data.get('videos', {}))} videos")
        except Exception as e:
            logger.warning(f"Failed to load B0 features: {e}. Will compute from landmarks (slower).")
            b0_features_data = None
    
    elif feature_level == 'B2':
        # B2 can be computed directly from extracted data (landmarks) without needing B0/B1 files
        # Optional: Load B1/B0 if available for speedup, but not required
        if b1_features_path and Path(b1_features_path).exists():
            logger.info(f"[OPTIONAL] Loading existing B1 features (B0+B1) from {b1_features_path} for faster B2 computation...")
            logger.info(f"Note: B2 can be computed directly from extracted data without B0/B1 files")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    b1_features_data = torch.load(b1_features_path, map_location='cpu', weights_only=False)
                logger.info(f"✓ Loaded B1 features for {len(b1_features_data.get('videos', {}))} videos (speedup mode)")
            except Exception as e:
                logger.warning(f"Failed to load B1 features: {e}. Will compute B2 directly from landmarks.")
                b1_features_data = None
        elif b0_features_path and Path(b0_features_path).exists():
            logger.info(f"[OPTIONAL] Loading existing B0 features from {b0_features_path} for faster B2 computation...")
            logger.info(f"Note: B2 can be computed directly from extracted data without B0/B1 files")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    b0_features_data = torch.load(b0_features_path, map_location='cpu', weights_only=False)
                logger.info(f"✓ Loaded B0 features for {len(b0_features_data.get('videos', {}))} videos (speedup mode)")
            except Exception as e:
                logger.warning(f"Failed to load B0 features: {e}. Will compute B2 directly from landmarks.")
                b0_features_data = None
        else:
            logger.info(f"Computing B2 directly from extracted data (no B0/B1 files needed)")
            logger.info(f"B2 will compute B0+B1+B2 from landmarks in one pass")
            logger.info(f"Result will be stored in B2 file with all features (B0+B1+B2 cumulative)")
            logger.info(f"No separate B0/B1 files will be created - everything in B2 file")
    
    elif feature_level == 'B3':
        if b2_features_path and Path(b2_features_path).exists():
            logger.info(f"Loading existing B2 features (B0+B1+B2) from {b2_features_path} for faster B3 computation...")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    b2_features_data = torch.load(b2_features_path, map_location='cpu', weights_only=False)
                logger.info(f"✓ Loaded B2 features for {len(b2_features_data.get('videos', {}))} videos")
            except Exception as e:
                logger.warning(f"Failed to load B2 features: {e}. Will compute from landmarks (slower).")
                b2_features_data = None
        elif b0_features_path and Path(b0_features_path).exists():
            logger.info(f"Loading existing B0 features from {b0_features_path} for faster B3 computation...")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    b0_features_data = torch.load(b0_features_path, map_location='cpu', weights_only=False)
                logger.info(f"✓ Loaded B0 features for {len(b0_features_data.get('videos', {}))} videos")
            except Exception as e:
                logger.warning(f"Failed to load B0 features: {e}. Will compute from landmarks (slower).")
                b0_features_data = None
    
    # Initialize output structure
    feature_data = {
        'split': split,
        'partition': partition,
        'feature_level': feature_level,  # Each file has unique feature_level
        'adjacency': adjacency.clone() if isinstance(adjacency, torch.Tensor) else adjacency,
        'node_mapping': node_mapping,
        'word_to_label': word_to_label.copy(),
        'videos': {},
        'meta': {}
    }
    
    # Clear references to reduce memory
    del adjacency, word_to_label, extracted_data
    
    ensure_dir(Path(output_path).parent)
    
    # Process all videos
    processed_count = 0
    logger.info(f"Processing {total_videos} videos...")
    
    # Add progress bar for long-running feature computation
    video_items = list(videos.items())
    progress_bar = tqdm(
        video_items,
        desc=f"Computing {feature_level} features",
        unit="video",
        total=total_videos,
        disable=False  # Always show progress bar
    )
    
    for video_id, video_data in progress_bar:
        # Extract video data
        landmarks = video_data['landmarks'].clone()  # Clone to avoid keeping reference
        meta = video_data['meta']
        video_path = video_data['video_path']
        word = video_data['word']
        label = video_data['label']
        speech_mask = video_data['speech_mask']
        
        # Compute features (cumulative: each level includes all previous levels)
        # OPTIMIZATION: Use existing previous level features if available (much faster)
        if feature_level == 'B1' and b0_features_data is not None:
            video_b0_features = b0_features_data['videos'].get(video_id)
            if video_b0_features is not None:
                # Use existing B0 features (faster - no B0 recomputation)
                b0_coords = video_b0_features['features']  # B0 only (2 features)
                # Compute B1 incremental features (velocity + speed)
                velocity = fe.compute_velocity(b0_coords)
                speed = fe.compute_speed(velocity)
                b1_incremental = torch.cat([velocity, speed], dim=-1)  # (frames, n_nodes, 3)
                # Concatenate B0 + B1 (cumulative)
                features = torch.cat([b0_coords, b1_incremental], dim=-1)  # (frames, n_nodes, 5)
                additional_meta = {}
            else:
                features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
        
        elif feature_level == 'B2':
            if b1_features_data is not None:
                video_b1_features = b1_features_data['videos'].get(video_id)
                if video_b1_features is not None:
                    # Use existing B1 features (B0+B1) - fastest path
                    b1_features = video_b1_features['features']  # B0+B1 (7 features)
                    b0_coords = b1_features[:, :, :2]  # Extract B0 for geometric computation
                    # OPTIMIZATION: Pass precomputed node_indices to avoid repeated lookups
                    features = fe.compute_B2(landmarks, meta, partition, b0_coords=b0_coords, b1_features=b1_features, node_indices=b2_node_indices)
                    additional_meta = {}
                else:
                    features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
            elif b0_features_data is not None:
                video_b0_features = b0_features_data['videos'].get(video_id)
                if video_b0_features is not None:
                    # Use existing B0 features
                    b0_coords = video_b0_features['features']  # B0 only (2 features)
                    # OPTIMIZATION: Pass precomputed node_indices to avoid repeated lookups
                    features = fe.compute_B2(landmarks, meta, partition, b0_coords=b0_coords, node_indices=b2_node_indices)
                    additional_meta = {}
                else:
                    features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
            else:
                # OPTIMIZATION: Pass precomputed node_indices even when computing from scratch
                if b2_node_indices is not None:
                    features = fe.compute_B2(landmarks, meta, partition, node_indices=b2_node_indices)
                    additional_meta = {}
                else:
                    features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
        
        elif feature_level == 'B3':
            if b2_features_data is not None:
                video_b2_features = b2_features_data['videos'].get(video_id)
                if video_b2_features is not None:
                    # Use existing B2 features (B0+B1+B2) - fastest path
                    b2_features = video_b2_features['features']  # B0+B1+B2 (7 features)
                    b0_coords = b2_features[:, :, :2]  # Extract B0 for AU computation
                    features, additional_meta = fe.compute_B3(landmarks, meta, partition, node_mapping, b0_coords=b0_coords, b2_features=b2_features)
                else:
                    features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
            elif b0_features_data is not None:
                video_b0_features = b0_features_data['videos'].get(video_id)
                if video_b0_features is not None:
                    # Use existing B0 features
                    b0_coords = video_b0_features['features']  # B0 only (2 features)
                    features, additional_meta = fe.compute_B3(landmarks, meta, partition, node_mapping, b0_coords=b0_coords)
                else:
                    features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
            else:
                features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
        
        else:
            # B0: Normal computation
            features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
        
        # Store processed video (only features for this level)
        # Keep as float32 for accuracy (float16 causes precision loss and accuracy degradation)
        # Memory optimization is handled via lazy loading and num_workers=0 instead
        if features.dtype != torch.float32:
            features = features.float()  # Ensure float32
        
        feature_data['videos'][video_id] = {
            'video_path': video_path,
            'word': word,
            'label': label,
            'features': features,  # Stored as float32 for accuracy
            'speech_mask': speech_mask,
            'meta': meta,
        }
        
        # Update global meta from first video
        if processed_count == 0:
            feature_data['meta']['n_features'] = features.shape[-1]
            feature_data['meta']['n_nodes'] = features.shape[1]
            if additional_meta:
                feature_data['meta'].update(additional_meta)
    
        processed_count += 1
        
        # Update progress bar description with count
        if processed_count % 100 == 0 or processed_count == total_videos:
            progress_bar.set_postfix({
                'processed': f'{processed_count}/{total_videos}',
                'progress': f'{100*processed_count/total_videos:.1f}%'
            })
        
        # Clear video data immediately (no overlap with other levels)
        del landmarks, features
    
    # Close progress bar
    progress_bar.close()
    
    # Save final result
    # Note: torch.save() uses zipfile format by default (since PyTorch 1.6)
    # Features are stored as float16 for disk efficiency (2x size reduction)
    # Converted to float32 during __getitem__ for training accuracy
    torch.save(feature_data, output_path)
    
    # Add metadata flag indicating float16 storage
    feature_data['meta']['storage_dtype'] = 'float16'
    
    # Clear remaining data
    del videos, feature_data
    gc.collect()
    
    logger.info(f"Saved features to: {output_path}")
    logger.info(f"Processed {processed_count} videos total")
    
    # Log size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"File size: {size_mb:.2f} MB")

