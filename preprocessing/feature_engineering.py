"""
Feature Engineering: B0 - B3 feature sets.

Each level stores CUMULATIVE features (includes all previous levels):
- B0: Raw normalized coordinates (X, Y, Z per node) - 3 features (3D coordinates)
- B1: B0 + Velocity + speed + acceleration - 10 features (B0: 3 + B1: 7 = 10 total)
  Note: Velocity (vx, vy, vz), speed (3D magnitude), acceleration (ax, ay, az) all in 3D
- B2: B0 + B1 + Global geometric features - 18 features (B0: 3 + B1: 7 + B2: 8 = 18 total)
  Note: B2 features are global per frame (broadcast to all nodes): MAR, lip width, lip height, jaw height, cheek puff, lip curvature, lip corner angle, jaw opening
  Note: Width and distances use 3D Euclidean distance for accurate measurements regardless of viewing angle
- B3: B0 + B1 + B2 + AU features - 22 features (B0: 3 + B1: 7 + B2: 8 + B3: 4 = 22 total)
  Note: AU features: 4 AU groups (AU25, AU26, AU12, AU27) using 3D displacement magnitudes. PCA and motion energy removed.

When loading for training, just load the target level file (no concatenation needed).
Each file is self-contained with all features up to that level.

IMPORTANT: All features now use 3D coordinates (X, Y, Z) for accurate measurements:
- Distances use 3D Euclidean distance: sqrt((x_diff)^2 + (y_diff)^2 + (z_diff)^2)
- This ensures consistent measurements regardless of viewing angle (facing camera vs. facing left/right)
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
import pickle


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
    
    def compute_B0(self, landmarks: torch.Tensor, meta: Dict, partition: str = 'mouth', node_mapping: Optional[Dict] = None) -> torch.Tensor:
        """
        B0: Raw normalized coordinates with pose, scale, and rotation invariance.
        
        Implements proper 3D normalization following the order:
        1. Pose Invariance: Subtract nose tip (anchor point) from all landmarks
        2. 3D Rotation Correction: Align face horizontally using 3D rotation matrix
        3. Scale Invariance: Normalize by IOD (full) or mouth width (mouth/lips)
        
        Input landmarks are already normalized to [0, 1] by resolution during extraction.
        MediaPipe coordinate system: X [0,1] (left to right), Y [0,1] (top to bottom), Z (depth).
        
        Args:
            landmarks: Shape (frames, n_nodes, 3) - MediaPipe normalized coordinates [0,1] for X,Y, Z for depth
            meta: Video metadata with width, height
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Optional node mapping dict (original MediaPipe idx -> new idx)
            
        Returns:
            Features of shape (frames, n_nodes, 3) - Normalized X, Y, Z coordinates
            Note: Z is preserved for 3D distance calculations in B2 geometric features
        """
        frames, n_nodes, n_dims = landmarks.shape
        
        # Handle both 2D (old) and 3D (new) landmarks
        if n_dims == 2:
            # Old format: add Z as zeros (backward compatibility)
            z_coords = torch.zeros(frames, n_nodes, 1, dtype=landmarks.dtype, device=landmarks.device)
            landmarks = torch.cat([landmarks, z_coords], dim=2)
        elif n_dims != 3:
            raise ValueError(f"Expected landmarks with 2 or 3 dimensions, got {n_dims}")
        
        # Get node indices for normalization
        if node_mapping is None:
            from preprocessing.mediapipe_nodes import get_partition_nodes
            nodes = get_partition_nodes(partition)
            node_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
        
        # Step 1: Pose Invariance - Subtract nose tip (anchor point) from all landmarks
        landmarks_pose_invariant = self._apply_pose_invariance(landmarks, partition, node_mapping)
        
        # Step 2: 3D Rotation Correction - Align face horizontally using 3D rotation
        landmarks_rotated = self._correct_rotation_3d(landmarks_pose_invariant, partition, node_mapping)
        
        # Step 3: Scale Invariance - Normalize by IOD or mouth width (using 3D distances)
        features = self._normalize_scale(landmarks_rotated, partition, node_mapping)
        
        return features  # Returns (frames, n_nodes, 3) - X, Y, Z
    
    def _apply_pose_invariance(self, landmarks: torch.Tensor, partition: str, node_mapping: Dict) -> torch.Tensor:
        """
        Step 1: Pose Invariance - Subtract nose tip (anchor point) from all landmarks.
        
        Formula: x'_i = x_i - x_nose
        
        Args:
            landmarks: Shape (frames, n_nodes, 3) - X, Y, Z coordinates
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Node mapping dict
            
        Returns:
            Pose-invariant landmarks (frames, n_nodes, 3)
        """
        frames, n_nodes, _ = landmarks.shape
        pose_invariant = landmarks.clone()
        
        # MediaPipe nose tip landmark index is 4 (not 34)
        nose_tip_mp = 4
        nose_tip_idx = node_mapping.get(nose_tip_mp)
        
        if nose_tip_idx is not None:
            # Get nose tip position for each frame
            nose_tip = pose_invariant[:, nose_tip_idx, :]  # (frames, 3)
            
            # Subtract nose tip from all landmarks
            pose_invariant = pose_invariant - nose_tip.unsqueeze(1)  # (frames, n_nodes, 3)
        else:
            # Fallback: use face/mouth center if nose tip not available
            if partition == 'full':
                # Use average of all landmarks as center
                center = pose_invariant.mean(dim=1, keepdim=True)  # (frames, 1, 3)
            else:
                # For mouth/lips: use mouth center
                left_corner_mp = 61
                right_corner_mp = 291
                left_corner_idx = node_mapping.get(left_corner_mp)
                right_corner_idx = node_mapping.get(right_corner_mp)
                
                if left_corner_idx is not None and right_corner_idx is not None:
                    left_corner = pose_invariant[:, left_corner_idx, :]  # (frames, 3)
                    right_corner = pose_invariant[:, right_corner_idx, :]  # (frames, 3)
                    center = ((left_corner + right_corner) / 2).unsqueeze(1)  # (frames, 1, 3)
                else:
                    # Final fallback: use average
                    center = pose_invariant.mean(dim=1, keepdim=True)  # (frames, 1, 3)
            
            pose_invariant = pose_invariant - center
        
        return pose_invariant
    
    def _ensure_camera_facing(self, landmarks: torch.Tensor, partition: str, node_mapping: Dict) -> torch.Tensor:
        """
        Step 0: Align face horizontally in 3D by making lip corners/eyes have the same Z (depth).
        
        For a face facing the camera and horizontally aligned:
        - Lip corners (or eyes) should have the SAME Z value (same depth)
        - This ensures the face is horizontally aligned in 3D space (no yaw rotation in depth)
        
        We rotate the face in 3D space around the Y-axis to align the corners/eyes to the same depth.
        This check must happen BEFORE pose invariance (before subtracting nose).
        
        Args:
            landmarks: Shape (frames, n_nodes, 3) - X, Y, Z coordinates (original)
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Node mapping dict
            
        Returns:
            Horizontally aligned landmarks (frames, n_nodes, 3)
        """
        frames, n_nodes, _ = landmarks.shape
        corrected = landmarks.clone()
        
        if partition == 'full':
            left_eye_mp = 33
            right_eye_mp = 263
            
            left_eye_idx = node_mapping.get(left_eye_mp)
            right_eye_idx = node_mapping.get(right_eye_mp)
            
            if left_eye_idx is not None and right_eye_idx is not None:
                # Get eye positions in 3D
                left_eye = corrected[:, left_eye_idx, :]  # (frames, 3)
                right_eye = corrected[:, right_eye_idx, :]  # (frames, 3)
                
                # Compute Z difference (depth difference)
                z_diff = right_eye[:, 2] - left_eye[:, 2]  # (frames,)
                
                # Rotate around Y-axis to make Z values equal (align horizontally in 3D)
                # Angle to rotate: atan2(z_diff, x_distance)
                x_diff = right_eye[:, 0] - left_eye[:, 0]  # (frames,)
                # Use XZ plane to compute rotation angle
                angle_y = torch.atan2(z_diff, x_diff)  # (frames,) - rotation around Y-axis
                
                # Apply rotation around Y-axis to align Z values
                cos_angle = torch.cos(-angle_y).unsqueeze(1)  # (frames, 1)
                sin_angle = torch.sin(-angle_y).unsqueeze(1)  # (frames, 1)
                
                # Rotation matrix around Y-axis: R_y = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
                x_rot = corrected[:, :, 0] * cos_angle + corrected[:, :, 2] * sin_angle
                y_rot = corrected[:, :, 1]  # Y unchanged for Y-axis rotation
                z_rot = -corrected[:, :, 0] * sin_angle + corrected[:, :, 2] * cos_angle
                
                corrected = torch.stack([x_rot, y_rot, z_rot], dim=2)  # (frames, n_nodes, 3)
        
        elif partition in ['mouth', 'lips']:
            left_corner_mp = 61
            right_corner_mp = 291
            
            left_corner_idx = node_mapping.get(left_corner_mp)
            right_corner_idx = node_mapping.get(right_corner_mp)
            
            if left_corner_idx is not None and right_corner_idx is not None:
                # Get corner positions in 3D
                left_corner = corrected[:, left_corner_idx, :]  # (frames, 3)
                right_corner = corrected[:, right_corner_idx, :]  # (frames, 3)
                
                # First, ensure face is facing RIGHT before Y-axis rotation
                # Check direction: right corner should be to the right (positive X direction)
                x_diff_before = right_corner[:, 0] - left_corner[:, 0]  # (frames,)
                facing_right_before = x_diff_before > 0  # (frames,)
                
                # If NOT facing right, flip horizontally BEFORE Y-axis rotation
                flip_mask = ~facing_right_before  # (frames,)
                if flip_mask.any():
                    corrected[flip_mask, :, 0] = -corrected[flip_mask, :, 0]  # Flip X
                    corrected[flip_mask, :, 2] = -corrected[flip_mask, :, 2]  # Flip Z
                    # Update corner positions after flip
                    left_corner = corrected[:, left_corner_idx, :]  # (frames, 3)
                    right_corner = corrected[:, right_corner_idx, :]  # (frames, 3)
                
                # Now rotate around Y-axis to make Z values equal (align horizontally in 3D)
                # Compute Z difference (depth difference)
                z_diff = right_corner[:, 2] - left_corner[:, 2]  # (frames,)
                
                # Angle to rotate: atan2(z_diff, x_distance)
                x_diff = right_corner[:, 0] - left_corner[:, 0]  # (frames,)
                # Use XZ plane to compute rotation angle
                angle_y = torch.atan2(z_diff, x_diff)  # (frames,) - rotation around Y-axis
                
                # Apply rotation around Y-axis to align Z values
                cos_angle = torch.cos(-angle_y).unsqueeze(1)  # (frames, 1)
                sin_angle = torch.sin(-angle_y).unsqueeze(1)  # (frames, 1)
                
                # Rotation matrix around Y-axis: R_y = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
                x_rot = corrected[:, :, 0] * cos_angle + corrected[:, :, 2] * sin_angle
                y_rot = corrected[:, :, 1]  # Y unchanged for Y-axis rotation
                z_rot = -corrected[:, :, 0] * sin_angle + corrected[:, :, 2] * cos_angle
                
                corrected = torch.stack([x_rot, y_rot, z_rot], dim=2)  # (frames, n_nodes, 3)
                
                # Verify direction is still correct after Y-axis rotation
                right_corner_x_after = corrected[:, right_corner_idx, 0]  # (frames,)
                left_corner_x_after = corrected[:, left_corner_idx, 0]  # (frames,)
                facing_right_after = right_corner_x_after > left_corner_x_after  # (frames,)
                
                # If direction flipped during rotation, flip back
                flip_mask_after = ~facing_right_after  # (frames,)
                if flip_mask_after.any():
                    corrected[flip_mask_after, :, 0] = -corrected[flip_mask_after, :, 0]  # Flip X
                    corrected[flip_mask_after, :, 2] = -corrected[flip_mask_after, :, 2]  # Flip Z
        
        return corrected
    
    def _correct_rotation_3d(self, landmarks: torch.Tensor, partition: str, node_mapping: Dict) -> torch.Tensor:
        """
        Step 2: 3D Rotation Correction - Align face horizontally and ensure facing CAMERA.
        
        For a face to be "facing CAMERA" (toward camera):
        1. Face must be horizontal (no tilt) - align corners/eyes horizontally
        2. Right side must be on positive X, left side on negative X
        3. Face should be facing camera - use Z coordinate to verify:
           - Nose tip should be CLOSEST to camera (highest Z value)
           - Face edges should be FARTHER from camera (lower Z values)
        
        Uses X, Y, Z coordinates to:
        - Rotate face horizontally (2D rotation in XY plane)
        - Determine face direction using Z coordinate (depth)
        - Ensure all faces face the camera (front-facing)
        
        Args:
            landmarks: Shape (frames, n_nodes, 3) - X, Y, Z coordinates (already pose-invariant)
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Node mapping dict
            
        Returns:
            Rotation-corrected landmarks (frames, n_nodes, 3) - but only X, Y are used later
        """
        frames, n_nodes, _ = landmarks.shape
        corrected = landmarks.clone()
        
        if partition == 'full':
            # Use eyes for rotation (most stable)
            left_eye_mp = 33  # Left eye center
            right_eye_mp = 263  # Right eye center
            nose_tip_mp = 4  # Nose tip
            
            left_eye_idx = node_mapping.get(left_eye_mp)
            right_eye_idx = node_mapping.get(right_eye_mp)
            nose_tip_idx = node_mapping.get(nose_tip_mp)
            
            if left_eye_idx is not None and right_eye_idx is not None:
                # Get eye positions in 3D
                left_eye = corrected[:, left_eye_idx, :]  # (frames, 3)
                right_eye = corrected[:, right_eye_idx, :]  # (frames, 3)
                eye_vector = right_eye - left_eye  # (frames, 3)
                
                # Step 1: Rotate to make eyes horizontal (Y=0)
                angle = torch.atan2(eye_vector[:, 1], eye_vector[:, 0])  # (frames,)
                cos_angle = torch.cos(-angle).unsqueeze(1)  # (frames, 1)
                sin_angle = torch.sin(-angle).unsqueeze(1)  # (frames, 1)
                
                # Apply 3D rotation matrix around Z-axis (yaw correction)
                x_rot = corrected[:, :, 0] * cos_angle - corrected[:, :, 1] * sin_angle
                y_rot = corrected[:, :, 0] * sin_angle + corrected[:, :, 1] * cos_angle
                z_rot = corrected[:, :, 2]  # Z unchanged for yaw rotation
                
                corrected = torch.stack([x_rot, y_rot, z_rot], dim=2)  # (frames, n_nodes, 3)
                
                # Step 2: Ensure all faces face CAMERA using Z coordinate
                # For face facing camera: nose should be closest (highest Z)
                if nose_tip_idx is not None:
                    nose_z = corrected[:, nose_tip_idx, 2]  # (frames,)
                    left_eye_z = corrected[:, left_eye_idx, 2]  # (frames,)
                    right_eye_z = corrected[:, right_eye_idx, 2]  # (frames,)
                    
                    # Check if nose is closest (facing camera)
                    nose_is_closest = (nose_z > left_eye_z) & (nose_z > right_eye_z)  # (frames,)
                    
                    # If nose is NOT closest, face is not facing camera - need to flip
                    # Flip around Y-axis (mirror horizontally) to face camera
                    flip_mask = ~nose_is_closest  # (frames,)
                else:
                    # Fallback: use X direction if nose not available
                    right_eye_rotated = corrected[:, right_eye_idx, 0]  # (frames,)
                    left_eye_rotated = corrected[:, left_eye_idx, 0]  # (frames,)
                    eye_direction = right_eye_rotated - left_eye_rotated  # (frames,)
                    flip_mask = eye_direction < 0  # (frames,)
                
                # Flip horizontally if not facing camera
                if flip_mask.any():
                    corrected[flip_mask, :, 0] = -corrected[flip_mask, :, 0]  # Flip X
                    # Also flip Z to maintain consistency (though Z is not used later)
                    corrected[flip_mask, :, 2] = -corrected[flip_mask, :, 2]  # Flip Z
        
        elif partition in ['mouth', 'lips']:
            # Use lip corners for rotation
            left_corner_mp = 61  # Left lip corner (outer)
            right_corner_mp = 291  # Right lip corner (outer)
            nose_tip_mp = 4  # Nose tip (available in 'mouth', not in 'lips')
            
            left_corner_idx = node_mapping.get(left_corner_mp)
            right_corner_idx = node_mapping.get(right_corner_mp)
            nose_tip_idx = node_mapping.get(nose_tip_mp)  # May be None for 'lips'
            
            if left_corner_idx is not None and right_corner_idx is not None:
                # Get corner positions in 3D
                left_corner = corrected[:, left_corner_idx, :]  # (frames, 3)
                right_corner = corrected[:, right_corner_idx, :]  # (frames, 3)
                corner_vector = right_corner - left_corner  # (frames, 3)
                
                # Step 1: Rotate to make lip corners horizontal (Y=0)
                # Target: corner vector should be horizontal (pointing right, Y=0)
                angle = torch.atan2(corner_vector[:, 1], corner_vector[:, 0])  # (frames,)
                cos_angle = torch.cos(-angle).unsqueeze(1)  # (frames, 1)
                sin_angle = torch.sin(-angle).unsqueeze(1)  # (frames, 1)
                
                # Apply 3D rotation matrix around Z-axis (yaw correction)
                x_rot = corrected[:, :, 0] * cos_angle - corrected[:, :, 1] * sin_angle
                y_rot = corrected[:, :, 0] * sin_angle + corrected[:, :, 1] * cos_angle
                z_rot = corrected[:, :, 2]  # Z unchanged for yaw rotation
                
                corrected = torch.stack([x_rot, y_rot, z_rot], dim=2)  # (frames, n_nodes, 3)
                
                # Step 2: Ensure all faces face RIGHT (right corner on positive X)
                # Check direction: right corner should be to the right of left corner
                right_corner_rotated = corrected[:, right_corner_idx, 0]  # (frames,)
                left_corner_rotated = corrected[:, left_corner_idx, 0]  # (frames,)
                corner_direction = right_corner_rotated - left_corner_rotated  # (frames,)
                facing_right = corner_direction > 0  # (frames,)
                
                # If NOT facing right, flip horizontally
                flip_mask = ~facing_right  # (frames,)
                if flip_mask.any():
                    corrected[flip_mask, :, 0] = -corrected[flip_mask, :, 0]  # Flip X
                    # Also flip Z to maintain 3D consistency
                    corrected[flip_mask, :, 2] = -corrected[flip_mask, :, 2]  # Flip Z
        
        return corrected
    
    def _normalize_scale(self, landmarks: torch.Tensor, partition: str, node_mapping: Dict) -> torch.Tensor:
        """
        Step 3: Scale Invariance - Normalize by IOD (full) or mouth width (mouth/lips).
        
        After pose invariance and rotation, normalize scale so face size doesn't matter.
        Uses 3D Euclidean distances for accurate measurements regardless of viewing angle.
        Formula: Divide all coordinates by IOD (inter-ocular distance) or mouth width.
        
        Args:
            landmarks: Shape (frames, n_nodes, 3) - Already pose-invariant and rotated
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Node mapping dict
            
        Returns:
            Scale-normalized coordinates (frames, n_nodes, 3) - X, Y, Z all normalized
        """
        frames, n_nodes, _ = landmarks.shape
        normalized = landmarks.clone()  # Keep 3D coordinates
        
        if partition == 'full':
            # Use Inter-Ocular Distance (IOD) normalization (3D distance)
            # After pose invariance, landmarks are centered at nose tip (origin)
            # We scale by IOD while keeping origin at nose tip
            left_eye_mp = 33  # Left eye center
            right_eye_mp = 263  # Right eye center
            
            left_eye_idx = node_mapping.get(left_eye_mp)
            right_eye_idx = node_mapping.get(right_eye_mp)
            
            if left_eye_idx is not None and right_eye_idx is not None:
                # Compute IOD using 3D Euclidean distance (accounts for depth)
                left_eye = normalized[:, left_eye_idx, :]  # (frames, 3)
                right_eye = normalized[:, right_eye_idx, :]  # (frames, 3)
                iod = torch.norm(right_eye - left_eye, dim=1, keepdim=True)  # (frames, 1) - 3D distance
                
                # Scale all coordinates (X, Y, Z) by IOD (keep origin at nose tip)
                normalized = normalized / (iod.unsqueeze(1) + 1e-6)  # (frames, n_nodes, 3)
            else:
                # Fallback to face bbox if eyes not available (2D only)
                normalized_2d = self._normalize_by_bbox(normalized[:, :, :2])
                normalized = torch.cat([normalized_2d, normalized[:, :, 2:]], dim=2)  # Keep Z
        
        else:  # 'mouth' or 'lips'
            # Use mouth width normalization (3D distance)
            # After pose invariance, landmarks are centered at nose tip (origin)
            # We scale by mouth width while keeping origin at nose tip
            left_corner_mp = 61  # Left lip corner
            right_corner_mp = 291  # Right lip corner
            
            left_corner_idx = node_mapping.get(left_corner_mp)
            right_corner_idx = node_mapping.get(right_corner_mp)
            
            if left_corner_idx is not None and right_corner_idx is not None:
                # Compute mouth width using 3D Euclidean distance (accounts for depth)
                # This gives true width regardless of viewing angle
                left_corner = normalized[:, left_corner_idx, :]  # (frames, 3)
                right_corner = normalized[:, right_corner_idx, :]  # (frames, 3)
                mouth_width = torch.norm(right_corner - left_corner, dim=1, keepdim=True)  # (frames, 1) - 3D distance
                
                # Scale all coordinates (X, Y, Z) by mouth width (keep origin at nose tip)
                normalized = normalized / (mouth_width.unsqueeze(1) + 1e-6)  # (frames, n_nodes, 3)
            else:
                # Fallback to face bbox if corners not available (2D only)
                normalized_2d = self._normalize_by_bbox(normalized[:, :, :2])
                normalized = torch.cat([normalized_2d, normalized[:, :, 2:]], dim=2)  # Keep Z
        
        return normalized
    
    def _normalize_by_bbox(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Fallback normalization using face bounding box (original method).
        
        Args:
            coords: Shape (frames, n_nodes, 2)
            
        Returns:
            Normalized coordinates
        """
        frames, n_nodes, _ = coords.shape
        features = coords.clone()
        
        # Vectorized per-frame normalization
        x_coords = features[:, :, 0]  # (frames, n_nodes)
        y_coords = features[:, :, 1]  # (frames, n_nodes)
        
        # Create valid mask
        valid_mask = (x_coords != 0) | (y_coords != 0)  # (frames, n_nodes)
        has_valid = valid_mask.any(dim=1)  # (frames,)
        
        # Compute bounding box
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
        
        # Face centers and sizes
        x_center = (x_min + x_max) / 2  # (frames, 1)
        y_center = (y_min + y_max) / 2  # (frames, 1)
        face_width = x_max - x_min  # (frames, 1)
        face_height = y_max - y_min  # (frames, 1)
        face_size = torch.max(face_width, face_height)  # (frames, 1)
        
        # Avoid division by zero
        face_size = torch.clamp(face_size, min=1e-6)
        
        # Center and scale
        features[:, :, 0] = (x_coords - x_center) / face_size
        features[:, :, 1] = (y_coords - y_center) / face_size
        
        # Zero out frames with no valid points
        features[~has_valid] = 0
        
        return features
    
    def compute_velocity(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity (first derivative) in 3D.
        OPTIMIZED: Vectorized computation for better performance.
        
        Args:
            coords: Shape (frames, n_nodes, 2 or 3) - normalized coordinates
            
        Returns:
            Velocity of shape (frames, n_nodes, 2 or 3) - same dimensionality as input
            Note: Returns 3D velocity (vx, vy, vz) if input is 3D, 2D (vx, vy) if input is 2D
        """
        frames = coords.shape[0]
        velocity = torch.zeros_like(coords)
        
        if frames > 1:
            # OPTIMIZATION: Vectorized computation
            if frames > 2:
                # Central difference for all frames (more accurate)
                # First frame: forward difference
                velocity[0] = coords[1] - coords[0]
                # Middle frames: central difference (vectorized)
                velocity[1:-1] = (coords[2:] - coords[:-2]) * 0.5  # Use multiplication instead of division
                # Last frame: backward difference
                velocity[-1] = coords[-1] - coords[-2]
            else:
                # Only 2 frames: simple forward difference
                velocity[0] = coords[1] - coords[0]
                velocity[1] = coords[1] - coords[0]
        
        return velocity
    
    def compute_acceleration(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration (second derivative) in 3D.
        
        Args:
            velocity: Shape (frames, n_nodes, 2 or 3) - velocity vectors
            
        Returns:
            Acceleration of shape (frames, n_nodes, 2 or 3) - same dimensionality as input
            Note: Returns 3D acceleration (ax, ay, az) if input is 3D, 2D (ax, ay) if input is 2D
        """
        return self.compute_velocity(velocity)
    
    def compute_speed(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute speed magnitude from velocity (works for both 2D and 3D).
        
        Args:
            velocity: Shape (frames, n_nodes, 2 or 3) - velocity vectors
            
        Returns:
            Speed of shape (frames, n_nodes, 1) - 3D magnitude if input is 3D, 2D magnitude if input is 2D
            Note: For 3D: speed = sqrt(vx^2 + vy^2 + vz^2)
                  For 2D: speed = sqrt(vx^2 + vy^2)
        """
        # Speed = ||velocity|| (works for both 2D and 3D)
        speed = torch.norm(velocity, dim=-1, keepdim=True)
        return speed
    
    def compute_B1(self, landmarks: torch.Tensor, meta: Dict, partition: str = 'mouth', node_mapping: Optional[Dict] = None) -> torch.Tensor:
        """
        B1: Returns B0 + velocity + speed + acceleration (cumulative, includes B0).
        OPTIMIZED: Reduced tensor concatenations and vectorized operations.
        
        Args:
            landmarks: Shape (frames, n_nodes, 2 or 3) - raw landmarks
            meta: Video metadata
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Optional node mapping dict
            
        Returns:
            Features of shape (frames, n_nodes, 10)  [B0: x, y, z] + [B1: vx, vy, vz, speed, ax, ay, az]
            Note: Velocity and acceleration computed in 3D, speed is 3D magnitude
        """
        # B0 features (now returns X, Y, Z - 3D)
        b0 = self.compute_B0(landmarks, meta, partition=partition, node_mapping=node_mapping)  # (frames, n_nodes, 3)
        
        # Velocity in 3D
        velocity = self.compute_velocity(b0)  # (frames, n_nodes, 3) - vx, vy, vz
        
        # OPTIMIZATION: Compute speed and acceleration in one pass, then concatenate once
        # Speed magnitude (3D): sqrt(vx^2 + vy^2 + vz^2)
        speed = torch.norm(velocity, dim=-1, keepdim=True)  # (frames, n_nodes, 1) - inline computation
        
        # Acceleration in 3D (reuse velocity computation)
        acceleration = self.compute_velocity(velocity)  # (frames, n_nodes, 3) - ax, ay, az
        
        # OPTIMIZATION: Single concatenation instead of two (reduces memory allocation)
        # Return B0 + velocity + speed + acceleration (cumulative: B0 + B1)
        features = torch.cat([b0, velocity, speed, acceleration], dim=-1)  # (frames, n_nodes, 10)
        
        return features
    
    def compute_B2(self, landmarks: torch.Tensor, meta: Dict, partition: str, b0_coords: Optional[torch.Tensor] = None, b1_features: Optional[torch.Tensor] = None, node_indices: Optional[Dict[str, int]] = None) -> torch.Tensor:
        """
        B2: Returns B0 + B1 + global geometric features (cumulative, includes B0+B1).
        Global features are broadcast to all nodes per frame.
        Features: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening (8 features).
        OPTIMIZED: Accepts precomputed node indices to avoid repeated lookups.
        
        Args:
            landmarks: Shape (frames, n_nodes, 2 or 3) - raw landmarks (used if b0_coords not provided)
            meta: Video metadata
            partition: 'lips', 'mouth', or 'full'
            b0_coords: Optional pre-computed B0 coordinates (faster, avoids recomputation) - now 3D (X, Y, Z)
            b1_features: Optional pre-computed B1 features (B0+B1, faster, avoids recomputation) - now B0(3) + B1(7) = 10
            node_indices: Optional precomputed node indices dict (faster, avoids repeated lookups)
            
        Returns:
            Features of shape (frames, n_nodes, 18)  [B0: x, y, z] + [B1: vx, vy, vz, speed, ax, ay, az] + [B2: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening]
            Note: B2 geometric features use 3D Euclidean distances for accurate measurements
        """
        # OPTIMIZATION: Use existing B0+B1 if provided (much faster)
        if b1_features is not None:
            b0_b1 = b1_features  # Already has B0+B1 (10 features: B0(3) + B1(7))
            b0 = b1_features[:, :, :3]  # Extract B0 (X, Y, Z) for geometric computation
        else:
            # Compute B0
            if b0_coords is not None:
                b0 = b0_coords
            else:
                # Get node_mapping for B0 if not provided
                from preprocessing.mediapipe_nodes import get_partition_nodes
                nodes = get_partition_nodes(partition)
                node_mapping_b0 = {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
                b0 = self.compute_B0(landmarks, meta, partition=partition, node_mapping=node_mapping_b0)
            
            # Compute B1 (B0 + velocity + speed + acceleration)
            from preprocessing.mediapipe_nodes import get_partition_nodes
            nodes = get_partition_nodes(partition)
            node_mapping_b1 = {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
            b1 = self.compute_B1(landmarks, meta, partition=partition, node_mapping=node_mapping_b1)  # Returns B0+B1
            b0_b1 = b1
        
        # Global geometric features (B2 incremental) - broadcast to all nodes
        # Features: MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening (8 features)
        # OPTIMIZATION: Pass precomputed node_indices to avoid repeated lookups
        geom_global = self.compute_global_geometric_features(b0, partition, node_indices=node_indices)
        
        # Broadcast global features to all nodes: (frames, 8) -> (frames, n_nodes, 8)
        frames, n_nodes, _ = b0_b1.shape
        geom_broadcast = geom_global.unsqueeze(1).expand(frames, n_nodes, -1)  # (frames, n_nodes, 8)
        
        # Return B0 + B1 + B2 (cumulative)
        features = torch.cat([b0_b1, geom_broadcast], dim=-1)  # (frames, n_nodes, 18) - B0(3) + B1(7) + B2(8)
        
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
            coords: Shape (frames, n_nodes, 2 or 3) - normalized coordinates (2D or 3D)
            partition: 'lips', 'mouth', or 'full'
            node_indices: Optional precomputed dict with keys: 'left_corner', 'right_corner', 'chin_center', 'lips_nodes', 'left_cheek', 'right_cheek'
                         If None, will compute on-the-fly (slower)
            
        Returns:
            Global features of shape (frames, 8) - [MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening]
            Note: Width and distances use 3D Euclidean distance if Z is available, otherwise 2D
        """
        frames = coords.shape[0]
        n_dims = coords.shape[2]
        device = coords.device
        dtype = coords.dtype
        use_3d = (n_dims == 3)
        
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
        
        # OPTIMIZATION: Pre-compute all needed tensor slices and bounds once
        # Cache lip coordinates (used multiple times)
        lip_coords = None
        lip_y = None
        lip_y_min = None
        lip_y_max = None
        lower_lip_y = None
        bottom_y = None
        
        if lips_nodes_remapped:
            lip_coords = coords[:, lips_nodes_remapped, :]  # (frames, n_lip_nodes, 2 or 3) - compute once
            lip_y = lip_coords[:, :, 1]  # (frames, n_lip_nodes) - Y coordinate, compute once
            
            # Compute lip Y bounds once (used for MAR, lip_height, jaw_height, jaw_opening)
            lip_y_min = lip_y.min(dim=1)[0]  # (frames,) - upper lip
            lip_y_max = lip_y.max(dim=1)[0]  # (frames,) - lower lip
            lower_lip_y = lip_y_max  # Same as lip_y_max, reuse
            lip_height = lip_y_max - lip_y_min  # (frames,)
        else:
            # Fallback: compute bottom_y once if needed
            bottom_y = coords[:, :, 1].max(dim=1)[0]  # (frames,)
            lip_height = torch.zeros(frames, device=device, dtype=dtype)
            
        # Pre-compute corner coordinates once (used for lip_width and corner_angle)
        left_corner_coords = None
        right_corner_coords = None
        if left_corner_remapped is not None and right_corner_remapped is not None:
            left_corner_coords = coords[:, left_corner_remapped, :]  # (frames, 2 or 3) - compute once
            right_corner_coords = coords[:, right_corner_remapped, :]  # (frames, 2 or 3) - compute once
        
        # Pre-compute chin coordinates once (used for jaw_height and jaw_opening)
        chin_coords = None
        chin_y = None
        if chin_center_remapped is not None:
            chin_coords = coords[:, chin_center_remapped, :]  # (frames, 2 or 3) - compute once
            chin_y = chin_coords[:, 1]  # (frames,) - Y coordinate
        
        # Pre-compute face center once (used for cheek_puff)
        face_center = None
        if left_cheek_remapped and right_cheek_remapped:
            face_center = coords.mean(dim=1)  # (frames, 2 or 3) - compute once
        
        # 1. MAR (Mouth Aspect Ratio) = lip_height / lip_width
        # 2. Lip Width = 3D Euclidean distance between left and right corners
        if left_corner_coords is not None and right_corner_coords is not None:
            # Lip width: Use 3D Euclidean distance if available, otherwise 2D
            corner_diff = right_corner_coords - left_corner_coords  # (frames, 2 or 3)
            lip_width = torch.norm(corner_diff, dim=1)  # (frames,) - 3D distance if Z available
            mar = lip_height / (lip_width + 1e-6)  # (frames,)
            global_features[:, 0] = mar
            global_features[:, 1] = lip_width
        elif lip_coords is not None:
            # Fallback: use lip bounding box (compute X bounds once)
            lip_x = lip_coords[:, :, 0]  # (frames, n_lip_nodes)
            lip_x_min = lip_x.min(dim=1)[0]  # (frames,)
            lip_x_max = lip_x.max(dim=1)[0]  # (frames,)
            lip_width = lip_x_max - lip_x_min
            mar = lip_height / (lip_width + 1e-6)
            global_features[:, 0] = mar
            global_features[:, 1] = lip_width
            
            # 3. Lip Height (already computed)
        if lip_height is not None:
            global_features[:, 2] = lip_height
            
            # 4. Jaw Height: distance from chin center to upper lip
        if chin_y is not None and lip_y_min is not None:
                jaw_height = chin_y - lip_y_min  # (frames,) - vertical distance
                global_features[:, 3] = jaw_height
        elif bottom_y is not None and lip_y_max is not None:
                jaw_height = bottom_y - lip_y_max  # Distance from bottom to lower lip
                global_features[:, 3] = jaw_height
        
        # 5. Cheek Puff: average 3D distance of cheek nodes from face center
        if face_center is not None and left_cheek_remapped and right_cheek_remapped:
            # Left and right cheek centers (compute once)
            left_cheek_coords = coords[:, left_cheek_remapped, :]  # (frames, n_left_cheek, 2 or 3)
            left_cheek_center = left_cheek_coords.mean(dim=1)  # (frames, 2 or 3)
            
            right_cheek_coords = coords[:, right_cheek_remapped, :]  # (frames, n_right_cheek, 2 or 3)
            right_cheek_center = right_cheek_coords.mean(dim=1)  # (frames, 2 or 3)
            
            # Distance from face center (3D Euclidean distance if Z available)
            left_dist = torch.norm(left_cheek_center - face_center, dim=1)  # (frames,)
            right_dist = torch.norm(right_cheek_center - face_center, dim=1)  # (frames,)
            
            # Average cheek puff
            cheek_puff = (left_dist + right_dist) / 2.0  # (frames,)
            global_features[:, 4] = cheek_puff
        
        # 6. Lip Curvature: approximate curvature from upper and lower lip
        # OPTIMIZATION: Reuse precomputed lip_y, lip_y_min, lip_y_max
        if lip_y is not None and len(lips_nodes_remapped) >= 3:
            # Compute variance efficiently (reuse lip_y_mean for both variance and range)
            lip_y_mean = lip_y.mean(dim=1, keepdim=True)  # (frames, 1)
            lip_y_var = ((lip_y - lip_y_mean) ** 2).mean(dim=1)  # (frames,)
            
            # Reuse precomputed lip_y_max and lip_y_min for range
            lip_y_range = lip_y_max - lip_y_min  # (frames,) - already computed above
            
            # Combine variance and range for better curvature measure
            lip_curvature = lip_y_var * lip_y_range  # (frames,)
            global_features[:, 5] = lip_curvature
        
        # 7. Lip Corner Angle: angle between left and right corners (relative to horizontal)
        # OPTIMIZATION: Reuse precomputed corner coordinates
        if left_corner_coords is not None and right_corner_coords is not None:
            # Vector from left to right corner (use X, Y only for angle calculation)
            corner_vector = right_corner_coords[:, :2] - left_corner_coords[:, :2]  # (frames, 2)
            
            # Angle relative to horizontal (atan2 of y/x)
            corner_angle = torch.atan2(corner_vector[:, 1], corner_vector[:, 0])  # (frames,)
            # Convert to degrees
            corner_angle_deg = corner_angle * 180.0 / math.pi  # (frames,)
            global_features[:, 6] = corner_angle_deg
        
        # 8. Jaw Opening: distance from chin center to lower lip (vertical, Y-axis only)
        # OPTIMIZATION: Reuse precomputed chin_y and lower_lip_y
        if chin_y is not None and lower_lip_y is not None:
            jaw_opening = lower_lip_y - chin_y  # (frames,)
            global_features[:, 7] = jaw_opening
        elif bottom_y is not None and lower_lip_y is not None:
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
        Uses 3D Euclidean distances if Z coordinate is available.
        Vectorized for performance.
        
        Args:
            coords: Shape (frames, n_nodes, 2 or 3) - normalized coordinates
            partition: 'lips', 'mouth', or 'full'
            anchor_idx: Pre-computed anchor index (optional, for optimization)
            
        Returns:
            Geometric features of shape (frames, n_nodes, 2) - [distance_to_anchor, angle]
            Note: Distance uses 3D Euclidean distance if Z is available, otherwise 2D
        """
        frames, n_nodes, n_dims = coords.shape
        device = coords.device
        dtype = coords.dtype
        use_3d = (n_dims == 3)
        
        # 1. Pairwise distances to anchor nodes (vectorized) - 3D if available
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
        anchor_coords = coords[:, anchor_remapped_idx:anchor_remapped_idx+1, :]  # (frames, 1, 2 or 3)
        
        # Compute distances more efficiently: (frames, n_nodes)
        # Use broadcasting: coords (frames, n_nodes, 2 or 3) - anchor_coords (frames, 1, 2 or 3)
        diff = coords - anchor_coords  # (frames, n_nodes, 2 or 3)
        distances = torch.norm(diff, dim=2, keepdim=True)  # (frames, n_nodes, 1) - 3D distance if Z available
        
        # 2. Angles between consecutive nodes (vectorized) - uses 3D if available
        # Compute vectors: prev->current and current->next
        # For each node i: vec1 = coords[i] - coords[i-1], vec2 = coords[i+1] - coords[i]
        # Edge cases: first node uses coords[0] as vec1, last node uses coords[-1] as vec2
        
        # Shift coordinates for vector computation
        coords_prev = torch.cat([coords[:, 0:1, :], coords[:, :-1, :]], dim=1)  # Shift right (prev node)
        coords_next = torch.cat([coords[:, 1:, :], coords[:, -1:, :]], dim=1)   # Shift left (next node)
        
        vec1 = coords - coords_prev  # (frames, n_nodes, 2 or 3)
        vec2 = coords_next - coords  # (frames, n_nodes, 2 or 3)
        
        # Compute dot products and norms (vectorized) - works for both 2D and 3D
        dot_products = torch.sum(vec1 * vec2, dim=2)  # (frames, n_nodes)
        norm1 = torch.norm(vec1, dim=2)  # (frames, n_nodes) - 3D norm if Z available
        norm2 = torch.norm(vec2, dim=2)  # (frames, n_nodes) - 3D norm if Z available
        
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
            landmarks: Shape (frames, n_nodes, 2 or 3) - raw landmarks
            meta: Video metadata
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Original MediaPipe index -> new index
            b0_coords: Optional pre-computed B0 coordinates (faster) - now 3D (X, Y, Z)
            b2_features: Optional pre-computed B2 features (B0+B1+B2, faster, avoids recomputation) - now 18 features
            
        Returns:
            Tuple of (features, additional_meta) - B0+B1+B2+B3 (22 features total)
            B0(3) + B1(7) + B2(8) + B3(4) = 22 features
        """
        # OPTIMIZATION: Use existing B0+B1+B2 if provided (much faster)
        if b2_features is not None:
            b0_b1_b2 = b2_features  # Already has B0+B1+B2 (18 features: B0(3) + B1(7) + B2(8))
            b0 = b2_features[:, :, :3]  # Extract B0 (X, Y, Z) for AU computation
        else:
            # Compute B0
            if b0_coords is not None:
                b0 = b0_coords
            else:
                # Get node_mapping for B0 if not provided
                if node_mapping is None:
                    from preprocessing.mediapipe_nodes import get_partition_nodes
                    nodes = get_partition_nodes(partition)
                    node_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
                b0 = self.compute_B0(landmarks, meta, partition=partition, node_mapping=node_mapping)
            
            # Compute B2 (B0+B1+B2)
            b2 = self.compute_B2(landmarks, meta, partition, b0_coords=b0)  # Returns B0+B1+B2
            b0_b1_b2 = b2
        
        # AU features only (B3 incremental)
        au_features, au_groups = self.compute_AU_features(b0, partition, node_mapping)
        
        # Return B0 + B1 + B2 + B3 (cumulative)
        features = torch.cat([b0_b1_b2, au_features], dim=-1)  # (frames, n_nodes, 22) - B0(3) + B1(7) + B2(8) + B3(4)
        
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
        Compute Action Unit inspired features using 3D displacement magnitudes.
        
        Args:
            coords: Shape (frames, n_nodes, 2 or 3) - normalized coordinates
            partition: 'lips', 'mouth', or 'full'
            node_mapping: Original MediaPipe index -> new index
            
        Returns:
            Tuple of (AU features, AU node mapping)
            Note: Displacement magnitudes use 3D Euclidean distance if Z is available
        """
        frames, n_nodes, n_dims = coords.shape
        use_3d = (n_dims == 3)
        
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
            
            # Extract group coordinates for all frames: (frames, n_group_nodes, 2 or 3)
            group_coords = coords[:, node_indices, :]
            
            # Compute displacement magnitude for all frames using 3D Euclidean distance if available
            # This gives true displacement regardless of viewing angle
            group_displacement = torch.norm(group_coords, dim=2)  # (frames, n_group_nodes) - 3D if Z available
            
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
            features = self.compute_B0(landmarks, meta, partition=partition, node_mapping=node_mapping)
            additional_meta = {}
        
        elif self.feature_level == 'B1':
            features = self.compute_B1(landmarks, meta, partition=partition, node_mapping=node_mapping)
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
    
    Storage format (cumulative, 3D coordinates):
    - B0: B0 features only (3 features: X, Y, Z - 3D normalized coordinates)
    - B1: B0 + B1 features (10 features: B0: 3 + B1: 7 = vx, vy, vz, speed, ax, ay, az)
    - B2: B0 + B1 + B2 features (18 features: B0: 3 + B1: 7 + B2: 8 = MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening)
    - B3: B0 + B1 + B2 + B3 features (22 features: B0: 3 + B1: 7 + B2: 8 + B3: 4 = AU25, AU26, AU12, AU27)
    
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
    
    # MEMORY OPTIMIZATION: Don't load previous level features to avoid memory issues
    # Loading .pt files loads everything into memory anyway, so we skip it and compute from landmarks
    # This uses less memory (no duplication) but is slightly slower
    logger.info(f"Computing {feature_level} features directly from landmarks (memory-efficient mode)")
    if feature_level == 'B2':
        logger.info(f"B2 will compute B0+B1+B2 from landmarks in one pass")
    
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
    
    # MEMORY OPTIMIZATION: Process videos sequentially one at a time
    # This prevents swap memory usage when processing large datasets
    logger.info(f"Processing {total_videos} videos sequentially...")
    
    # OPTIMIZATION: Use sequential processing with optimized code (avoids memory duplication from multiprocessing)
    # ProcessPoolExecutor duplicates all data for each process, causing massive memory usage
    # Sequential processing is slower but uses much less memory
    processed_count = 0
    fe = FeatureEngineer(feature_level=feature_level)
    progress_bar = tqdm(videos.items(), desc=f"Computing {feature_level} features", unit="video")
    
    # Process videos one at a time
    for video_id, video_data in progress_bar:
        landmarks = video_data['landmarks']
        meta = video_data['meta']
        
        # Compute features directly from landmarks (memory-efficient, no previous level loading)
        if feature_level == 'B2':
            features = fe.compute_B2(landmarks, meta, partition, node_indices=b2_node_indices)
            additional_meta = {}
        else:
            features, additional_meta = fe.compute_features(landmarks, meta, partition=partition, node_mapping=node_mapping)
        
        # MEMORY OPTIMIZATION: Convert to float16 immediately to reduce memory by 50%
        # This is safe because we'll convert back to float32 during training if needed
        if features.dtype != torch.float16:
            features = features.half()  # Convert to float16 immediately
        
        # Store processed video
        feature_data['videos'][video_id] = {
            'video_path': video_data['video_path'],
            'word': video_data['word'],
            'label': video_data['label'],
            'features': features,
            'speech_mask': video_data['speech_mask'],
            'meta': meta,
        }
        
        # Update global meta from first video
        if processed_count == 0:
            feature_data['meta']['n_features'] = features.shape[-1]
            feature_data['meta']['n_nodes'] = features.shape[1]
            if additional_meta:
                feature_data['meta'].update(additional_meta)
        
        processed_count += 1
        
        # MEMORY OPTIMIZATION: Clear original landmarks from videos dict to free memory
        # The landmarks are no longer needed after features are computed
        del landmarks
        video_data['landmarks'] = None  # Mark as cleared
        # Note: 'features' is stored in feature_data['videos'], so we can't delete it yet
        # But converting to float16 reduces memory by 50%
        
        if processed_count % 100 == 0:
            import gc
            gc.collect()  # Force garbage collection periodically
            # Clear Python's internal caches
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Log memory usage periodically
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / (1024 * 1024)
                logger.info(f"Memory usage after {processed_count} videos: {mem_mb:.1f} MB")
            except ImportError:
                pass  # psutil not available
    
    progress_bar.close()
    
    
    # Save final result
    # Note: Features are already in float16 format (converted during processing)
    # This reduces memory usage by 50% compared to float32
    # Converted to float32 during __getitem__ for training accuracy if needed
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

