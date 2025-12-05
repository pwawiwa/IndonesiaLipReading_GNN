"""
Face-aware augmentations for lip reading GNN models.

All augmentations:
1. Keep features and speech_mask synchronized
2. Respect facial geometry and anatomical constraints
3. Are memory-efficient (operate in-place when possible)
4. Handle float16/float32 properly
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import random


class FaceAwareAugmentation:
    """
    Base class for face-aware augmentations.
    All augmentations must return (features, speech_mask) tuple.
    """
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying augmentation
        """
        self.p = p
    
    def __call__(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation.
        
        Args:
            features: (frames, nodes, features) tensor
            speech_mask: (frames,) binary mask
            
        Returns:
            (augmented_features, augmented_speech_mask)
        """
        if random.random() < self.p:
            return self.apply(features, speech_mask)
        return features, speech_mask
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subclasses implement this."""
        raise NotImplementedError


class NodeFeatureDropout(FaceAwareAugmentation):
    """
    Dropout on node features (spatial dropout).
    Randomly zero out some node features per frame.
    Memory-efficient: operates in-place when possible.
    """
    def __init__(self, p: float = 0.5, dropout_rate: float = 0.1):
        """
        Args:
            p: Probability of applying augmentation
            dropout_rate: Fraction of features to zero out (per frame, per node)
        """
        super().__init__(p)
        self.dropout_rate = dropout_rate
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to float32 if needed (for proper dropout)
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Create dropout mask: (frames, nodes, features)
        dropout_mask = torch.rand_like(features) > self.dropout_rate
        
        # Apply dropout
        features = features * dropout_mask
        
        # Convert back to float16 if original was float16
        if was_float16:
            features = features.half()
        
        # Speech mask unchanged (spatial augmentation only)
        return features, speech_mask


class TemporalJitter(FaceAwareAugmentation):
    """
    Small temporal jitter: randomly skip or duplicate a few frames.
    Keeps speech_mask synchronized with features.
    """
    def __init__(self, p: float = 0.5, max_skip: int = 2, max_duplicate: int = 1):
        """
        Args:
            p: Probability of applying augmentation
            max_skip: Maximum number of frames to skip
            max_duplicate: Maximum number of frames to duplicate
        """
        super().__init__(p)
        self.max_skip = max_skip
        self.max_duplicate = max_duplicate
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames = features.shape[0]
        
        # Skip frames: randomly remove 0 to max_skip frames
        n_skip = random.randint(0, min(self.max_skip, num_frames // 4))  # Don't skip more than 25%
        if n_skip > 0:
            skip_indices = sorted(random.sample(range(num_frames), n_skip))
            keep_indices = [i for i in range(num_frames) if i not in skip_indices]
            features = features[keep_indices]
            speech_mask = speech_mask[keep_indices]
            num_frames = features.shape[0]
        
        # Duplicate frames: randomly duplicate 0 to max_duplicate frames
        n_duplicate = random.randint(0, min(self.max_duplicate, num_frames // 10))  # Don't duplicate more than 10%
        if n_duplicate > 0:
            duplicate_indices = sorted(random.sample(range(num_frames), n_duplicate))
            features_list = [features]
            mask_list = [speech_mask]
            
            for idx in duplicate_indices:
                features_list.append(features[idx:idx+1])
                mask_list.append(speech_mask[idx:idx+1])
            
            # Interleave: insert duplicates after original frames
            new_features = []
            new_mask = []
            insert_idx = 0
            for i in range(num_frames):
                new_features.append(features[i:i+1])
                new_mask.append(speech_mask[i:i+1])
                if insert_idx < len(duplicate_indices) and i == duplicate_indices[insert_idx]:
                    # Insert duplicate after this frame
                    dup_idx = duplicate_indices.index(i)
                    new_features.append(features_list[dup_idx + 1])
                    new_mask.append(mask_list[dup_idx + 1])
                    insert_idx += 1
            
            features = torch.cat(new_features, dim=0)
            speech_mask = torch.cat(new_mask, dim=0)
        
        return features, speech_mask


class GaussianNoise(FaceAwareAugmentation):
    """
    Add small Gaussian noise to features.
    Respects feature scales (noise proportional to feature magnitude).
    """
    def __init__(self, p: float = 0.5, noise_std: float = 0.01):
        """
        Args:
            p: Probability of applying augmentation
            noise_std: Standard deviation of noise (relative to feature scale)
        """
        super().__init__(p)
        self.noise_std = noise_std
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to float32 if needed
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Compute feature scale (per feature dimension)
        feature_std = features.std(dim=(0, 1), keepdim=True)  # (1, 1, features)
        feature_std = torch.clamp(feature_std, min=1e-6)  # Avoid division by zero
        
        # Generate noise proportional to feature scale
        noise = torch.randn_like(features) * self.noise_std * feature_std
        
        # Add noise
        features = features + noise
        
        # Convert back to float16 if original was float16
        if was_float16:
            features = features.half()
        
        # Speech mask unchanged
        return features, speech_mask


class FeatureScaling(FaceAwareAugmentation):
    """
    Small random scaling of features (per video).
    Preserves relative relationships between features.
    """
    def __init__(self, p: float = 0.5, scale_range: Tuple[float, float] = (0.95, 1.05)):
        """
        Args:
            p: Probability of applying augmentation
            scale_range: (min_scale, max_scale) for random scaling
        """
        super().__init__(p)
        self.scale_range = scale_range
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random scale factor for entire video
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Apply scaling
        features = features * scale
        
        # Speech mask unchanged
        return features, speech_mask


class HorizontalFlip(FaceAwareAugmentation):
    """
    Horizontal flip (mirror) augmentation.
    CRITICAL: Only works if features contain B0 (x, y coordinates) as first 2 features.
    Flips x-coordinates and adjusts for face orientation.
    """
    def __init__(self, p: float = 0.5, feature_level: str = 'B0'):
        """
        Args:
            p: Probability of applying augmentation
            feature_level: Feature level (B0, B1, B2, B3) - determines which features to flip
        """
        super().__init__(p)
        self.feature_level = feature_level
        # B0 has x,y at indices 0,1
        # B1 adds vx,vy at indices 2,3 (need to flip vx)
        # B2, B3 don't have direct spatial coordinates to flip
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only flip if we have B0 features (x, y coordinates)
        if features.shape[2] < 2:
            return features, speech_mask
        
        # Convert to float32 if needed
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Flip x-coordinates (index 0)
        # Assuming normalized coordinates [0, 1], flip: x_new = 1 - x_old
        features[:, :, 0] = 1.0 - features[:, :, 0]
        
        # If B1 features exist (vx, vy at indices 2, 3), flip vx (velocity in x-direction)
        if features.shape[2] >= 4:
            features[:, :, 2] = -features[:, :, 2]  # Flip x-velocity
        
        # Convert back to float16 if original was float16
        if was_float16:
            features = features.half()
        
        # Speech mask unchanged (spatial augmentation only)
        return features, speech_mask


class FaceRotation(FaceAwareAugmentation):
    """
    Face rotation augmentation (2D rotation around face center).
    Rotates landmarks around the face center point.
    Handles B0 (x,y), B1 (vx,vy), and updates velocities correctly.
    """
    def __init__(self, p: float = 0.5, max_angle_deg: float = 15.0):
        """
        Args:
            p: Probability of applying augmentation
            max_angle_deg: Maximum rotation angle in degrees (±max_angle_deg)
        """
        super().__init__(p)
        self.max_angle_deg = max_angle_deg
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only works if we have B0 features (x, y coordinates)
        if features.shape[2] < 2:
            return features, speech_mask
        
        # Convert to float32 if needed
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Random rotation angle
        angle_deg = random.uniform(-self.max_angle_deg, self.max_angle_deg)
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Compute face center (mean of all landmarks, all frames)
        x_coords = features[:, :, 0]  # (frames, nodes)
        y_coords = features[:, :, 1]  # (frames, nodes)
        center_x = x_coords.mean().item()
        center_y = y_coords.mean().item()
        
        # Rotate coordinates around center
        x_centered = x_coords - center_x
        y_centered = y_coords - center_y
        
        # Apply rotation matrix
        x_rotated = x_centered * cos_a - y_centered * sin_a
        y_rotated = x_centered * sin_a + y_centered * cos_a
        
        # Translate back
        features[:, :, 0] = x_rotated + center_x
        features[:, :, 1] = y_rotated + center_y
        
        # If B1 features exist (vx, vy at indices 2, 3), rotate velocities
        if features.shape[2] >= 4:
            vx = features[:, :, 2]  # (frames, nodes)
            vy = features[:, :, 3]  # (frames, nodes)
            
            # Rotate velocity vectors
            vx_rotated = vx * cos_a - vy * sin_a
            vy_rotated = vx * sin_a + vy * cos_a
            
            features[:, :, 2] = vx_rotated
            features[:, :, 3] = vy_rotated
        
        # Convert back to float16 if original was float16
        if was_float16:
            features = features.half()
        
        return features, speech_mask


class FaceTranslation(FaceAwareAugmentation):
    """
    Small random translation of face landmarks.
    Simulates slight head movement or camera shift.
    """
    def __init__(self, p: float = 0.5, max_translation: float = 0.02):
        """
        Args:
            p: Probability of applying augmentation
            max_translation: Maximum translation in normalized coordinates (0-1 range)
        """
        super().__init__(p)
        self.max_translation = max_translation
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.shape[2] < 2:
            return features, speech_mask
        
        # Convert to float32 if needed
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Random translation
        tx = random.uniform(-self.max_translation, self.max_translation)
        ty = random.uniform(-self.max_translation, self.max_translation)
        
        # Apply translation to coordinates
        features[:, :, 0] = features[:, :, 0] + tx
        features[:, :, 1] = features[:, :, 1] + ty
        
        # Clamp to [0, 1] range (normalized coordinates)
        features[:, :, 0] = torch.clamp(features[:, :, 0], 0.0, 1.0)
        features[:, :, 1] = torch.clamp(features[:, :, 1], 0.0, 1.0)
        
        # Convert back to float16 if original was float16
        if was_float16:
            features = features.half()
        
        return features, speech_mask


class FaceOrientationAwareAugmentation(FaceAwareAugmentation):
    """
    Augmentation that considers face orientation (left/right facing).
    Detects face orientation from landmark positions and applies orientation-aware augmentations.
    Enhanced version with better rotation handling.
    """
    def __init__(self, p: float = 0.5, max_rotation_deg: float = 10.0):
        """
        Args:
            p: Probability of applying augmentation
            max_rotation_deg: Maximum rotation angle for orientation correction
        """
        super().__init__(p)
        self.max_rotation_deg = max_rotation_deg
    
    def _detect_face_orientation(self, features: torch.Tensor) -> Tuple[str, float]:
        """
        Detect if face is facing left or right based on landmark positions.
        Uses first frame's x-coordinates (B0 feature index 0).
        
        Returns:
            ('left', 'right', or 'center', angle_estimate)
        """
        if features.shape[2] < 2:
            return 'center', 0.0
        
        # Use first frame's x-coordinates
        x_coords = features[0, :, 0]  # (nodes,)
        y_coords = features[0, :, 1]  # (nodes,)
        
        # Compute center of mass
        center_x = x_coords.mean().item()
        center_y = y_coords.mean().item()
        
        # Compute asymmetry: compare left and right sides
        # For mouth partition, we can use x-coordinate distribution
        x_std = x_coords.std().item()
        
        # Heuristic: if center < 0.45, face is on left side (facing right)
        # if center > 0.55, face is on right side (facing left)
        if center_x < 0.45:
            orientation = 'right'  # Face on left, facing right
            # Estimate rotation angle based on deviation
            angle = min(15.0, (0.5 - center_x) * 30.0)  # Scale to degrees
        elif center_x > 0.55:
            orientation = 'left'   # Face on right, facing left
            angle = min(15.0, (center_x - 0.5) * 30.0)
        else:
            orientation = 'center'
            angle = 0.0
        
        return orientation, angle
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detect orientation
        orientation, estimated_angle = self._detect_face_orientation(features)
        
        if orientation == 'center':
            return features, speech_mask
        
        # Convert to float32 if needed
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Apply small rotation to normalize orientation
        # Use smaller rotation than estimated to avoid over-correction
        correction_angle_deg = min(estimated_angle * 0.3, self.max_rotation_deg)
        correction_angle_rad = np.deg2rad(correction_angle_deg)
        cos_a = np.cos(correction_angle_rad)
        sin_a = np.sin(correction_angle_rad)
        
        # Rotate in opposite direction to normalize
        if orientation == 'left':
            sin_a = -sin_a  # Rotate right to normalize
        elif orientation == 'right':
            sin_a = sin_a   # Rotate left to normalize
        
        # Compute face center
        x_coords = features[:, :, 0]
        y_coords = features[:, :, 1]
        center_x = x_coords.mean().item()
        center_y = y_coords.mean().item()
        
        # Apply rotation
        x_centered = x_coords - center_x
        y_centered = y_coords - center_y
        
        x_rotated = x_centered * cos_a - y_centered * sin_a
        y_rotated = x_centered * sin_a + y_centered * cos_a
        
        features[:, :, 0] = x_rotated + center_x
        features[:, :, 1] = y_rotated + center_y
        
        # Rotate velocities if B1 features exist
        if features.shape[2] >= 4:
            vx = features[:, :, 2]
            vy = features[:, :, 3]
            vx_rotated = vx * cos_a - vy * sin_a
            vy_rotated = vx * sin_a + vy * cos_a
            features[:, :, 2] = vx_rotated
            features[:, :, 3] = vy_rotated
        
        # Convert back to float16 if original was float16
        if was_float16:
            features = features.half()
        
        return features, speech_mask


class CompositeAugmentation:
    """
    Composite augmentation that applies multiple augmentations sequentially.
    All augmentations keep features and speech_mask synchronized.
    """
    def __init__(self, augmentations: list):
        """
        Args:
            augmentations: List of FaceAwareAugmentation instances
        """
        self.augmentations = augmentations if augmentations else []
    
    def __call__(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all augmentations sequentially."""
        if not self.augmentations:
            return features, speech_mask
        for aug in self.augmentations:
            features, speech_mask = aug(features, speech_mask)
        return features, speech_mask


def create_augmentation_pipeline(config: Dict) -> Optional[CompositeAugmentation]:
    """
    Create augmentation pipeline from config.
    
    Config format:
    {
        'enabled': bool,
        'augmentations': {
            'node_dropout': {'p': 0.5, 'dropout_rate': 0.1},
            'temporal_jitter': {'p': 0.5, 'max_skip': 2, 'max_duplicate': 1},
            'gaussian_noise': {'p': 0.5, 'noise_std': 0.01},
            'feature_scaling': {'p': 0.5, 'scale_range': [0.95, 1.05]},
            'horizontal_flip': {'p': 0.5},
            'face_orientation': {'p': 0.5}
        },
        'feature_level': 'B0'  # For orientation-aware augmentations
    }
    
    Returns:
        CompositeAugmentation or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    augmentations = []
    aug_config = config.get('augmentations', {})
    feature_level = config.get('feature_level', 'B0')
    
    # Node feature dropout
    if 'node_dropout' in aug_config:
        aug_params = aug_config['node_dropout']
        augmentations.append(NodeFeatureDropout(
            p=aug_params.get('p', 0.5),
            dropout_rate=aug_params.get('dropout_rate', 0.1)
        ))
    
    # Temporal jitter
    if 'temporal_jitter' in aug_config:
        aug_params = aug_config['temporal_jitter']
        augmentations.append(TemporalJitter(
            p=aug_params.get('p', 0.5),
            max_skip=aug_params.get('max_skip', 2),
            max_duplicate=aug_params.get('max_duplicate', 1)
        ))
    
    # Gaussian noise
    if 'gaussian_noise' in aug_config:
        aug_params = aug_config['gaussian_noise']
        augmentations.append(GaussianNoise(
            p=aug_params.get('p', 0.5),
            noise_std=aug_params.get('noise_std', 0.01)
        ))
    
    # Feature scaling
    if 'feature_scaling' in aug_config:
        aug_params = aug_config['feature_scaling']
        scale_range = aug_params.get('scale_range', [0.95, 1.05])
        augmentations.append(FeatureScaling(
            p=aug_params.get('p', 0.5),
            scale_range=tuple(scale_range)
        ))
    
    # Horizontal flip
    if 'horizontal_flip' in aug_config:
        aug_params = aug_config['horizontal_flip']
        augmentations.append(HorizontalFlip(
            p=aug_params.get('p', 0.5),
            feature_level=feature_level
        ))
    
    # Face rotation
    if 'face_rotation' in aug_config:
        aug_params = aug_config['face_rotation']
        augmentations.append(FaceRotation(
            p=aug_params.get('p', 0.5),
            max_angle_deg=aug_params.get('max_angle_deg', 15.0)
        ))
    
    # Face translation
    if 'face_translation' in aug_config:
        aug_params = aug_config['face_translation']
        augmentations.append(FaceTranslation(
            p=aug_params.get('p', 0.5),
            max_translation=aug_params.get('max_translation', 0.02)
        ))
    
    # Face orientation aware
    if 'face_orientation' in aug_config:
        aug_params = aug_config['face_orientation']
        augmentations.append(FaceOrientationAwareAugmentation(
            p=aug_params.get('p', 0.5),
            max_rotation_deg=aug_params.get('max_rotation_deg', 10.0)
        ))
    
    if not augmentations:
        return None
    
    return CompositeAugmentation(augmentations)

