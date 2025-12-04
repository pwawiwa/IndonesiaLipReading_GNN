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


class FaceOrientationAwareAugmentation(FaceAwareAugmentation):
    """
    Augmentation that considers face orientation (left/right facing).
    Detects face orientation from landmark positions and applies orientation-aware augmentations.
    """
    def __init__(self, p: float = 0.5, feature_level: str = 'B0'):
        """
        Args:
            p: Probability of applying augmentation
            feature_level: Feature level to determine if we have spatial coordinates
        """
        super().__init__(p)
        self.feature_level = feature_level
    
    def _detect_face_orientation(self, features: torch.Tensor) -> str:
        """
        Detect if face is facing left or right based on landmark positions.
        Uses first frame's x-coordinates (B0 feature index 0).
        
        Returns:
            'left', 'right', or 'center'
        """
        if features.shape[2] < 2:
            return 'center'
        
        # Use first frame's x-coordinates
        x_coords = features[0, :, 0]  # (nodes,)
        
        # Compute center of mass in x-direction
        x_center = x_coords.mean().item()
        
        # Heuristic: if center < 0.5, face is on left side (facing right)
        # if center > 0.5, face is on right side (facing left)
        if x_center < 0.45:
            return 'right'  # Face on left, facing right
        elif x_center > 0.55:
            return 'left'   # Face on right, facing left
        else:
            return 'center'
    
    def apply(self, features: torch.Tensor, speech_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detect orientation
        orientation = self._detect_face_orientation(features)
        
        # Apply orientation-specific augmentation
        # For now, just apply small rotation-like transformation based on orientation
        # This is a placeholder - can be extended with more sophisticated transformations
        
        # Convert to float32 if needed
        was_float16 = features.dtype == torch.float16
        if was_float16:
            features = features.float()
        
        # Small translation based on orientation (simulates slight rotation)
        if orientation == 'left':
            # Slight shift to compensate for left-facing
            if features.shape[2] >= 2:
                features[:, :, 0] = features[:, :, 0] + 0.01  # Small right shift
        elif orientation == 'right':
            # Slight shift to compensate for right-facing
            if features.shape[2] >= 2:
                features[:, :, 0] = features[:, :, 0] - 0.01  # Small left shift
        
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
    
    # Face orientation aware
    if 'face_orientation' in aug_config:
        aug_params = aug_config['face_orientation']
        augmentations.append(FaceOrientationAwareAugmentation(
            p=aug_params.get('p', 0.5),
            feature_level=feature_level
        ))
    
    if not augmentations:
        return None
    
    return CompositeAugmentation(augmentations)

