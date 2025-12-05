# Face-Aware Augmentation Guide

This guide explains how to use face-aware augmentations for lip reading GNN models.

## Overview

All augmentations are designed to:
1. **Keep features and speech_mask synchronized** - Any temporal augmentation automatically updates the speech mask
2. **Respect facial geometry** - Augmentations preserve anatomical relationships
3. **Be memory-efficient** - Runtime augmentations don't explode RAM (operate per-sample)
4. **Handle face orientation** - Consider left/right facing and face shape variations

## Available Augmentations

### 1. Node Feature Dropout
Randomly zero out some node features per frame (spatial dropout).

**Use case**: Robustness to missing landmarks, regularization

**Config:**
```yaml
node_dropout:
  p: 0.5              # Probability of applying
  dropout_rate: 0.1   # Fraction of features to zero (0.0-1.0)
```

### 2. Temporal Jitter
Small random frame skipping or duplication. **Automatically synchronizes speech_mask**.

**Use case**: Temporal robustness, handling frame rate variations

**Config:**
```yaml
temporal_jitter:
  p: 0.5              # Probability of applying
  max_skip: 2         # Max frames to skip (won't skip >25% of frames)
  max_duplicate: 1    # Max frames to duplicate (won't duplicate >10% of frames)
```

**Important**: Speech mask is automatically adjusted to match frame changes.

### 3. Gaussian Noise
Add small Gaussian noise proportional to feature scale.

**Use case**: Generalization, robustness to measurement noise

**Config:**
```yaml
gaussian_noise:
  p: 0.5              # Probability of applying
  noise_std: 0.01     # Noise standard deviation (relative to feature scale)
```

### 4. Feature Scaling
Small random scaling of all features (per video).

**Use case**: Handling different face sizes, scale variations

**Config:**
```yaml
feature_scaling:
  p: 0.5              # Probability of applying
  scale_range: [0.95, 1.05]  # Min/max scale factors
```

### 5. Horizontal Flip
Mirror augmentation (flips x-coordinates and x-velocity).

**Use case**: Left/right face orientation variations

**Config:**
```yaml
horizontal_flip:
  p: 0.5              # Probability of applying
```

**Note**: Only works if features contain B0 (x, y coordinates) as first 2 features.

### 6. Face Rotation
2D rotation of face landmarks around face center.

**Use case**: Handling head rotation, yaw variations

**Config:**
```yaml
face_rotation:
  p: 0.5              # Probability of applying
  max_angle_deg: 15.0  # Maximum rotation angle in degrees (±15°)
```

**Note**: Automatically rotates velocities (B1 features) correctly.

### 7. Face Translation
Small random translation of face landmarks.

**Use case**: Handling slight head movement, camera shift

**Config:**
```yaml
face_translation:
  p: 0.5              # Probability of applying
  max_translation: 0.02  # Maximum translation in normalized coordinates (0-1 range)
```

### 8. Face Orientation Aware (Enhanced)
Augmentation that detects and adapts to face orientation (left/right/center).
Now includes proper rotation correction.

**Use case**: Handling different face orientations in dataset, normalizing orientation

**Config:**
```yaml
face_orientation:
  p: 0.5              # Probability of applying
  max_rotation_deg: 10.0  # Maximum rotation angle for orientation correction
```

## Configuration Example

Add to your YAML config file:

```yaml
data:
  partition: mouth
  feature_level: B1
  feature_dir: /path/to/features

model:
  name: gin_lstm_mamba
  params:
    # ... model params ...

training:
  # ... training params ...

# Augmentation configuration (optional)
augmentation:
  enabled: true
  feature_level: B1  # Must match data.feature_level
  
  augmentations:
    # Enable specific augmentations
    node_dropout:
      p: 0.5
      dropout_rate: 0.1
    
    temporal_jitter:
      p: 0.3
      max_skip: 2
      max_duplicate: 1
    
    gaussian_noise:
      p: 0.4
      noise_std: 0.01
    
    feature_scaling:
      p: 0.3
      scale_range: [0.95, 1.05]
    
    horizontal_flip:
      p: 0.5
    
    face_rotation:
      p: 0.3
      max_angle_deg: 15.0
    
    face_translation:
      p: 0.3
      max_translation: 0.02
    
    face_orientation:
      p: 0.3
      max_rotation_deg: 10.0
```

## Memory Considerations

✅ **Safe (Memory-Efficient)**:
- All augmentations operate in `__getitem__` (per-sample)
- No dataset duplication
- Temporary tensors are garbage collected
- Works with lazy loading and float16 storage

❌ **Would Explode RAM** (Don't do this):
- Pre-computing all augmentations
- Storing multiple augmented versions
- Creating augmented datasets offline

## Speech Mask Synchronization

**Critical**: All temporal augmentations automatically keep speech_mask synchronized:

```python
# Example: Temporal jitter
features, speech_mask = temporal_jitter(features, speech_mask)
# speech_mask is automatically adjusted to match frame changes
```

Spatial augmentations (dropout, noise, scaling) don't change speech_mask (they only affect features).

## Face-Aware Considerations

### Face Orientation Detection
The `FaceOrientationAwareAugmentation` detects face orientation by:
1. Computing center of mass of x-coordinates
2. Classifying as 'left', 'right', or 'center'
3. Applying orientation-specific transformations

### Anatomical Constraints
All augmentations preserve:
- Relative spatial relationships between landmarks
- Temporal ordering (even with jitter)
- Feature scales and distributions

## Best Practices

1. **Start conservative**: Use low probabilities (p=0.3-0.5) and small magnitudes
2. **Test incrementally**: Enable one augmentation at a time to see effects
3. **Monitor validation**: Augmentations should improve generalization, not hurt it
4. **Consider feature level**: Some augmentations (horizontal_flip) require B0 features
5. **Speech mask aware**: Always verify speech_mask stays synchronized after temporal augmentations

## Example: Minimal Augmentation

For a conservative setup:

```yaml
augmentation:
  enabled: true
  feature_level: B1
  
  augmentations:
    node_dropout:
      p: 0.3
      dropout_rate: 0.05
    
    gaussian_noise:
      p: 0.3
      noise_std: 0.005
```

## Example: Aggressive Augmentation

For maximum robustness:

```yaml
augmentation:
  enabled: true
  feature_level: B1
  
  augmentations:
    node_dropout:
      p: 0.5
      dropout_rate: 0.15
    
    temporal_jitter:
      p: 0.5
      max_skip: 3
      max_duplicate: 2
    
    gaussian_noise:
      p: 0.5
      noise_std: 0.02
    
    feature_scaling:
      p: 0.4
      scale_range: [0.90, 1.10]
    
    horizontal_flip:
      p: 0.5
```

## Troubleshooting

**Issue**: Speech mask out of sync
- **Solution**: Ensure you're using the provided augmentation classes (they handle synchronization automatically)

**Issue**: Memory usage increases
- **Solution**: Check that `num_workers=0` and `lazy_load=True` (default). Augmentations shouldn't increase memory significantly.

**Issue**: Augmentation not applying
- **Solution**: Check that `enabled: true` and augmentation probabilities are > 0

**Issue**: Horizontal flip not working
- **Solution**: Ensure feature_level includes B0 (x, y coordinates)

## Integration

Augmentations are automatically:
- Applied only to training data (not validation/test)
- Integrated with lazy loading
- Compatible with float16 storage
- Synchronized with speech_mask

No code changes needed beyond config file!

