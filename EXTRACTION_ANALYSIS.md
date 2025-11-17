# Facemesh Extraction Analysis - What's Missing?

## Current Extraction Features

### ✅ What You Have:
1. **Landmarks**: ~88 ROI landmarks (lips, jaw, cheeks)
2. **Action Units (AUs)**: 18 FACS-based features
3. **Geometric Features**: 10 geometric measurements
4. **Temporal Smoothing**: Exponential moving average
5. **Speech Mask**: Frame-level speech timing
6. **Motion Features**: Velocity and acceleration (computed but **NOT USED**)

## ❌ Critical Missing Features

### 1. **Motion Features Not Used** (CRITICAL)
**Problem**: You compute velocity and acceleration but they're never used in the model!

```python
# In extract_video():
velocity = np.diff(landmarks_sequence, axis=0)  # [T-1, N, 3]
acceleration = np.diff(velocity, axis=0)  # [T-2, N, 3]
# But these are saved and never used in dataset.py!
```

**Impact**: Motion is crucial for lip reading! Static landmarks miss temporal dynamics.

**Solution**: 
- Include velocity/acceleration as node features
- Or create motion-based AUs (rate of change of mouth opening, etc.)

### 2. **No Relative Motion Features**
**Problem**: Only absolute positions, no relative movements between landmarks.

**Missing**:
- Relative motion between upper/lower lip
- Jaw movement relative to lips
- Corner-to-corner motion
- Opening/closing speed

**Solution**: Add relative motion features:
```python
# Relative motion between key landmarks
lip_opening_velocity = velocity[:, upper_lip_idx] - velocity[:, lower_lip_idx]
corner_motion = velocity[:, left_corner_idx] - velocity[:, right_corner_idx]
```

### 3. **Limited Temporal Context**
**Problem**: Only simple smoothing, no temporal feature extraction.

**Missing**:
- Frame-to-frame differences (already computed but not used)
- Temporal derivatives (rate of change)
- Temporal frequency features (how fast mouth moves)
- Phase information (opening vs closing)

**Solution**: Add temporal features:
```python
# Temporal derivatives
mouth_opening_rate = np.diff(mouth_height, axis=0)
lip_protrusion_rate = np.diff(lip_protrusion, axis=0)
```

### 4. **No Intensity/Amplitude Features**
**Problem**: Only geometric measurements, no intensity of movements.

**Missing**:
- Movement amplitude (how wide mouth opens)
- Movement speed (how fast)
- Movement acceleration (how quickly speed changes)
- Peak opening/closing positions

**Solution**: Add intensity features:
```python
# Movement intensity
opening_amplitude = np.max(mouth_height) - np.min(mouth_height)
opening_speed = np.max(np.abs(np.diff(mouth_height)))
```

### 5. **Incomplete Action Units**
**Problem**: Only 18 AUs, missing speech-critical ones.

**Missing AUs**:
- **AU28: Lip Suck** - Important for certain phonemes
- **AU43: Eyes Closed** - Can affect mouth visibility
- **AU44: Squint** - Related to speech effort
- **AU45: Blink** - Temporal marker
- **AU46: Wink** - Rare but useful

**Also Missing**:
- **AU Intensity**: Current AUs are binary-like, need intensity levels
- **AU Combinations**: Some phonemes require AU combinations

### 6. **No Phoneme-Specific Features**
**Problem**: Generic features, not optimized for phoneme discrimination.

**Missing**:
- Features for bilabial sounds (p, b, m)
- Features for labiodental sounds (f, v)
- Features for rounded vowels (u, o)
- Features for unrounded vowels (i, e)

**Solution**: Add phoneme-specific geometric features:
```python
# Bilabial closure (lips together)
bilabial_closure = distance(upper_lip_center, lower_lip_center)

# Labiodental contact (upper lip to lower teeth)
labiodental = distance(upper_lip, lower_teeth_landmark)

# Lip rounding (for vowels)
lip_rounding = distance(corner_left, corner_right) / mouth_width
```

### 7. **No Head Pose Normalization**
**Problem**: Face orientation not normalized, causing variation.

**Missing**:
- Head rotation (yaw, pitch, roll)
- Face orientation normalization
- Viewpoint-invariant features

**Impact**: Same word with different head angles looks different.

**Solution**: Normalize head pose:
```python
# Compute head pose from face landmarks
# Normalize landmarks relative to face plane
```

### 8. **No Frequency Domain Features**
**Problem**: Only spatial features, no frequency analysis.

**Missing**:
- FFT of landmark trajectories
- Dominant frequencies of mouth movements
- Spectral features (energy in different frequency bands)

**Solution**: Add frequency features:
```python
# FFT of mouth opening over time
mouth_opening_fft = np.fft.fft(mouth_height_sequence)
dominant_freq = np.argmax(np.abs(mouth_opening_fft[1:])) + 1
```

### 9. **Normalization Issues**
**Problem**: Face-relative normalization might lose important scale information.

**Current**: Normalizes to [0,1] using face bounding box
**Issue**: 
- Loses absolute scale (big vs small mouth)
- Might normalize away important differences
- Different face sizes get same normalization

**Solution**: 
- Keep both normalized and absolute features
- Use multiple normalization schemes
- Add scale-invariant features (ratios)

### 10. **No Appearance Features**
**Problem**: Only geometry, no appearance/texture.

**Missing**:
- Lip color/intensity
- Texture features (smooth vs textured)
- Lighting conditions
- Contrast features

**Note**: MediaPipe doesn't provide this, would need image patches.

### 11. **Limited Graph Structure**
**Problem**: Simple k-NN edges, might miss important connections.

**Missing**:
- Phoneme-specific edge connections
- Dynamic edges based on mouth state
- Hierarchical connections (local + global)

**Solution**: 
- Use anatomical edges (already have but verify)
- Add phoneme-specific connections
- Use attention-based edge weighting

### 12. **No Multi-Scale Features**
**Problem**: Single scale features only.

**Missing**:
- Fine-grained (individual landmarks)
- Medium-scale (lip regions)
- Coarse-scale (entire mouth area)

**Solution**: Multi-scale feature extraction:
```python
# Fine: individual landmark positions
# Medium: lip region statistics
# Coarse: entire mouth bounding box features
```

### 13. **Speech Mask Not Optimized**
**Problem**: Binary mask, no intensity or confidence.

**Missing**:
- Speech intensity (how clearly speaking)
- Speech confidence (certainty of speech detection)
- Phoneme-level masks (if available)

**Solution**: 
- Use continuous mask (0-1) instead of binary
- Add speech intensity features

### 14. **No Context Features**
**Problem**: Only current frame features.

**Missing**:
- Previous frame context
- Future frame context (if available)
- Long-term temporal patterns

**Solution**: Add context windows:
```python
# Include features from t-1, t, t+1 frames
context_features = concatenate([features[t-1], features[t], features[t+1]])
```

## Recommended Improvements (Priority Order)

### 🔴 HIGH PRIORITY (Do First)

1. **Use Motion Features**
   - Include velocity/acceleration in node features
   - Add motion-based AUs

2. **Add Relative Motion**
   - Compute relative motion between key landmarks
   - Add to feature set

3. **Improve Temporal Features**
   - Add temporal derivatives
   - Include frame-to-frame differences

### 🟡 MEDIUM PRIORITY

4. **Enhance Action Units**
   - Add missing speech-critical AUs
   - Add AU intensity levels

5. **Phoneme-Specific Features**
   - Add features for different phoneme types
   - Optimize for Indonesian phonemes

6. **Head Pose Normalization**
   - Normalize face orientation
   - Add viewpoint-invariant features

### 🟢 LOW PRIORITY (Nice to Have)

7. **Frequency Domain Features**
8. **Multi-Scale Features**
9. **Appearance Features** (requires image patches)

## Implementation Example

Here's how to add motion features to your extraction:

```python
def compute_motion_features(self, landmarks_seq: np.ndarray) -> np.ndarray:
    """Compute motion-based features"""
    T, N, _ = landmarks_seq.shape
    
    if T < 2:
        return np.zeros((T, N, 3), dtype=np.float32)
    
    # Velocity
    velocity = np.diff(landmarks_seq, axis=0)  # [T-1, N, 3]
    
    # Acceleration
    acceleration = np.diff(velocity, axis=0) if velocity.shape[0] > 1 else np.zeros((0, N, 3))
    
    # Motion magnitude
    velocity_magnitude = np.linalg.norm(velocity, axis=2)  # [T-1, N]
    acceleration_magnitude = np.linalg.norm(acceleration, axis=2) if acceleration.shape[0] > 0 else np.zeros((0, N))
    
    # Pad to match T
    velocity_padded = np.pad(velocity, ((0, 1), (0, 0), (0, 0)), mode='edge')
    velocity_mag_padded = np.pad(velocity_magnitude, ((0, 1), (0, 0)), mode='edge')
    
    # Combine into motion features [T, N, 4]: [vx, vy, vz, |v|]
    motion_features = np.concatenate([
        velocity_padded,
        velocity_mag_padded[..., np.newaxis]
    ], axis=2)
    
    return motion_features.astype(np.float32)
```

Then include in node features:
```python
# In dataset.py, add motion to node features
motion = sample.get('motion', None)  # [T, N, 4]
if motion is not None:
    node_feat = torch.cat([node_pos, au_broadcast, geo_broadcast, motion[t]], dim=1)
    # Now [N, 3+18+10+4] = [N, 35]
```

## Summary

**Biggest Issues:**
1. ❌ Motion features computed but NOT USED
2. ❌ No relative motion between landmarks
3. ❌ Limited temporal context
4. ❌ Missing phoneme-specific features

**Quick Wins:**
- Use existing velocity/acceleration
- Add relative motion features
- Enhance temporal features

These changes could significantly improve your 18-25% accuracy!

