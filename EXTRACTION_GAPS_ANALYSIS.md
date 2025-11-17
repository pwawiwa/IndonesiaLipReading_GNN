# Extraction Gaps Analysis - Based on Recent Research

## Research Findings from Recent Papers

### Key Papers Found:
1. **Multi-View Information Bottleneck (MVIB)** - MDPI Entropy 2024
2. **Gabor-Based Feature Extraction** - MDPI Entropy 2020
3. **Inner Lip Contour Features** - CalPoly Research
4. **GLip Framework** - Global + Local Features (ArXiv 2024)
5. **Viseme-based Features** - ArXiv 2023

## What's Missing from Your Current Extraction

### 🔴 CRITICAL MISSING FEATURES

#### 1. **Inner Lip Contour Features** (High Priority)
**Research Finding**: Inner lip contours + tongue/teeth visibility significantly improve recognition

**What You Have**:
- ✅ Inner lip landmarks (indices 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, etc.)
- ❌ **NO separate inner lip contour features**
- ❌ **NO tongue visibility detection**
- ❌ **NO teeth visibility detection**

**Missing Features**:
```python
# Inner lip specific features
inner_lip_width = distance(inner_corner_left, inner_corner_right)
inner_lip_height = distance(inner_upper_center, inner_lower_center)
inner_lip_area = compute_polygon_area(inner_lip_contour)

# Tongue visibility (approximate from mouth opening)
tongue_visibility = mouth_opening_height > threshold

# Teeth visibility (approximate from mouth opening and lip separation)
teeth_visibility = (mouth_opening_height > threshold) and (lip_separation > threshold)
```

#### 2. **Gabor-Based Features** (Medium Priority)
**Research Finding**: Gabor filters extract 3D geometric features that are lightweight and interpretable

**What You Have**:
- ❌ **NO Gabor filter features**
- ✅ Basic geometric features (width, height, aspect ratio)

**Missing**:
- Gabor filter responses on lip region
- Multi-scale Gabor features
- Orientation-specific features

#### 3. **Recurrence Plots** (Medium Priority)
**Research Finding**: Transforming trajectories into recurrence plots captures structural dynamics

**What You Have**:
- ✅ Raw landmark trajectories
- ❌ **NO recurrence plot representation**

**Missing**:
- Recurrence plot generation from trajectories
- Texture-based features from recurrence plots

#### 4. **Viseme-Based Features** (High Priority)
**Research Finding**: Visemes (phonetically similar lip shapes) reduce word error rate

**What You Have**:
- ❌ **NO viseme classification**
- ❌ **NO viseme-based features**

**Missing**:
- Viseme classification per frame
- Viseme transition features
- Viseme duration features

#### 5. **Global + Local Feature Integration** (Medium Priority)
**Research Finding**: Combining global facial features with local lip movements improves robustness

**What You Have**:
- ✅ Local lip features (ROI landmarks)
- ❌ **NO global facial context**
- ❌ **NO integration of global + local**

**Missing**:
- Global face pose/expression features
- Face context features (cheeks, jaw, etc.)
- Integration mechanism for global + local

### 🟡 MODERATE PRIORITY MISSING

#### 6. **Multi-Scale Temporal Features**
**What You Have**:
- ✅ Frame-level features
- ✅ Basic temporal smoothing
- ❌ **NO multi-scale temporal analysis**

**Missing**:
- Short-term temporal patterns (2-3 frames)
- Medium-term patterns (5-10 frames)
- Long-term patterns (entire sequence)

#### 7. **Frequency Domain Features**
**What You Have**:
- ❌ **NO frequency analysis**

**Missing**:
- FFT of landmark trajectories
- Dominant frequencies
- Spectral energy features

#### 8. **Relative Motion Between Landmarks**
**What You Have**:
- ✅ Absolute velocity/acceleration (just added)
- ❌ **NO relative motion between key landmarks**

**Missing**:
- Upper lip vs lower lip relative motion
- Corner-to-corner relative motion
- Symmetry features (left vs right)

#### 9. **Phoneme-Specific Features**
**What You Have**:
- ✅ Generic geometric features
- ❌ **NO phoneme-specific features**

**Missing**:
- Bilabial features (p, b, m)
- Labiodental features (f, v)
- Vowel rounding features (u, o vs i, e)

### 🟢 LOW PRIORITY (Nice to Have)

#### 10. **Appearance Features**
- Lip color/intensity
- Texture features
- Lighting conditions

#### 11. **Head Pose Normalization**
- Face orientation normalization
- Viewpoint-invariant features

## Recommended Implementation Priority

### Phase 1: Quick Wins (Implement First)
1. ✅ **Motion Features** - DONE (velocity + acceleration)
2. **Inner Lip Contour Features** - Add separate inner lip measurements
3. **Relative Motion Features** - Add relative motion between landmarks
4. **Tongue/Teeth Visibility** - Approximate from mouth opening

### Phase 2: Medium Effort
5. **Viseme Classification** - Classify each frame into viseme categories
6. **Multi-Scale Temporal** - Add short/medium/long-term temporal features
7. **Frequency Features** - Add FFT-based features

### Phase 3: Advanced
8. **Gabor Features** - Implement Gabor filter extraction
9. **Recurrence Plots** - Generate recurrence plot features
10. **Global + Local Integration** - Add global facial context

## Specific Feature Additions Needed

### Inner Lip Features (Add to `compute_geometric_features`):
```python
# Inner lip dimensions
inner_mouth_width = safe_dist(78, 308)  # Inner corners
inner_mouth_height = safe_dist(13, 14)  # Inner upper/lower centers
inner_mouth_area = inner_mouth_width * inner_mouth_height

# Inner vs outer ratio
inner_outer_width_ratio = inner_mouth_width / (mouth_width + 1e-6)
inner_outer_height_ratio = inner_mouth_height / (mouth_height + 1e-6)
```

### Tongue/Teeth Visibility (Add to `compute_action_units`):
```python
# Tongue visibility (approximate)
mouth_opening = safe_dist(13, 14)
tongue_visible = 1.0 if mouth_opening > 0.15 else 0.0

# Teeth visibility (approximate)
teeth_visible = 1.0 if (mouth_opening > 0.12) and (inner_mouth_height > 0.08) else 0.0
```

### Relative Motion (Add new function):
```python
def compute_relative_motion(self, velocity: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Compute relative motion between key landmarks"""
    # Upper vs lower lip relative motion
    upper_lip_idx = self._get_roi_index(13)  # Upper lip center
    lower_lip_idx = self._get_roi_index(14)  # Lower lip center
    
    if upper_lip_idx and lower_lip_idx:
        relative_vertical = velocity[upper_lip_idx, 1] - velocity[lower_lip_idx, 1]
    else:
        relative_vertical = 0.0
    
    # Left vs right corner relative motion
    left_corner_idx = self._get_roi_index(61)
    right_corner_idx = self._get_roi_index(291)
    
    if left_corner_idx and right_corner_idx:
        relative_horizontal = velocity[left_corner_idx, 0] - velocity[right_corner_idx, 0]
    else:
        relative_horizontal = 0.0
    
    return np.array([relative_vertical, relative_horizontal])
```

## Expected Impact

Based on research:
- **Inner lip features**: +5-10% accuracy improvement
- **Viseme features**: Significant reduction in word error rate
- **Relative motion**: Better discrimination of similar words
- **Tongue/teeth visibility**: Better phoneme discrimination

## Next Steps

1. **Add inner lip contour features** to geometric features
2. **Add tongue/teeth visibility** to action units
3. **Add relative motion features** as new feature type
4. **Consider viseme classification** for future enhancement

These additions should help break through the 18-25% accuracy plateau!

