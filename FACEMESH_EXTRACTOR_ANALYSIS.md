# FaceMesh Extractor Analysis & Improvement Strategy

## Current Condition Assessment

### ✅ **Strengths**
1. **ROI-based extraction**: Focuses on mouth-centric region (60+ landmarks)
2. **Parallel processing**: Uses ThreadPoolExecutor for efficient extraction
3. **Motion features**: Extracts velocity and acceleration
4. **Quality checks**: 70% face detection threshold
5. **Speech timing**: Parses timing from .txt files
6. **Error handling**: Basic fallback for missing detections

### ❌ **Critical Issues**

#### 1. **Index Mapping Bug (CRITICAL)**
- **Problem**: `compute_action_units()` and `compute_geometric_features()` use hardcoded MediaPipe indices (0, 13, 61, 291, etc.)
- **But**: They receive ROI-normalized landmarks where indices are remapped (0-59 instead of original MediaPipe indices)
- **Impact**: Features are computed on wrong landmarks, leading to incorrect feature values
- **Example**: `safe_coord(13, 1)` tries to access ROI index 13, but MediaPipe landmark 13 might be at ROI index 20

#### 2. **Speech Timing Bug**
- **Problem**: Line 308 uses `end_match.group(1)` but should extract the end time correctly
- **Impact**: Speech mask might not be correctly applied

#### 3. **Normalization Issues**
- **Problem**: Per-dimension min-max normalization loses relative spatial relationships
- **Impact**: Features might not preserve mouth shape relationships

#### 4. **Missing Temporal Smoothing**
- **Problem**: No smoothing applied to landmarks across frames
- **Impact**: Jittery landmarks can introduce noise

#### 5. **Weak Fallback Strategy**
- **Problem**: Uses previous frame or zeros when face not detected
- **Impact**: Can introduce artifacts in sequences

## Improvement Strategy

### Phase 1: Fix Critical Bugs (Priority 1)

#### 1.1 Fix Index Mapping
- Create mapping from MediaPipe indices to ROI indices
- Update `compute_action_units()` and `compute_geometric_features()` to use ROI indices
- Add validation to ensure indices exist in ROI

#### 1.2 Fix Speech Timing Parser
- Correct the end time extraction bug
- Add validation for timing data

#### 1.3 Improve Normalization
- Use face-relative normalization (normalize relative to nose/face center)
- Preserve relative distances and angles

### Phase 2: Enhance Feature Extraction (Priority 2)

#### 2.1 Add Temporal Smoothing
- Apply moving average or Kalman filter to landmarks
- Reduce jitter while preserving motion

#### 2.2 Better Fallback Strategy
- Use interpolation between detected frames
- Apply temporal smoothing to fill gaps

#### 2.3 Enhanced Features
- Add frequency domain features (FFT of motion)
- Add relative angle features
- Add curvature features
- Add optical flow-like features

### Phase 3: Robustness & Quality (Priority 3)

#### 3.1 Improved Error Handling
- Better validation of extracted features
- More informative error messages
- Graceful degradation

#### 3.2 Quality Metrics
- Track detection quality per video
- Report statistics on extraction success
- Flag low-quality extractions

#### 3.3 Data Augmentation (Optional)
- Add augmentation during extraction
- Temporal warping
- Spatial jittering
- Noise injection

## Implementation Plan

### Step 1: Create Index Mapping Helper
```python
def _get_roi_index(self, mp_index: int) -> Optional[int]:
    """Map MediaPipe index to ROI index"""
    if mp_index in self.ROI_INDICES:
        return self.ROI_INDICES.index(mp_index)
    return None
```

### Step 2: Fix Action Units Computation
- Map all MediaPipe indices to ROI indices
- Add bounds checking
- Handle missing landmarks gracefully

### Step 3: Fix Geometric Features
- Same mapping approach
- Validate all indices before use

### Step 4: Improve Normalization
- Use nose tip (index 0) as reference
- Normalize relative to face center
- Preserve scale relationships

### Step 5: Add Temporal Smoothing
- Apply exponential moving average
- Configurable smoothing factor

## Expected Impact

### Before Fixes
- ❌ Incorrect feature values due to index mismatch
- ❌ Poor feature quality affecting model performance
- ❌ Jittery landmarks introducing noise

### After Fixes
- ✅ Correct feature extraction
- ✅ Better feature quality
- ✅ Smoother, more stable landmarks
- ✅ Improved model performance (estimated +5-10% accuracy)

## Testing Strategy

1. **Unit Tests**: Test index mapping with known landmarks
2. **Feature Validation**: Compare features before/after fixes
3. **Visual Inspection**: Plot landmarks and features
4. **Model Performance**: Retrain and compare accuracy

