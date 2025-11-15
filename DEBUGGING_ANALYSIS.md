# Model Debugging Analysis

## Executive Summary

**Status: SEVERE OVERFITTING DETECTED**

- **Train Loss**: 3.28 → 0.28 (approaching 0)
- **Train Acc**: 0.17 → 0.91 (91%)
- **Val Loss**: 3.80 → 15.80 (INCREASING - bad!)
- **Val Acc**: 0.11 (STAGNANT at 11%)
- **Model Parameters**: 33,219,694 (33M - TOO COMPLEX)

## 1. Preprocessing Pipeline Analysis

### ✅ Landmarks Extraction
- **Status**: CORRECT
- **Source**: MediaPipe FaceMesh
- **ROI**: 88 landmarks (lips, jaw, cheeks, chin)
- **Normalization**: Face-relative normalization to [0, 1]
- **Temporal Smoothing**: Exponential moving average (α=0.7)
- **Format**: `[T, N, 3]` where N=88, T=variable

### ⚠️ Connection/Edge Index
- **Status**: MISMATCH DETECTED
- **Extractor** (`facemesh_extractor.py`): 
  - Has anatomical edge connections defined (`EDGE_PAIRS_ORIGINAL`)
  - Builds proper edge index with anatomical + k-NN connections
  - Edge index stored in `self.edge_index` but **NOT SAVED** to .pt files
- **Dataset** (`dataset.py`):
  - **IGNORES** extractor's edge index
  - Rebuilds edges using **simple k-NN (k=5)** in index space
  - This loses anatomical structure information!
- **Model** (`combined.py`):
  - Also rebuilds edges using k-NN (k=5)
  - **DOUBLE MISMATCH**: Neither uses anatomical edges

**Issue**: Anatomical connections (lips, jaw structure) are lost. Model uses arbitrary k-NN connections.

### ✅ Action Units (AU)
- **Status**: CORRECT
- **Count**: 18 AUs computed per frame
- **Features**: AU10, AU12, AU15, AU17, AU18, AU20, AU23, AU25, AU26, AU27
- **Format**: `[T, 18]` normalized to [0, 1]
- **Computation**: Based on landmark distances and positions

### ✅ Geometric Features
- **Status**: CORRECT
- **Count**: 10 geometric features per frame
- **Features**: Mouth width, height, aspect ratio, jaw opening, protrusion, area, symmetry
- **Format**: `[T, 10]` normalized to [0, 1]

### ✅ Feature Engineering
- **Status**: CORRECT
- **Node Features**: `[x, y, z] + AU[18] + Geometric[10] = 31 dims`
- **Broadcasting**: AU and geometric features broadcasted to all nodes
- **Format**: `[T, N, 31]` where N=88 nodes

### ❌ Speech Mask
- **Status**: EXTRACTED BUT NOT USED
- **Extraction**: Parsed from .txt files (Start/End times)
- **Format**: `[T]` binary mask (1.0 = speech, 0.0 = silence)
- **Problem**: 
  - Mask is extracted and passed to model
  - **NEVER USED** in forward pass
  - Model processes all frames equally, including silence
- **Impact**: Model learns from non-speech frames, adding noise

## 2. Dataset Loading Analysis

### ✅ Data Loading
- **Status**: CORRECT
- **Files**: `train.pt`, `val.pt`, `test.pt`
- **Format**: List of dictionaries with all features
- **PyG Integration**: Properly converts to PyG Data objects
- **Batching**: Works correctly with PyG DataLoader

### ⚠️ Edge Index Issue
- Dataset rebuilds edges instead of using extractor's anatomical edges
- This is a **design flaw** - anatomical structure is lost

## 3. Model Architecture Analysis

### Model Components

#### Spatial Module (`SpatialGCN`)
- **Layers**: 4 GCN layers (config: `spatial_layers=4`)
- **Hidden Dim**: 384 (`spatial_dim=384`)
- **Output**: Mean + Max pooling → `768 dim` (384*2)
- **Dropout**: 0.6 with progressive increase
- **Status**: Reasonable complexity

#### Temporal Module (3 Branches!)

**Branch 1: LSTM**
- **Layers**: 2 LSTM layers (bidirectional)
- **Hidden Dim**: 384 (`temporal_dim=384`)
- **Output**: `768 dim` (384*2)

**Branch 2: Transformer Attention**
- **Layers**: 1-2 Transformer layers
- **Heads**: 8 attention heads
- **Hidden Dim**: 384
- **Output**: `384 dim`

**Branch 3: Conv1D**
- **Layers**: 2 Conv1D layers
- **Kernel**: 3, padding 1
- **Output**: `384 dim`

#### Fusion Module
- **Gating**: Adaptive fusion gate (3-way softmax)
- **Fusion**: Concatenates all 3 branches + gated blend
- **Output**: `1536 dim` (384*4)

#### Classifier
- **Layers**: 3 fully connected layers
- **Dims**: 1536 → 768 → 384 → num_classes
- **Dropout**: 0.6-0.9 (progressive)

### Model Complexity

**Total Parameters**: 33,219,694 (33M)

**Breakdown**:
- Spatial GCN: ~1-2M
- Temporal LSTM: ~5-8M
- Temporal Attention: ~3-5M
- Temporal Conv1D: ~1-2M
- Fusion + Classifier: ~15-20M

**Problem**: 
- **TOO COMPLEX** for the dataset size
- Multiple redundant temporal branches
- Large classifier head
- High capacity → overfitting

## 4. Training Analysis

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW (weight_decay=2e-4)
- **Scheduler**: CosineAnnealingLR
- **Dropout**: 0.6
- **Label Smoothing**: 0.1
- **Gradient Clipping**: 1.0

### Training Progress (from checkpoint_summary.csv)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|------------|------------|----------|---------|--------|
| 33    | 3.28       | 0.17       | 3.80     | 0.11    | 0.06   |
| 100   | 2.48       | 0.31       | 5.83     | 0.10    | 0.06   |
| 200   | 1.39       | 0.57       | 10.12    | 0.09    | 0.06   |
| 300   | 0.73       | 0.76       | 11.82    | 0.09    | 0.07   |
| 400   | 0.41       | 0.87       | 14.70    | 0.10    | 0.08   |
| 500   | 0.28       | 0.91       | 15.80    | 0.10    | 0.08   |
| 600+  | ~0.07      | ~0.07      | -        | 0.07    | 0.01   |

### Overfitting Indicators

1. **Train Loss → 0**: Model memorizing training data
2. **Val Loss Increasing**: Generalization getting worse
3. **Val Acc Stagnant**: No learning on validation set
4. **Large Gap**: Train acc (91%) vs Val acc (11%) = 80% gap

## 5. Root Causes

### Primary Issues

1. **Model Too Complex** (33M params)
   - Multiple redundant temporal branches
   - Large classifier head
   - High capacity relative to dataset

2. **Speech Mask Not Used**
   - Model processes silence frames
   - Adds noise to learning signal

3. **Edge Connections Mismatch**
   - Anatomical structure lost
   - Using arbitrary k-NN instead of meaningful connections

4. **Dataset Size vs Model Capacity**
   - Need to check dataset size
   - Likely insufficient data for 33M params

### Secondary Issues

1. **Feature Redundancy**
   - Broadcasting AU/geometric to all nodes creates redundancy
   - Could use global features more efficiently

2. **Temporal Processing**
   - Processing all frames equally
   - Should focus on speech frames

## 6. Recommendations

### Immediate Fixes

1. **Simplify Model**
   - Remove redundant temporal branches (keep only LSTM)
   - Reduce hidden dimensions (384 → 256 or 128)
   - Reduce classifier size
   - Target: <10M parameters

2. **Use Speech Mask**
   - Apply mask to temporal sequence
   - Only process speech frames
   - Weight loss by speech mask

3. **Fix Edge Connections**
   - Save edge index from extractor to .pt files
   - Use anatomical edges in dataset/model
   - Or at least use distance-based k-NN (not index-based)

4. **Increase Regularization**
   - Higher dropout (0.7-0.8)
   - Stronger weight decay (1e-3)
   - Data augmentation
   - Early stopping (already implemented)

### Long-term Improvements

1. **Architecture Simplification**
   - Single temporal branch (LSTM only)
   - Smaller hidden dimensions
   - Simpler fusion

2. **Feature Engineering**
   - Use global features more efficiently
   - Consider attention over nodes instead of broadcasting

3. **Data Analysis**
   - Check dataset size and class distribution
   - Analyze if dataset is sufficient for task

## 7. What the Model Does

### Spatial Processing
1. For each timestep t:
   - Takes node features `[N, 31]`
   - Applies GCN layers to learn spatial relationships
   - Pools to graph-level embedding `[768]`

### Temporal Processing
1. Stacks spatial embeddings: `[T, 768]`
2. Processes with 3 branches:
   - **LSTM**: Sequential modeling
   - **Attention**: Long-range dependencies
   - **Conv1D**: Local temporal patterns
3. Fuses branches with adaptive gating
4. Classifies to word classes

### Issues
- Too many branches (redundant)
- No use of speech mask (processes silence)
- Edge connections not anatomical (loses structure)

## 8. Next Steps

1. ✅ **Analysis Complete** (this document)
2. ⏳ **Organize Directory Structure**
3. ⏳ **Create Simplified Model**
4. ⏳ **Fix Speech Mask Usage**
5. ⏳ **Fix Edge Connections**
6. ⏳ **Test with Simplified Model**



