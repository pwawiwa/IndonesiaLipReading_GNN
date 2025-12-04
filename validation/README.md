# Pipeline Validation

Comprehensive validation system for checking every variable and number passing through the pipeline.

## Overview

The validation system performs exhaustive checks on:

1. **Data Extraction** - Landmarks, adjacency matrices, speech masks, metadata
2. **Feature Engineering** - Feature shapes, dtypes, ranges, normalization
3. **Data Loading** - Dataset, DataLoader, collate function
4. **Model** - Initialization, forward pass, gradients, outputs
5. **Training** - Loss values, gradient norms, metrics

## Usage

### Basic Usage

Validate using a training config file:

```bash
python validation/pipeline_validator.py --config results/mouth/B1/gin_lstm_mamba/seed_0/config.yaml
```

### With Extraction/Feature Files

Validate specific files:

```bash
python validation/pipeline_validator.py \
    --config results/mouth/B1/gin_lstm_mamba/seed_0/config.yaml \
    --extraction-file data/extracted/mouth/mouth_train.pt \
    --feature-file data/features/B1/mouth_train.pt
```

### Using Shell Script

```bash
bash scripts/validate_pipeline.sh \
    --config results/mouth/B1/gin_lstm_mamba/seed_0/config.yaml \
    --extraction-file data/extracted/mouth/mouth_train.pt \
    --feature-file data/features/B1/mouth_train.pt \
    --output results/validation_report.txt
```

## Validation Checks

### Extraction File Checks

- ✅ File loads successfully
- ✅ Required keys present (partition, split, adjacency, videos, word_to_label)
- ✅ Adjacency matrix: shape, dtype, symmetry, finite values
- ✅ Landmarks: shape (frames, n_nodes, 2), dtype float32, finite values
- ✅ Speech masks: shape (frames,), binary (0/1), matches frame count
- ✅ Labels: valid range, correct type
- ✅ Video consistency: all videos have matching node counts

### Feature File Checks

- ✅ File loads successfully
- ✅ Required keys present
- ✅ Feature level matches expected
- ✅ Features: shape (frames, n_nodes, n_features), dtype float32, finite values
- ✅ Feature count matches expected (B0=2, B1=3, B2=2, B3=4)
- ✅ Speech masks valid
- ✅ Video consistency

### DataLoader Checks

- ✅ Dataset loads successfully
- ✅ Dataset properties: num_classes, num_nodes, feature_dim
- ✅ Batch shapes: (batch, frames, nodes, features)
- ✅ Batch dtypes: float32 for features/masks, long for labels
- ✅ All tensors finite, no NaN/Inf
- ✅ Label ranges valid
- ✅ Adjacency shape matches

### Model Checks

- ✅ Model builds successfully
- ✅ Forward pass works
- ✅ Output shape: (batch, num_classes)
- ✅ Outputs finite, no NaN/Inf
- ✅ Backward pass works
- ✅ Gradients finite, no NaN/Inf
- ✅ Gradient norms reasonable

### Training Step Checks

- ✅ Training step executes
- ✅ Loss finite and reasonable
- ✅ Gradient norms finite
- ✅ Accuracy in valid range [0, 100]
- ✅ Optimizer step works

## Output Format

The validator produces a detailed report with:

- **Status**: ✅ PASSED, ❌ FAILED, or ⚠️ UNKNOWN
- **Errors**: Detailed error messages for failures
- **Warnings**: Non-critical issues
- **Details**: Key values and statistics

Example output:

```
✅ PASSED - 001. Extraction File: mouth_train.pt
  DETAILS:
    • partition: mouth
    • split: train
    • num_videos: 1500
    • n_nodes: 78
    • n_edges: 234

❌ FAILED - 002. Feature File: mouth_train.pt (B1)
  ERRORS:
    ❌ Feature count mismatch: expected 3, got 2
  DETAILS:
    • featureset: B1
    • sample_n_features: 2
```

## Integration

Add validation to your training pipeline:

```python
from validation import PipelineValidator

validator = PipelineValidator()
validator.validate_extraction_file("data/extracted/mouth/mouth_train.pt", "mouth")
validator.validate_feature_file("data/features/B1/mouth_train.pt", "B1")
# ... more checks ...
all_passed = validator.print_summary()
```

## Notes

- Validation uses CPU by default for safety (can use GPU for model checks)
- Checks first 10 videos in files for performance
- NaN/Inf checks are performed on all tensors
- Shape and dtype checks are strict
- Value range checks are configurable

