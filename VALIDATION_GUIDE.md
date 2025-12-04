# Pipeline Validation Guide

## Overview

A comprehensive validation system has been created to perform **brutal checking** of every variable and number passing through the pipeline. The validator produces a detailed checklist with PASS/FAIL status for each check.

## Files Created

1. **`validation/pipeline_validator.py`** - Main validation module
2. **`validation/__init__.py`** - Module initialization
3. **`validation/README.md`** - Detailed documentation
4. **`scripts/validate_pipeline.sh`** - Shell script for easy execution

## Quick Start

### Validate Using Config File

```bash
python3 validation/pipeline_validator.py \
    --config results/mouth/B1/gin_lstm_mamba/seed_0/config.yaml
```

### Validate Specific Files

```bash
python3 validation/pipeline_validator.py \
    --config results/mouth/B1/gin_lstm_mamba/seed_0/config.yaml \
    --extraction-file data/extracted/mouth/mouth_train.pt \
    --feature-file data/features/B1/mouth_train.pt \
    --output validation_report.txt
```

### Using Shell Script

```bash
bash scripts/validate_pipeline.sh \
    --config results/mouth/B1/gin_lstm_mamba/seed_0/config.yaml \
    --extraction-file data/extracted/mouth/mouth_train.pt \
    --output results/validation_report.txt
```

## What Gets Checked

### ✅ Extraction Stage
- File loading and structure
- Adjacency matrix: shape, dtype, symmetry, finite values
- Landmarks: shape (frames, n_nodes, 2), dtype float32, no NaN/Inf
- Speech masks: binary (0/1), matches frame count
- Labels: valid range, correct type
- Video consistency across all samples

### ✅ Feature Engineering Stage
- Feature file structure and metadata
- Features: shape (frames, n_nodes, n_features), dtype float32
- Feature counts: B0=2, B1=3, B2=2, B3=4
- No NaN/Inf values
- Normalization consistency
- Video consistency

### ✅ Data Loading Stage
- Dataset initialization
- DataLoader batch shapes
- Collate function: padding, dtype conversion
- Batch consistency: all tensors match expected shapes
- Label validation

### ✅ Model Stage
- Model initialization
- Forward pass: input/output shapes
- Output ranges and finite values
- Backward pass: gradient flow
- Gradient norms and NaN/Inf checks

### ✅ Training Stage
- Loss values: finite and reasonable
- Gradient norms: finite and reasonable
- Accuracy: valid range [0, 100]
- Optimizer step execution

## Output Format

The validator produces a detailed report:

```
================================================================================
PIPELINE VALIDATION SUMMARY
================================================================================

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

✅ PASSED - 003. DataLoader: mouth_train.pt
  DETAILS:
    • num_classes: 100
    • num_nodes: 78
    • feature_dim: 5
    • batch_size: 2

================================================================================
TOTAL CHECKS: 5
✅ PASSED: 4
❌ FAILED: 1
⚠️  WARNINGS: 0
================================================================================
```

## Integration Example

```python
from validation import PipelineValidator
from pathlib import Path

# Create validator
validator = PipelineValidator()

# Validate extraction
validator.validate_extraction_file(
    "data/extracted/mouth/mouth_train.pt",
    partition="mouth",
    expected_n_nodes=78
)

# Validate features
validator.validate_feature_file(
    "data/features/B1/mouth_train.pt",
    feature_level="B1",
    expected_n_features=3
)

# Validate dataloader
validator.validate_dataloader(
    "mouth_train.pt",
    batch_size=2,
    feature_level="B1",
    feature_dir="data/features",
    expected_total_features=5  # B0(2) + B1(3)
)

# Print summary
all_passed = validator.print_summary()

# Save report
validator.save_report("validation_report.txt")
```

## Key Features

1. **Comprehensive Checks**: Every tensor, shape, dtype, and value is validated
2. **NaN/Inf Detection**: All tensors checked for invalid values
3. **Shape Validation**: Strict shape checking at every stage
4. **Range Validation**: Value ranges checked where applicable
5. **Gradient Monitoring**: Gradient norms and NaN/Inf detection
6. **Detailed Reporting**: Clear PASS/FAIL status with diagnostics

## Notes

- Uses CPU by default for safety (GPU optional for model checks)
- Checks first 10 videos in files for performance
- All checks are non-destructive (read-only)
- Can be run at any stage of the pipeline
- Exit code 0 = all passed, 1 = failures detected

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the project root directory
2. **File not found**: Check paths are correct and files exist
3. **CUDA errors**: Use `--device cpu` if GPU unavailable
4. **Memory errors**: Reduce batch_size in validation

### Getting Help

See `validation/README.md` for detailed documentation.

