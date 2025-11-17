# Model Debugging and Explainability Guide

This guide explains how to use the debugging and explainability tools to analyze your lip reading model and improve its performance.

## Overview

The debugging system includes:
1. **Facemesh Visualization** - Inspect extraction quality
2. **Enhanced Evaluation** - Per-word accuracy and detailed metrics
3. **Model Explainability** - Understand what the model focuses on
4. **Alternative Models** - Try different architectures

## Quick Start

### Run Comprehensive Debugging

```bash
# Using the latest checkpoint
./scripts/debug_model.sh --checkpoint outputs/v4/best_model.pth

# Or specify a version (uses latest checkpoint in that version)
./scripts/debug_model.sh --version v4

# With custom parameters
./scripts/debug_model.sh \
    --checkpoint outputs/v4/best_model.pth \
    --num_facemesh_samples 20 \
    --num_explanation_samples 15
```

### Individual Tools

#### 1. Facemesh Visualization

Visualize extracted landmarks from processed data:

```bash
python -m src.debug.visualize_facemesh \
    --mode pt \
    --input data/processed_v2/test.pt \
    --output debug_outputs/facemesh_viz \
    --num_samples 10
```

Or visualize from a video file:

```bash
python -m src.debug.visualize_facemesh \
    --mode video \
    --input data/IDLRW-DATASET/ada/train/video.mp4 \
    --output debug_outputs/facemesh_viz \
    --num_frames 20
```

#### 2. Enhanced Evaluation

Get detailed per-word metrics and confusion matrix:

```bash
python -m src.debug.enhanced_evaluation \
    --checkpoint outputs/v4/best_model.pth \
    --test_pt data/processed_v2/test.pt \
    --output debug_outputs/evaluation \
    --input_dim 31
```

This generates:
- `evaluation_results.json` - All metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `per_word_accuracy.png` - Per-word accuracy bar chart

#### 3. Model Explainability

Understand what the model focuses on:

```bash
python -m src.debug.model_explainability \
    --checkpoint outputs/v4/best_model.pth \
    --test_pt data/processed_v2/test.pt \
    --output debug_outputs/explanations \
    --num_samples 10 \
    --input_dim 31
```

This generates:
- Saliency maps showing which landmarks/frames are important
- Heatmaps of model attention
- Prediction distributions

## Understanding the Results

### Facemesh Visualization

- **Check extraction quality**: Are landmarks correctly detected?
- **Verify ROI selection**: Are the right facial regions captured?
- **Inspect normalization**: Are landmarks properly normalized?

### Enhanced Evaluation

Key metrics to check:

1. **Overall Accuracy**: Current model performance
2. **Per-Word Accuracy**: Which words are hardest?
3. **Confusion Matrix**: What words are confused with each other?
4. **Error Analysis**: 
   - High confidence errors (model is wrong but confident)
   - Low confidence correct (model is right but uncertain)

**Action items based on results:**
- If certain words have low accuracy → Check if they have enough training samples
- If words are confused → They might be visually similar (consider data augmentation)
- If high confidence errors → Model might be overfitting

### Model Explainability

- **Temporal Saliency**: Which frames are most important?
- **Spatial Saliency**: Which landmarks are most important?
- **Prediction Distribution**: How confident is the model?

**Action items:**
- If model focuses on wrong frames → Check speech mask
- If model ignores important landmarks → Consider feature engineering
- If predictions are uncertain → Model might need more training or better features

## Alternative Models

If your current model is stuck at 18-25% accuracy, try alternative architectures:

### Model V2: Attention-based

```python
from models.alternative_models import ModelV2_Attention

model = ModelV2_Attention(
    input_dim=31,
    num_classes=num_classes,
    spatial_dim=256,
    temporal_dim=256,
    spatial_layers=3,
    temporal_layers=2,
    dropout=0.5
)
```

**Features:**
- Attention-based spatial pooling
- Attention-based temporal aggregation
- Better at focusing on important features

### Model V3: Convolutional Temporal

```python
from models.alternative_models import ModelV3_ConvTemporal

model = ModelV3_ConvTemporal(
    input_dim=31,
    num_classes=num_classes,
    spatial_dim=256,
    temporal_dim=256,
    spatial_layers=3,
    temporal_layers=3,
    dropout=0.5
)
```

**Features:**
- 1D convolution instead of LSTM for temporal processing
- Faster training
- Better at capturing local temporal patterns

## Debugging Workflow

1. **Run comprehensive debugging** to get baseline metrics
2. **Check facemesh visualization** - Is extraction quality good?
3. **Analyze per-word accuracy** - Which words are problematic?
4. **Examine confusion matrix** - What patterns do you see?
5. **Review model explanations** - What is the model focusing on?
6. **Try alternative models** if current architecture isn't working
7. **Iterate** based on findings

## Common Issues and Solutions

### Issue: Low overall accuracy (18-25%)

**Possible causes:**
1. **Poor feature extraction** - Check facemesh visualization
2. **Insufficient training data** - Check per-word sample counts
3. **Model architecture mismatch** - Try alternative models
4. **Overfitting** - Check train/val accuracy gap

**Solutions:**
- Improve facemesh extraction (check ROI indices)
- Add data augmentation
- Try alternative model architectures
- Increase regularization (dropout, weight decay)

### Issue: Some words have very low accuracy

**Possible causes:**
1. **Insufficient samples** - Check sample counts in evaluation
2. **Visual similarity** - Check confusion matrix
3. **Extraction issues** - Check facemesh for those words

**Solutions:**
- Collect more data for low-accuracy words
- Use data augmentation
- Check if those words have unique visual features

### Issue: High confidence errors

**Possible causes:**
1. **Overfitting** - Model memorized training data
2. **Data quality issues** - Mislabeled or poor quality samples

**Solutions:**
- Increase regularization
- Check data quality
- Use more diverse training data

## Output Structure

```
debug_outputs/
├── comprehensive/
│   └── debug_session_YYYYMMDD_HHMMSS/
│       ├── facemesh_visualization/
│       │   └── sample_*.png
│       ├── evaluation/
│       │   ├── evaluation_results.json
│       │   ├── confusion_matrix.png
│       │   └── per_word_accuracy.png
│       ├── explanations/
│       │   ├── sample_*_saliency.png
│       │   ├── sample_*_heatmap.png
│       │   ├── sample_*_predictions.png
│       │   └── explanation_summary.json
│       └── debug_summary.json
```

## Next Steps

1. Run comprehensive debugging on your current model
2. Analyze the results to identify issues
3. Try alternative model architectures
4. Improve feature extraction if needed
5. Iterate based on findings

For more details, see the individual tool documentation in `src/debug/`.

