# Debugging Tools Summary

## What Was Created

### 1. Facemesh Visualization (`src/debug/visualize_facemesh.py`)
- Visualize extracted landmarks from videos or processed .pt files
- Check extraction quality and ROI selection
- Generate 2D and 3D visualizations

### 2. Enhanced Evaluation (`src/debug/enhanced_evaluation.py`)
- Per-word accuracy metrics
- Detailed confusion matrix
- Error analysis (high confidence errors, low confidence correct)
- Visualizations of per-word performance

### 3. Model Explainability (`src/debug/model_explainability.py`)
- Gradient-based saliency maps
- Occlusion sensitivity analysis
- Visualization of what the model focuses on
- Temporal and spatial attention analysis

### 4. Alternative Models (`src/models/alternative_models.py`)
- **ModelV2_Attention**: Attention-based spatial and temporal processing
- **ModelV3_ConvTemporal**: 1D convolution for temporal processing instead of LSTM

### 5. Comprehensive Debugging (`src/debug/comprehensive_debug.py`)
- Runs all analysis tools in one go
- Generates comprehensive reports

### 6. Shell Script (`scripts/debug_model.sh`)
- Easy-to-use script to run all debugging tools
- Handles paths and parameters automatically

## Quick Usage

```bash
# Run comprehensive debugging
./scripts/debug_model.sh --checkpoint outputs/v4/best_model.pth

# Or with version
./scripts/debug_model.sh --version v4
```

## Output Structure

All results are saved to `debug_outputs/comprehensive/debug_session_TIMESTAMP/`:
- `facemesh_visualization/` - Landmark visualizations
- `evaluation/` - Detailed metrics and plots
- `explanations/` - Model explainability visualizations
- `debug_summary.json` - Summary of all results

## Key Features

1. **Per-word analysis**: See which words are problematic
2. **Confusion matrix**: Understand what words are confused
3. **Saliency maps**: See what the model focuses on
4. **Error analysis**: Identify high-confidence errors
5. **Alternative architectures**: Try different models if stuck

## Next Steps

1. Run debugging on your current model
2. Analyze results to identify issues
3. Try alternative models if needed
4. Improve feature extraction based on visualizations
5. Iterate based on findings

See `DEBUGGING_GUIDE.md` for detailed instructions.

