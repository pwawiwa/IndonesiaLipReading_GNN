# Model Comparison: V2 vs V3

## Results Summary

| Metric | V2 | V3 |
|--------|----|----|
| **Test Accuracy** | 22.73% | 15.86% |
| **Test F1 (Macro)** | 0.1082 | 0.0322 |
| **Test F1 (Weighted)** | 0.1974 | 0.1064 |
| **Parameters** | 3,453,796 | 858,212 |

## Model Parameters

| Parameter | V2 | V3 |
|-----------|----|----|
| Spatial Dim | 256 | 128 |
| Temporal Dim | 256 | 128 |
| Spatial Layers | 3 | 2 |
| Temporal Layers | 2 | 2 |
| Dropout | 0.5 | 0.7 |
| Weight Decay | 2e-4 | 1e-3 |
| Label Smoothing | 0.1 | 0.2 |
| Learning Rate | 1e-3 | 5e-4 |
| Batch Size | 64 | 32 |

## Preprocessing (Same for Both)

| Component | Value |
|-----------|-------|
| Landmarks | 62 ROI landmarks |
| Normalization | Face-relative [0, 1] |
| Action Units | 18 AUs per frame |
| Geometric Features | 10 features per frame |
| Node Features | 31 dims (3 + 18 + 10) |
| Edge Connections | k-NN (k=5) |
| Speech Mask | Extracted (used in V3, not in V2) |
| Temporal Smoothing | EMA (α=0.7) |

## Notes

- **V2**: Better accuracy but 4x larger, severe overfitting (29% gap)
- **V3**: Smaller model, stronger regularization, but lower accuracy
- Both use same preprocessing pipeline
- V3 uses speech mask in forward pass (V2 does not)
