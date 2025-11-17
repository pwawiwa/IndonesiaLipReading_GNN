# Overfitting Fixes - Recommendations

## Current Status
- **Train Acc**: 51.5%
- **Val Acc**: 22.2%
- **Gap**: 29% (SEVERE OVERFITTING)

## Recommended Fixes (Priority Order)

### 1. Further Reduce Model Complexity ⭐⭐⭐
**Current**: 256 dims, 3 spatial layers
**Recommendation**: 
- Reduce to 128 dims (50% reduction)
- Reduce to 2 spatial layers
- This will cut parameters by ~75%

### 2. Increase Regularization ⭐⭐⭐
**Current**: dropout=0.5, weight_decay=2e-4, label_smoothing=0.1
**Recommendation**:
- Dropout: 0.5 → 0.7 (40% increase)
- Weight decay: 2e-4 → 1e-3 (5x increase)
- Label smoothing: 0.1 → 0.2 (2x increase)

### 3. Reduce Learning Rate ⭐⭐
**Current**: lr=1e-3
**Recommendation**: lr=5e-4 (50% reduction)
- Slower learning = better generalization
- Use learning rate finder or warmup

### 4. Add Data Augmentation ⭐⭐
**Current**: None
**Recommendation**: Add temporal augmentation
- Random temporal masking (mask some frames)
- Temporal jittering (slight frame shifts)
- Feature noise injection (small Gaussian noise)

### 5. Adjust Training Strategy ⭐
**Recommendation**:
- Reduce batch size: 64 → 32 (more gradient updates)
- More aggressive early stopping: 50 → 30 epochs
- Use validation loss for early stopping (not accuracy)

### 6. Model Architecture Changes ⭐
**Recommendation**:
- Add batch normalization in more places
- Use layer normalization instead of batch norm
- Add residual connections with dropout

## Implementation Priority

### Immediate (High Impact, Easy)
1. ✅ Reduce model dims: 256 → 128
2. ✅ Increase dropout: 0.5 → 0.7
3. ✅ Increase weight decay: 2e-4 → 1e-3
4. ✅ Reduce learning rate: 1e-3 → 5e-4

### Short-term (Medium Impact)
5. Add temporal augmentation
6. Reduce batch size: 64 → 32
7. More aggressive early stopping

### Long-term (If still overfitting)
8. Further reduce model (64 dims)
9. Add mixup augmentation
10. Ensemble smaller models

## Expected Results

After applying immediate fixes:
- **Target**: Reduce gap from 29% to <15%
- **Val Acc**: Should improve to 30-35%
- **Train Acc**: May drop to 40-45% (acceptable)

