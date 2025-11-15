# Checkpoints Summary

This document contains a comprehensive summary of all training checkpoints in `outputs/combined/checkpoints/`.

## Model Architecture (Consistent Across All Checkpoints)

All checkpoints use the same model architecture:
- **Spatial Dimension**: 512
- **Temporal Dimension**: 512
- **LSTM Layers**: 3 (bidirectional)
- **Number of Classes**: 100
- **Total Parameters**: 33,219,694

## Training Configuration

- **Initial Learning Rate**: 0.001
- **Weight Decay**: 0.0001 (1e-4)
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Training Progress**: Up to epoch 800

## Checkpoint Summary Table

| Epoch | Best Val Acc | Current Val Acc | Train Loss | Learning Rate | Notes |
|-------|--------------|-----------------|------------|---------------|-------|
| 1     | 0.089304     | 0.089304        | 3.864742   | 0.00100000    | Initial |
| 2     | 0.098705     | 0.098705        | 3.800580   | 0.00099999    | |
| 3     | 0.102778     | 0.102778        | 3.781655   | 0.00099998    | |
| 4     | 0.107374     | 0.107374        | 3.766921   | 0.00099996    | |
| 7     | 0.085858     | 0.085858        | 4.605722   | 0.00099988    | Performance drop |
| 18    | 0.109881     | 0.109881        | 3.541063   | 0.00099921    | Recovery |
| **33**| **0.112179** | **0.112179**    | **3.279792**| **0.00099734** | **🏆 BEST** |
| 100   | 0.112179     | 0.096407        | 2.478594   | 0.00097577    | Val acc decreased |
| 200   | 0.112179     | 0.090244        | 1.394623   | 0.00090546    | Overfitting |
| 300   | 0.112179     | 0.091498        | 0.733861   | 0.00079595    | |
| 400   | 0.112179     | 0.096825        | 0.408226   | 0.00065796    | |
| 500   | 0.112179     | 0.095885        | 0.284559   | 0.00050500    | |
| 600   | 0.112179     | 0.074055        | N/A        | 0.00035204    | ⚠️ Training loss NaN |
| 700   | 0.112179     | 0.074055        | N/A        | 0.00021405    | ⚠️ Training loss NaN |
| 800   | 0.112179     | 0.074055        | N/A        | 0.00010454    | ⚠️ Training loss NaN |

## Key Observations

### 🏆 Best Performance
- **Best Checkpoint**: Epoch 33 (`checkpoint_epoch_0033.pth`)
- **Best Validation Accuracy**: 11.22% (0.112179)
- **Best Validation F1**: 6.48% (0.064831)
- **Training Loss at Best**: 3.28

### 📈 Training Progress
1. **Early Training (Epochs 1-33)**:
   - Validation accuracy improved from 8.93% to 11.22%
   - Training loss decreased from 3.86 to 3.28
   - Best performance achieved at epoch 33

2. **Mid Training (Epochs 100-500)**:
   - Training loss continued to decrease (2.48 → 0.28)
   - Validation accuracy decreased (9.64% → 9.59%)
   - Clear signs of overfitting

3. **Late Training (Epochs 600-800)**:
   - Training loss became NaN (numerical instability)
   - Validation accuracy dropped to 7.41%
   - Learning rate decayed to very low values (0.0001)

### ⚠️ Training Issues
- **Numerical Instability**: Training loss became NaN starting from epoch 600
- **Overfitting**: Validation accuracy peaked at epoch 33 and then decreased
- **Learning Rate Decay**: LR decayed from 0.001 to 0.0001 by epoch 800

## Checkpoint Files

### Regular Checkpoints
- `checkpoint_epoch_0001.pth` - Epoch 1
- `checkpoint_epoch_0002.pth` - Epoch 2
- `checkpoint_epoch_0003.pth` - Epoch 3
- `checkpoint_epoch_0004.pth` - Epoch 4
- `checkpoint_epoch_0007.pth` - Epoch 7
- `checkpoint_epoch_0018.pth` - Epoch 18
- `checkpoint_epoch_0033.pth` - Epoch 33 ⭐ **BEST**
- `checkpoint_epoch_0100.pth` - Epoch 100
- `checkpoint_epoch_0200.pth` - Epoch 200
- `checkpoint_epoch_0300.pth` - Epoch 300
- `checkpoint_epoch_0400.pth` - Epoch 400
- `checkpoint_epoch_0500.pth` - Epoch 500
- `checkpoint_epoch_0600.pth` - Epoch 600
- `checkpoint_epoch_0700.pth` - Epoch 700
- `checkpoint_epoch_0800.pth` - Epoch 800

### Special Checkpoints
- `best_checkpoint.pth` - Copy of epoch 33 (best performance)
- `latest_checkpoint.pth` - Copy of epoch 800 (most recent)

## Recommendations

1. **Use Best Checkpoint**: The checkpoint at epoch 33 (`checkpoint_epoch_0033.pth` or `best_checkpoint.pth`) should be used for inference as it has the best validation performance.

2. **Early Stopping**: Training should have been stopped around epoch 33-100 when validation accuracy started decreasing.

3. **Learning Rate**: Consider using a different learning rate schedule or stopping training earlier to prevent numerical instability.

4. **Regularization**: The model shows clear overfitting. Consider:
   - Increasing dropout
   - Adding more regularization
   - Using data augmentation
   - Reducing model capacity

## File Sizes

All checkpoint files are approximately **381 MB** each (except epoch 7 which is 322 MB), containing:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training history
- Best metrics

