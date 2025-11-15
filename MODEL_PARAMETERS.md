# Model Parameters Summary

This document contains the training parameters extracted from the saved model checkpoints.

## Best Model (`outputs/combined/best_model.pth`)

### Model Architecture
- **Spatial Module**: 
  - Hidden dimension: 512 (inferred from classifier input)
  - Number of GCN layers: 3 (default from train.py)
  
- **Temporal Module**:
  - LSTM hidden dimension: 512
  - LSTM input dimension: 1024 (from spatial output)
  - Number of LSTM layers: 3
  - Bidirectional: Yes (indicated by reverse weights)
  
- **Classifier**:
  - Input dimension: 2048 (from temporal fusion)
  - Hidden dimension: 1024
  - Number of classes: 100
  - Total parameters: 33,219,694

### Training Parameters
- **Epoch saved**: 33
- **Learning Rate**: 0.0009973422519357583 (decayed from initial)
- **Initial Learning Rate**: 0.001
- **Weight Decay**: 0.0001 (1e-4)
- **Optimizer**: AdamW
- **Betas**: (0.9, 0.999)

### Performance Metrics
- **Validation Accuracy**: 0.112179 (11.22%)
- **Validation F1 Score**: 0.064831 (6.48%)

---

## Final Model (`outputs/combined/final_model.pth`)

### Model Architecture
- **Spatial Module**: 
  - Hidden dimension: 256
  - Number of GCN layers: 3 (default from train.py)
  
- **Temporal Module**:
  - LSTM hidden dimension: 512
  - LSTM input dimension: 512 (from spatial output)
  - Number of LSTM layers: 2
  - Bidirectional: Yes (indicated by reverse weights)
  
- **Classifier**:
  - Input dimension: 1024 (from temporal fusion)
  - Hidden dimension: 512
  - Number of classes: 100
  - Total parameters: 11,223,143

### Training Parameters
- **Epoch saved**: 100 (final epoch)
- **Learning Rate**: Not saved in final model (only model state dict)
- **Initial Learning Rate**: 0.001 (from train.py default)
- **Weight Decay**: 0.0001 (1e-4) (from train.py default)

### Performance Metrics
- **Best Validation Accuracy**: 0.196783 (19.68%) at epoch 33
- **Final Validation Accuracy**: 0.173700 (17.37%)
- **Final Training Loss**: 2.414095
- **Training epochs completed**: 100

---

## Default Training Configuration (from `src/train.py`)

### Model Architecture Defaults
- **spatial_dim**: 256
- **temporal_dim**: 512
- **spatial_layers**: 3
- **temporal_layers**: 2
- **dropout**: 0.4

### Training Hyperparameters
- **batch_size**: 64
- **learning_rate**: 1e-3 (0.001)
- **num_epochs**: 100
- **weight_decay**: 1e-4 (0.0001)
- **optimizer**: AdamW
- **scheduler**: CosineAnnealingLR (T_max=num_epochs, eta_min=lr/100)
- **label_smoothing**: 0.0
- **gradient_clip_norm**: 1.0
- **mixed_precision**: Enabled (AMP) if CUDA available

### Data Configuration
- **data_dir**: `data/processed`
- **num_workers**: 8
- **checkpoint_interval**: 25

---

## Notes

1. **Different Architectures**: The `best_model.pth` and `final_model.pth` have different architectures:
   - Best model: spatial_dim=512, temporal_layers=3, 33M parameters
   - Final model: spatial_dim=256, temporal_layers=2, 11M parameters
   
   This suggests they may have been trained with different configurations or the best model was from a different training run.

2. **Best Model Performance**: The best model was saved at epoch 33 with validation accuracy of 11.22%, which is lower than the final model's best accuracy of 19.68%. This indicates the best model checkpoint might be from an earlier or different training run.

3. **Training History**: The final model contains complete training history showing:
   - Training loss decreased from 3.85 to 2.41
   - Validation accuracy peaked at 19.68% around epoch 33
   - Final validation accuracy was 17.37%

4. **No Config Files**: No separate configuration JSON files were found in the outputs directory, so parameters must be inferred from the checkpoints and the training script defaults.

