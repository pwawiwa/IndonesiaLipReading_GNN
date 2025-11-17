"""
V4 Balanced Configuration
Sweet spot between V2 (overfitting) and V3 (underfitting)
"""
from pathlib import Path

config = {
    'data_dir': Path('data/processed'),
    'batch_size': 32,
    'num_workers': 8,
    'lr': 7.5e-4,  # Between V2 (1e-3) and V3 (5e-4)
    'num_epochs': 1000,
    
    # Architecture: Moderate complexity
    'spatial_dim': 192,  # Between V2 (256) and V3 (128) - 25% reduction from V2
    'temporal_dim': 192,  # Same
    'spatial_layers': 2,  # Keep at 2 (V3 was right here)
    'temporal_layers': 2,
    
    # Regularization: Moderate (not too aggressive)
    'dropout': 0.6,  # Between V2 (0.5) and V3 (0.7)
    'weight_decay': 5e-4,  # Between V2 (2e-4) and V3 (1e-3)
    'label_smoothing': 0.15,  # Between V2 (0.1) and V3 (0.2)
    
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'save_dir': 'outputs/v4',
    'checkpoint_interval': 100,
    
    # Early stopping: Very patient (allow model to fully converge)
    'early_stopping_patience': 200,  # Very patient - allow full convergence
    'early_stopping_min_delta': 0.0005,  # Smaller threshold (was 0.001)
    
    'enable_log_server': False,
    'log_server': {
        'host': '0.0.0.0',
        'port': 8080,
        'entries': 10,
        'refresh_minutes': 5,
    },
}

# Expected parameters: ~1.5-2M (between V2's 3.4M and V3's 858K)
# Expected test accuracy: 24-27%
# Expected train-val gap: <10%

