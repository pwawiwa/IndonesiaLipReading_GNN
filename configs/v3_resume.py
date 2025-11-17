"""
V3 Resume Configuration
Continue training V3 with adjusted early stopping and slightly reduced regularization
"""
from pathlib import Path

config = {
    'data_dir': Path('data/processed'),
    'batch_size': 32,
    'num_workers': 8,
    'lr': 5e-4,  # Keep same as V3
    'num_epochs': 1000,
    
    # Architecture: Keep V3's simple architecture
    'spatial_dim': 128,
    'temporal_dim': 128,
    'spatial_layers': 2,
    'temporal_layers': 2,
    
    # Regularization: Slightly reduced from V3
    'dropout': 0.65,  # Reduced from 0.7
    'weight_decay': 7e-4,  # Reduced from 1e-3
    'label_smoothing': 0.15,  # Reduced from 0.2
    
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'save_dir': 'outputs/v3_resume',
    'checkpoint_interval': 100,
    
    # Early stopping: Much more patient
    'early_stopping_patience': 70,  # Increased from 30
    'early_stopping_min_delta': 0.0005,  # Smaller threshold
    
    'enable_log_server': False,
    'log_server': {
        'host': '0.0.0.0',
        'port': 8080,
        'entries': 10,
        'refresh_minutes': 5,
    },
    
    # Resume training from V3 checkpoint
    'resume': True,
    'resume_checkpoint': 'outputs/v3/checkpoints/checkpoint_epoch_0074.pth',  # Last epoch before early stop
}

# Expected: V3 might reach 18-20% if given more time
# This is a quick experiment to see if V3 was stopped too early

