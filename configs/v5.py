"""
V5 Configuration - AST-GCN Model (Reduced Complexity)
Attention-based Spatial-Temporal Graph Convolutional Network
Reduced model complexity and advanced features disabled for memory efficiency
Updated for full face (468 landmarks) with MediaPipe FACEMESH_TESSELATION connections
"""
from pathlib import Path

config = {
    'data_dir': Path('data/processed_v3'),  # Full face (468 landmarks) with MediaPipe connections
    'batch_size': 32,
    'num_workers': 8,
    'lr': 5e-4,  # Slightly lower for attention mechanisms
    'num_epochs': 1000,
    
    # Architecture: AST-GCN with attention (reduced complexity)
    'model_type': 'ast_gcn',  # Use AST-GCN model
    'spatial_dim': 128,  # Reduced from 256 (50% reduction)
    'temporal_dim': 128,  # Reduced from 256 (50% reduction)
    'spatial_layers': 2,  # Reduced from 3 (33% reduction)
    'temporal_layers': 2,  # Keep at 2 for multi-scale temporal modeling
    'num_heads': 2,  # Reduced from 4 (50% reduction)
    'dropout': 0.5,
    
    # Disable advanced features (redundant with GCN+LSTM)
    'use_advanced_features': False,  # Disable Gabor, FFT, Recurrence, Multi-scale, Relative motion
    
    # Regularization
    'weight_decay': 5e-4,
    'label_smoothing': 0.15,
    
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'save_dir': 'outputs/v5',
    'checkpoint_interval': 100,
    
    # Early stopping
    'early_stopping_patience': 200,
    'early_stopping_min_delta': 0.0005,
    
    'enable_log_server': False,
    'log_server': {
        'host': '0.0.0.0',
        'port': 8080,
        'entries': 10,
        'refresh_minutes': 5,
    },
}

# Expected parameters: ~3-4M (attention mechanisms add parameters)
# Expected test accuracy: 25-30% (with advanced features)
# Expected train-val gap: <10%

