"""
V5 Configuration - AST-GCN Model
Attention-based Spatial-Temporal Graph Convolutional Network
With advanced features: Gabor, Recurrence, FFT, Multi-scale, Relative motion
"""
from pathlib import Path

config = {
    'data_dir': Path('data/processed_v2'),  # Using existing processed data
    'batch_size': 32,
    'num_workers': 8,
    'lr': 5e-4,  # Slightly lower for attention mechanisms
    'num_epochs': 1000,
    
    # Architecture: AST-GCN with attention
    'model_type': 'ast_gcn',  # Use AST-GCN model
    'spatial_dim': 256,
    'temporal_dim': 256,
    'spatial_layers': 3,
    'temporal_layers': 2,
    'num_heads': 4,  # Attention heads for GAT
    'dropout': 0.5,
    
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

