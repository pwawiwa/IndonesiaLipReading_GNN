"""
Utility functions for loading models from checkpoints.
"""
import torch
import inspect
from typing import Dict, Any
from models import get_model


def filter_model_config(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter model config to only include valid parameters for model initialization.
    
    Args:
        model_name: Name of the model ('gcn', 'gat', etc.)
        model_config: Config dict from checkpoint (may include invalid keys like 'model_class')
        
    Returns:
        Filtered config dict with only valid parameters
    """
    # Get the model class
    model_map = {
        'gcn': 'GCNModel',
        'gat': 'GATModel',
        'graphsage': 'GraphSAGEModel',
        'gin': 'GINModel',
        'adaptive_gcn': 'AdaptiveGCNModel',
        'adaptive_gcn_lstm': 'AdaptiveGCNLSTMModel',
        'stgcn': 'STGCNModel',
        'gconvlstm': 'GConvLSTMModel',
        'gnn_lstm': 'GNNLSTMModel',
        'gnn_gru': 'GNNGRUModel',
        'gin_lstm': 'GINLSTMModel',
        'gin_gru': 'GINGRUModel',
        'graphsage_gru': 'GraphSAGEGRUModel',
        'graphsage_lstm': 'GraphSAGELSTMModel',
        'gnn_temporal_conv': 'GNNTemporalConvModel',
        'graphwavenet': 'GraphWaveNetModel',
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Import the model class
    if model_name.lower() == 'gcn':
        from models.gcn import GCNModel
        model_class = GCNModel
    elif model_name.lower() == 'gat':
        from models.gat import GATModel
        model_class = GATModel
    elif model_name.lower() == 'graphsage':
        from models.graphsage import GraphSAGEModel
        model_class = GraphSAGEModel
    elif model_name.lower() == 'gin':
        from models.gin import GINModel
        model_class = GINModel
    elif model_name.lower() == 'adaptive_gcn':
        from models.adaptive_gcn import AdaptiveGCNModel
        model_class = AdaptiveGCNModel
    elif model_name.lower() == 'adaptive_gcn_lstm':
        from models.adaptive_gcn_lstm import AdaptiveGCNLSTMModel
        model_class = AdaptiveGCNLSTMModel
    elif model_name.lower() == 'stgcn':
        from models.stgcn import STGCNModel
        model_class = STGCNModel
    elif model_name.lower() == 'gconvlstm':
        from models.gconvlstm import GConvLSTMModel
        model_class = GConvLSTMModel
    elif model_name.lower() == 'gnn_lstm':
        from models.gnn_lstm import GNNLSTMModel
        model_class = GNNLSTMModel
    elif model_name.lower() == 'gnn_gru':
        from models.gnn_lstm import GNNGRUModel
        model_class = GNNGRUModel
    elif model_name.lower() == 'gin_lstm':
        from models.gin_lstm import GINLSTMModel
        model_class = GINLSTMModel
    elif model_name.lower() == 'gin_gru':
        from models.gin_gru import GINGRUModel
        model_class = GINGRUModel
    elif model_name.lower() == 'graphsage_gru':
        from models.graphsage_gru import GraphSAGEGRUModel
        model_class = GraphSAGEGRUModel
    elif model_name.lower() == 'graphsage_lstm':
        from models.graphsage_lstm import GraphSAGELSTMModel
        model_class = GraphSAGELSTMModel
    elif model_name.lower() == 'gnn_temporal_conv':
        from models.gnn_temporal_conv import GNNTemporalConvModel
        model_class = GNNTemporalConvModel
    elif model_name.lower() == 'graphwavenet':
        from models.graphwavenet import GraphWaveNetModel
        model_class = GraphWaveNetModel
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get valid parameter names from __init__ signature
    sig = inspect.signature(model_class.__init__)
    valid_param_names = set(sig.parameters.keys()) - {'self'}
    
    # Filter config to only include valid parameters
    filtered_config = {}
    for key, value in model_config.items():
        if key in valid_param_names:
            filtered_config[key] = value
    
    return filtered_config


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        model_name: Name of the model ('gcn', 'gat', etc.)
        device: Device to load model on
        
    Returns:
        Loaded model with state dict applied
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint['model_config'].copy()
    
    # Filter to only valid parameters
    filtered_config = filter_model_config(model_name, model_config)
    
    # Create model
    model = get_model(model_name, **filtered_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

