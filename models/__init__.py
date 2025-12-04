"""
GNN model architectures for lip reading.
"""
from .base_model import BaseGNNModel
from .gcn import GCNModel
from .gat import GATModel
from .graphsage import GraphSAGEModel
from .gin import GINModel
from .adaptive_gcn import AdaptiveGCNModel
from .adaptive_gcn_lstm import AdaptiveGCNLSTMModel
from .stgcn import STGCNModel
from .gconvlstm import GConvLSTMModel
from .gnn_lstm import GNNLSTMModel, GNNGRUModel
from .gin_lstm import GINLSTMModel
from .gin_gru import GINGRUModel
from .graphsage_gru import GraphSAGEGRUModel
from .graphsage_lstm import GraphSAGELSTMModel
from .gnn_temporal_conv import GNNTemporalConvModel
from .graphwavenet import GraphWaveNetModel
from .gin_lstm_mamba import GINLSTMMambaModel
from .gnn_lstm_mamba import GNNLSTMMambaModel
from .graphsage_lstm_mamba import GraphSAGELSTMMambaModel
from .adaptive_gcn_lstm_mamba import AdaptiveGCNLSTMMambaModel

__all__ = [
    'BaseGNNModel',
    'GCNModel',
    'GATModel',
    'GraphSAGEModel',
    'GINModel',
    'AdaptiveGCNModel',
    'AdaptiveGCNLSTMModel',
    'STGCNModel',
    'GConvLSTMModel',
    'GNNLSTMModel',
    'GNNGRUModel',
    'GINLSTMModel',
    'GINGRUModel',
    'GraphSAGEGRUModel',
    'GraphSAGELSTMModel',
    'GNNTemporalConvModel',
    'GraphWaveNetModel',
    'GINLSTMMambaModel',
    'GNNLSTMMambaModel',
    'GraphSAGELSTMMambaModel',
    'AdaptiveGCNLSTMMambaModel',
]


def get_model(model_name: str, **kwargs):
    """
    Get model by name.
    
    Args:
        model_name: Model name (e.g., 'gcn', 'gat', 'stgcn', 'gin', 'gconvlstm', 'gnn_temporal_conv', 'graphwavenet')
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    model_map = {
        'gcn': GCNModel,
        'gat': GATModel,
        'graphsage': GraphSAGEModel,
        'gin': GINModel,
        'adaptive_gcn': AdaptiveGCNModel,
        'adaptive_gcn_lstm': AdaptiveGCNLSTMModel,
        'stgcn': STGCNModel,
        'gconvlstm': GConvLSTMModel,
        'gnn_lstm': GNNLSTMModel,
        'gnn_gru': GNNGRUModel,
        'gin_lstm': GINLSTMModel,
        'gin_gru': GINGRUModel,
        'graphsage_gru': GraphSAGEGRUModel,
        'graphsage_lstm': GraphSAGELSTMModel,
        'gnn_temporal_conv': GNNTemporalConvModel,
        'graphwavenet': GraphWaveNetModel,
        'gin_lstm_mamba': GINLSTMMambaModel,
        'gnn_lstm_mamba': GNNLSTMMambaModel,
        'graphsage_lstm_mamba': GraphSAGELSTMMambaModel,
        'adaptive_gcn_lstm_mamba': AdaptiveGCNLSTMMambaModel,
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name.lower()](**kwargs)

