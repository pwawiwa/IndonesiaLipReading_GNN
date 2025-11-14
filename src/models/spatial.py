"""
src/models/spatial.py
Simple Spatial GCN for processing facial landmark graphs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class SpatialGCN(nn.Module):
    """Spatial GCN - processes graph structure"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
        self.output_dim = hidden_dim * 2  # mean + max pooling
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connections [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Apply GCN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool to graph level
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        return torch.cat([x_mean, x_max], dim=1)