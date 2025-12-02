"""
GNN-TemporalConv1D hybrid model.
This model uses GNN layers for spatial feature extraction, followed by
1D temporal convolutions to capture temporal patterns in lip movements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import GCNConv, TemporalConv


class GNNTemporalConvModel(BaseGNNModel):
    """
    GNN-TemporalConv1D hybrid model.
    
    This hybrid architecture combines GNN layers for spatial feature learning
    with 1D temporal convolutions for temporal pattern recognition. Unlike
    recurrent models (LSTM/GRU), temporal convolutions can process sequences
    in parallel and capture long-range dependencies through dilated convolutions,
    making them efficient and effective for lip reading tasks.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_gnn_layers: int = 2,
        num_temporal_layers: int = 2,
        temporal_kernel_size: int = 3,
        dropout: float = 0.5,
        temporal_pool: str = 'max'
    ):
        """
        Initialize GNN-TemporalConv1D model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_gnn_layers: Number of GNN layers for spatial processing
            num_temporal_layers: Number of temporal convolution layers
            temporal_kernel_size: Kernel size for temporal convolutions
            dropout: Dropout rate
            temporal_pool: Temporal pooling method ('mean', 'max', 'last')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_gnn_layers = num_gnn_layers
        self.num_temporal_layers = num_temporal_layers
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_pool = temporal_pool
        
        # GNN layers (spatial)
        self.gnn_convs = nn.ModuleList()
        self.gnn_convs.append(GCNConv(in_features, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch norm for GNN layers
        self.gnn_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList()
        for i in range(num_temporal_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.temporal_convs.append(
                nn.Sequential(
                    TemporalConv(in_channels, hidden_dim, temporal_kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        speech_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (batch, frames, nodes, features)
            adj: Adjacency matrix of shape (nodes, nodes)
            speech_mask: Optional speech mask of shape (batch, frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, num_frames, num_nodes, _ = x.shape
        
        # Spatial: Process each frame with GNN
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply GNN layers
        for i, conv in enumerate(self.gnn_convs):
            x = conv(x, adj)
            
            # Batch norm
            x_bn = x.reshape(-1, self.hidden_dim)
            x_bn = self.gnn_bns[i](x_bn)
            x = x_bn.reshape(batch_size * num_frames, num_nodes, self.hidden_dim)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Spatial pooling (aggregate nodes) - using max to preserve discriminative signals
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        x = x.max(dim=2)[0]  # (batch, frames, hidden_dim)
        
        # Temporal: Apply 1D convolutions
        for temporal_conv in self.temporal_convs:
            x = temporal_conv(x)  # (batch, frames, hidden_dim)
        
        # Use speech mask as attention/guidance (not masking)
        if speech_mask is not None:
            # speech_mask: (batch, frames)
            # Normalize to attention weights (soft attention)
            attention_weights = torch.softmax(speech_mask * 10.0, dim=1)  # Scale for sharper attention
            attention_weights = attention_weights.unsqueeze(-1)  # (batch, frames, 1)
        else:
            attention_weights = None
        
        # Temporal pooling
        if self.temporal_pool == 'mean':
            if speech_mask is not None and attention_weights is not None:
                # Weighted mean by attention
                weighted_sum = (x * attention_weights).sum(dim=1)  # (batch, hidden_dim)
                weights_sum = attention_weights.sum(dim=1).squeeze(-1)  # (batch,)
                x = weighted_sum / (weights_sum.unsqueeze(-1) + 1e-6)  # (batch, hidden_dim)
            else:
                x = x.mean(dim=1)
        elif self.temporal_pool == 'max':
            x = x.max(dim=1)[0]
        elif self.temporal_pool == 'last':
            x = x[:, -1, :]
        else:
            raise ValueError(f"Unknown temporal pooling: {self.temporal_pool}")
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_gnn_layers': self.num_gnn_layers,
            'num_temporal_layers': self.num_temporal_layers,
            'temporal_kernel_size': self.temporal_kernel_size,
            'temporal_pool': self.temporal_pool,
        })
        return config

