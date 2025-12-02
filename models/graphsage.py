"""
GraphSAGE model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import SAGEConv


class GraphSAGEModel(BaseGNNModel):
    """GraphSAGE model with neighborhood sampling."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = 'mean',
        temporal_pool: str = 'max'
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of SAGE layers
            dropout: Dropout rate
            aggregator: Aggregation method ('mean', 'max', 'lstm')
            temporal_pool: Temporal pooling method
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.temporal_pool = temporal_pool
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_features, hidden_dim, aggregator))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggregator))
        
        # Batch norm
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
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
        
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply SAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            
            # Batch norm
            x_bn = x.reshape(-1, self.hidden_dim)
            x_bn = self.bns[i](x_bn)
            x = x_bn.reshape(batch_size * num_frames, num_nodes, self.hidden_dim)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Reshape back
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        
        # Spatial pooling - using max to preserve discriminative signals
        x = x.max(dim=2)[0]
        
        # Use speech mask as attention/guidance (not masking)
        if speech_mask is not None:
            attention_weights = torch.softmax(speech_mask * 10.0, dim=1)
            attention_weights = attention_weights.unsqueeze(-1)  # (batch, frames, 1)
        else:
            attention_weights = None
        
        # Temporal pooling
        if self.temporal_pool == 'mean':
            if speech_mask is not None and attention_weights is not None:
                # Weighted mean by attention (all frames contribute)
                # Ensure proper broadcasting: (batch, frames, hidden_dim) * (batch, frames, 1)
                weighted_sum = (x * attention_weights).sum(dim=1)  # (batch, hidden_dim)
                weights_sum = attention_weights.sum(dim=1).squeeze(-1)  # (batch,)
                x = weighted_sum / (weights_sum.unsqueeze(-1) + 1e-6)  # (batch, hidden_dim)
            else:
                x = x.mean(dim=1)
        elif self.temporal_pool == 'max':
            x = x.max(dim=1)[0]
        elif self.temporal_pool == 'last':
            x = x[:, -1, :]
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'aggregator': self.aggregator,
            'temporal_pool': self.temporal_pool,
        })
        return config

