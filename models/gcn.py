"""
Graph Convolutional Network (GCN) model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import GCNConv


class GCNModel(BaseGNNModel):
    """GCN model for spatial graph learning."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        temporal_pool: str = 'max'
    ):
        """
        Initialize GCN model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
            temporal_pool: Temporal pooling method ('mean', 'max', 'last')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_layers = num_layers
        self.temporal_pool = temporal_pool
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch norm layers
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
        
        # Process each frame independently through GCN layers
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            
            # Reshape for batch norm: (batch * frames * nodes, hidden_dim)
            x_bn = x.reshape(-1, self.hidden_dim)
            x_bn = self.bns[i](x_bn)
            x = x_bn.reshape(batch_size * num_frames, num_nodes, self.hidden_dim)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Reshape back to (batch, frames, nodes, hidden_dim)
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        
        # Spatial pooling (aggregate over nodes) - using max to preserve discriminative signals
        x = x.max(dim=2)[0]  # (batch, frames, hidden_dim)
        
        # Use speech mask as attention/guidance (not masking)
        # Learn from all frames, but use speech_mask to guide attention
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
                # Weighted mean by attention (all frames contribute, but speech frames weighted more)
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
        else:
            raise ValueError(f"Unknown temporal pooling: {self.temporal_pool}")
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'temporal_pool': self.temporal_pool,
        })
        return config

