"""
Adaptive Graph Convolutional Network (GCN) model.
Uses learnable adjacency matrix to discover task-specific graph structures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import AdaptiveGCNConv


class AdaptiveGCNModel(BaseGNNModel):
    """
    Adaptive GCN model that learns graph structure dynamically.
    
    Why Adaptive GCN improves lip reading:
    1. Fixed adjacency only captures anatomical connections (MediaPipe topology)
    2. Adaptive adjacency learns task-specific connections:
       - Long-range dependencies (e.g., lip corners ↔ jaw for certain visemes)
       - Co-articulation patterns (e.g., lip movement ↔ cheek movement)
       - Symmetry relationships (left-right facial landmark correlations)
    3. Different words/visemes may require different graph structures
    4. Can discover non-local relationships not in original topology
    5. More flexible than fixed topology, adapts to the specific lip reading task
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        alpha: float = 0.5,
        temporal_pool: str = 'mean'
    ):
        """
        Initialize Adaptive GCN model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_nodes: Number of nodes in the graph (e.g., 468 for full face)
            num_layers: Number of GCN layers
            dropout: Dropout rate
            alpha: Weight for combining fixed and adaptive adjacency (0=only adaptive, 1=only fixed)
            temporal_pool: Temporal pooling method ('mean', 'max', 'last')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.temporal_pool = temporal_pool
        
        # Adaptive GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(AdaptiveGCNConv(in_features, hidden_dim, num_nodes, alpha))
        for _ in range(num_layers - 1):
            self.convs.append(AdaptiveGCNConv(hidden_dim, hidden_dim, num_nodes, alpha))
        
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
            adj: Fixed adjacency matrix of shape (nodes, nodes) - used as initialization
            speech_mask: Optional speech mask of shape (batch, frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, num_frames, num_nodes, _ = x.shape
        
        # Process each frame independently through Adaptive GCN layers
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply Adaptive GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)  # Uses both fixed and adaptive adjacency
            
            # Reshape for batch norm: (batch * frames * nodes, hidden_dim)
            x_bn = x.reshape(-1, self.hidden_dim)
            x_bn = self.bns[i](x_bn)
            x = x_bn.reshape(batch_size * num_frames, num_nodes, self.hidden_dim)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Reshape back to (batch, frames, nodes, hidden_dim)
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        
        # Spatial pooling (aggregate over nodes) - using mean to capture all information
        x = x.mean(dim=2)  # (batch, frames, hidden_dim)
        
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
            'num_layers': self.num_layers,
            'num_nodes': self.num_nodes,
            'alpha': self.alpha,
            'temporal_pool': self.temporal_pool,
        })
        return config

