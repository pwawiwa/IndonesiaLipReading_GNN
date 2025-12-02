"""
Graph Isomorphism Network (GIN) model.
GIN is a powerful GNN architecture that can distinguish non-isomorphic graphs,
making it suitable for learning expressive node representations in lip reading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import GCNConv


class GINConv(nn.Module):
    """Graph Isomorphism Network layer with MLP."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 0.0,
        train_eps: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=train_eps)
        
        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (..., nodes, in_features)
            adj: Adjacency matrix of shape (nodes, nodes)
            
        Returns:
            Output features of shape (..., nodes, out_features)
        """
        # Aggregate neighbor features
        # Add self-loops to adjacency
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Normalize adjacency (row normalization)
        deg = adj.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1  # Avoid division by zero
        norm_adj = adj / deg
        
        # Aggregate: (1 + eps) * x + AGG(neighbors)
        # First aggregate neighbors
        agg = torch.matmul(norm_adj, x)  # (..., nodes, in_features)
        
        # Combine with self features: (1 + eps) * x + agg
        out = (1 + self.eps) * x + agg
        
        # Apply MLP
        # Reshape for MLP: (..., nodes, in_features) -> (..., nodes * in_features) -> MLP -> reshape back
        *batch_dims, nodes, features = out.shape
        out_flat = out.reshape(-1, features)  # (batch * nodes, features)
        out_flat = self.mlp(out_flat)  # (batch * nodes, out_features)
        out = out_flat.reshape(*batch_dims, nodes, self.out_features)
        
        return out


class GINModel(BaseGNNModel):
    """
    Graph Isomorphism Network (GIN) model for spatial graph learning.
    
    GIN is theoretically powerful and can distinguish non-isomorphic graphs,
    making it suitable for learning expressive spatial representations of facial
    landmarks in lip reading tasks. It uses MLPs to transform aggregated features,
    allowing it to capture complex spatial relationships.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        eps: float = 0.0,
        train_eps: bool = False,
        temporal_pool: str = 'max'
    ):
        """
        Initialize GIN model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GIN layers
            dropout: Dropout rate
            eps: Initial epsilon value for GIN aggregation
            train_eps: Whether to make epsilon trainable
            temporal_pool: Temporal pooling method ('mean', 'max', 'last')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_layers = num_layers
        self.temporal_pool = temporal_pool
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(in_features, hidden_dim, eps, train_eps))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim, eps, train_eps))
        
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
        
        # Process each frame independently through GIN layers
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply GIN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Reshape back to (batch, frames, nodes, hidden_dim)
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        
        # Spatial pooling (aggregate over nodes) - using max to preserve discriminative signals
        x = x.max(dim=2)[0]  # (batch, frames, hidden_dim)
        
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
            'temporal_pool': self.temporal_pool,
            'eps': float(self.convs[0].eps.item()) if len(self.convs) > 0 else 0.0,
            'train_eps': self.convs[0].eps.requires_grad if len(self.convs) > 0 else False,
        })
        return config

