"""
Hybrid GIN-GRU model: GIN for spatial, GRU for temporal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .gin import GINConv


class GINGRUModel(BaseGNNModel):
    """Hybrid GIN-GRU model: GIN for spatial, GRU for temporal."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_gin_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
        eps: float = 0.0,
        train_eps: bool = False
    ):
        """
        Initialize GIN-GRU model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_gin_layers: Number of GIN layers
            num_gru_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Use bidirectional GRU
            eps: Initial epsilon value for GIN aggregation
            train_eps: Whether to make epsilon trainable
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_gin_layers = num_gin_layers
        self.num_gru_layers = num_gru_layers
        self.bidirectional = bidirectional
        
        # GIN layers (spatial)
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(in_features, hidden_dim, eps, train_eps))
        for _ in range(num_gin_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim, eps, train_eps))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # GRU (temporal)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(gru_output_dim, num_classes)
    
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
        
        # Spatial: Process each frame with GIN
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply GIN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Spatial pooling (aggregate nodes) - using max to preserve discriminative signals
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        x = x.max(dim=2)[0]  # (batch, frames, hidden_dim)
        
        # Learn from all frames (don't mask)
        # Temporal: GRU processes all frames
        gru_out, h_n = self.gru(x)  # gru_out: (batch, frames, hidden_dim)
        
        # Use speech mask as attention on GRU output (guidance, not masking)
        if speech_mask is not None:
            # speech_mask: (batch, frames)
            # Convert to attention weights
            attention_weights = torch.softmax(speech_mask * 10.0, dim=1)  # (batch, frames)
            attention_weights = attention_weights.unsqueeze(-1)  # (batch, frames, 1)
            
            # Weighted sum of GRU outputs (all frames contribute, speech frames weighted more)
            h_weighted = (gru_out * attention_weights).sum(dim=1) / (attention_weights.sum(dim=1) + 1e-6)
        else:
            # Use last hidden state if no speech mask
            if self.bidirectional:
                h_weighted = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                h_weighted = h_n[-1]
        
        # Classification
        logits = self.classifier(h_weighted)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_gin_layers': self.num_gin_layers,
            'num_gru_layers': self.num_gru_layers,
            'bidirectional': self.bidirectional,
            'eps': float(self.convs[0].eps.item()) if len(self.convs) > 0 else 0.0,
            'train_eps': self.convs[0].eps.requires_grad if len(self.convs) > 0 else False,
        })
        return config

