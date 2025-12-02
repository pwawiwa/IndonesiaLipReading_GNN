"""
Base model class for all GNN models.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class BaseGNNModel(nn.Module):
    """Base class for all GNN models."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5
    ):
        """
        Initialize base model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
    
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
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'model_class': self.__class__.__name__,
            'in_features': self.in_features,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
        }
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

