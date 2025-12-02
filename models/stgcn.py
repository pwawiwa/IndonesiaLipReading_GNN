"""
Spatial-Temporal Graph Convolutional Network (ST-GCN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import GCNConv, TemporalConv


class STGCNBlock(nn.Module):
    """ST-GCN block with spatial and temporal convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Spatial graph convolution
        self.gcn = GCNConv(in_channels, out_channels)
        
        # Temporal convolution
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                (padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = lambda x: x
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features of shape (batch, channels, frames, nodes)
            adj: Adjacency matrix of shape (nodes, nodes)
            
        Returns:
            Output of shape (batch, out_channels, frames, nodes)
        """
        batch_size, in_channels, num_frames, num_nodes = x.shape
        
        # Spatial GCN
        # Reshape to (batch * frames, nodes, channels)
        x_gcn = x.permute(0, 2, 3, 1).reshape(batch_size * num_frames, num_nodes, in_channels)
        x_gcn = self.gcn(x_gcn, adj)
        
        # Reshape back to (batch, channels, frames, nodes)
        x_gcn = x_gcn.reshape(batch_size, num_frames, num_nodes, -1).permute(0, 3, 1, 2)
        
        # Temporal convolution
        res = self.residual(x)
        x_out = self.tcn(x_gcn) + res
        
        return self.relu(x_out)


class STGCNModel(BaseGNNModel):
    """ST-GCN model for spatio-temporal graph learning."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int = 3,
        kernel_size: int = 9,
        dropout: float = 0.5,
        temporal_pool: str = 'max'
    ):
        """
        Initialize ST-GCN model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_blocks: Number of ST-GCN blocks
            kernel_size: Temporal kernel size
            dropout: Dropout rate
            temporal_pool: Temporal pooling method
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.temporal_pool = temporal_pool
        
        # ST-GCN blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(STGCNBlock(in_features, hidden_dim, kernel_size, 1, dropout))
        
        for _ in range(num_blocks - 1):
            self.blocks.append(STGCNBlock(hidden_dim, hidden_dim, kernel_size, 1, dropout))
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        batch_size, num_frames, num_nodes, in_features = x.shape
        
        # Reshape to (batch, features, frames, nodes) for Conv2d
        x = x.permute(0, 3, 1, 2)
        
        # Apply ST-GCN blocks
        for block in self.blocks:
            x = block(x, adj)
        
        # x: (batch, hidden_dim, frames, nodes)
        
        # Use speech mask as attention/guidance (not masking)
        # x shape: (batch, hidden_dim, frames, nodes)
        if speech_mask is not None:
            # speech_mask: (batch, frames)
            # Convert to attention weights
            attention_weights = torch.softmax(speech_mask * 10.0, dim=1)  # (batch, frames)
            attention_weights = attention_weights.unsqueeze(1).unsqueeze(-1)  # (batch, 1, frames, 1)
        else:
            attention_weights = None
        
        # Temporal pooling
        if self.temporal_pool == 'mean':
            # Average over time and space
            if speech_mask is not None:
                # Weighted average by attention (all frames contribute)
                x_weighted = x * attention_weights  # (batch, hidden_dim, frames, nodes)
                x_sum = x_weighted.sum(dim=2)  # (batch, hidden_dim, nodes)
                weight_sum = attention_weights.sum(dim=2).squeeze(-1)  # (batch, 1, 1)
                x = x_sum / (weight_sum.unsqueeze(-1) + 1e-6)
                x = x.max(dim=2)[0]  # (batch, hidden_dim) - using max to preserve discriminative signals
            else:
                x = x.max(dim=2)[0].max(dim=2)[0]  # (batch, hidden_dim) - using max to preserve discriminative signals
        elif self.temporal_pool == 'global':
            x = self.pool(x).squeeze(-1).squeeze(-1)
        else:
            # Max pooling
            x = x.max(dim=2)[0].max(dim=2)[0]
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_blocks': self.num_blocks,
            'kernel_size': self.kernel_size,
            'temporal_pool': self.temporal_pool,
        })
        return config

