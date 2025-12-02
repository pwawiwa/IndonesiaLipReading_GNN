"""
Graph WaveNet model.
GraphWaveNet combines dilated temporal convolutions with graph convolutions
to capture both long-range temporal dependencies and spatial graph structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import GCNConv


class DilatedTemporalConv(nn.Module):
    """Dilated temporal convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding to maintain sequence length
        padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features of shape (batch, time, channels)
            
        Returns:
            Output of shape (batch, time, out_channels)
        """
        # Conv1d expects (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, time, out_channels)
        return x


class GraphWaveNetBlock(nn.Module):
    """GraphWaveNet block with dilated temporal conv and graph conv."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Dilated temporal convolution
        self.temporal_conv = DilatedTemporalConv(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout
        )
        
        # Graph convolution
        self.gcn = GCNConv(out_channels, out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features of shape (batch, time, nodes, channels)
            adj: Adjacency matrix of shape (nodes, nodes)
            
        Returns:
            Output of shape (batch, time, nodes, out_channels)
        """
        batch_size, num_frames, num_nodes, in_channels = x.shape
        
        # Temporal convolution
        # Reshape to (batch * nodes, time, channels) for per-node temporal conv
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, num_frames, in_channels)
        x_temporal = self.temporal_conv(x_reshaped)  # (batch * nodes, time, out_channels)
        x_temporal = x_temporal.reshape(batch_size, num_nodes, num_frames, -1)
        x_temporal = x_temporal.permute(0, 2, 1, 3)  # (batch, time, nodes, out_channels)
        
        # Graph convolution (apply to each frame)
        x_gcn = x_temporal.reshape(batch_size * num_frames, num_nodes, -1)
        x_gcn = self.gcn(x_gcn, adj)
        x_gcn = x_gcn.reshape(batch_size, num_frames, num_nodes, -1)
        
        # Residual connection
        x_res = self.residual(x)
        x_out = x_gcn + x_res
        
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)
        
        return x_out


class GraphWaveNetModel(BaseGNNModel):
    """
    Graph WaveNet model.
    
    GraphWaveNet combines dilated temporal convolutions with graph convolutions
    to capture both long-range temporal dependencies (via dilated convolutions)
    and spatial graph structure (via GCN). The dilated convolutions allow the
    model to capture patterns at multiple temporal scales without increasing
    the number of parameters significantly.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.5,
        temporal_pool: str = 'max'
    ):
        """
        Initialize GraphWaveNet model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_blocks: Number of GraphWaveNet blocks
            kernel_size: Temporal kernel size
            dropout: Dropout rate
            temporal_pool: Temporal pooling method ('mean', 'max', 'last')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.temporal_pool = temporal_pool
        
        # GraphWaveNet blocks with increasing dilation
        self.blocks = nn.ModuleList()
        dilation = 1
        for i in range(num_blocks):
            in_channels = in_features if i == 0 else hidden_dim
            self.blocks.append(
                GraphWaveNetBlock(
                    in_channels,
                    hidden_dim,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            dilation *= 2  # Exponentially increasing dilation: 1, 2, 4, 8, ...
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        
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
        
        # Apply GraphWaveNet blocks
        for block in self.blocks:
            x = block(x, adj)
        
        # Final projection
        x = self.final_proj(x)
        x = F.relu(x)
        
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
            'num_blocks': self.num_blocks,
            'kernel_size': self.kernel_size,
            'temporal_pool': self.temporal_pool,
        })
        return config

