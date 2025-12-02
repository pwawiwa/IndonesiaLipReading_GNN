"""
Graph Attention Network (GAT) model - Memory-efficient implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel


class GATConv(nn.Module):
    """Memory-efficient Graph Attention Network layer using sparse attention."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.6,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # Attention mechanism: single parameter vector per head
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse attention computation.
        
        Args:
            x: Node features of shape (batch, nodes, in_features)
            adj: Adjacency matrix of shape (nodes, nodes) - sparse (only edges)
            
        Returns:
            Output features of shape (batch, nodes, out_features * num_heads) if concat else (batch, nodes, out_features)
        """
        batch_size, N, _ = x.shape
        
        # Transform features: (batch, nodes, in_features) -> (batch, nodes, heads, out_features)
        h = torch.einsum('bni,hio->bnho', x, self.W)  # More memory efficient than matmul
        
        # Get edge indices from adjacency (only compute attention for existing edges)
        edge_index = adj.nonzero(as_tuple=False)  # (num_edges, 2)
        num_edges = edge_index.size(0)
        
        if num_edges == 0:
            # No edges, return zero output
            if self.concat:
                return torch.zeros(batch_size, N, self.num_heads * self.out_features, device=x.device, dtype=x.dtype)
            else:
                return torch.zeros(batch_size, N, self.out_features, device=x.device, dtype=x.dtype)
        
        # Extract source and target node indices
        src, dst = edge_index[:, 0], edge_index[:, 1]  # (num_edges,)
        
        # Compute attention scores only for edges (sparse computation)
        # h_src: (batch, num_edges, heads, out_features)
        # h_dst: (batch, num_edges, heads, out_features)
        h_src = h[:, src, :, :]  # (batch, num_edges, heads, out_features)
        h_dst = h[:, dst, :, :]  # (batch, num_edges, heads, out_features)
        
        # Concatenate source and target features: (batch, num_edges, heads, 2*out_features)
        h_cat = torch.cat([h_src, h_dst], dim=-1)
        
        # Compute attention scores: (batch, num_edges, heads)
        # a: (heads, 2*out_features) -> einsum with h_cat
        e = torch.einsum('beho,ho->beh', h_cat, self.a)  # (batch, num_edges, heads)
        e = self.leakyrelu(e)
        
        # Apply dropout to attention scores
        e = self.dropout(e)
        
        # Create attention matrix: need full N×N for softmax
        # Initialize with -inf for non-edges
        attention = torch.full((batch_size, self.num_heads, N, N), 
                              float('-inf'), device=x.device, dtype=x.dtype)
        
        # Fill in attention scores for edges
        # e: (batch, num_edges, heads) -> transpose to (batch, heads, num_edges)
        e_transposed = e.transpose(1, 2)  # (batch, heads, num_edges)
        
        # Fill attention scores using simple indexing
        # This is efficient because we only set values for existing edges
        attention[:, :, src, dst] = e_transposed
        
        # Softmax over target nodes (dim=-1)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to features
        # h: (batch, nodes, heads, out_features) -> (batch, heads, nodes, out_features)
        h = h.transpose(1, 2)  # (batch, heads, nodes, out_features)
        
        # Attention aggregation: (batch, heads, nodes, nodes) @ (batch, heads, nodes, out_features)
        h_prime = torch.einsum('bhij,bhjo->bhio', attention, h)  # (batch, heads, nodes, out_features)
        
        # Concatenate or average heads
        if self.concat:
            # Concatenate: (batch, heads, nodes, out_features) -> (batch, nodes, heads*out_features)
            output = h_prime.transpose(1, 2).reshape(batch_size, N, self.num_heads * self.out_features)
        else:
            # Average: (batch, heads, nodes, out_features) -> (batch, nodes, out_features)
            output = h_prime.mean(dim=1)  # (batch, nodes, out_features)
        
        return output


class GATModel(BaseGNNModel):
    """Memory-efficient GAT model with frame-by-frame processing."""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.6,
        temporal_pool: str = 'max'
    ):
        """
        Initialize GAT model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension (per head)
            num_classes: Number of output classes
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            temporal_pool: Temporal pooling method ('mean', 'max', 'last')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.temporal_pool = temporal_pool
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(in_features, hidden_dim, num_heads, dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=True)
            )
        
        # Last layer: average heads instead of concatenate
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=False)
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
        Forward pass with memory-efficient frame-by-frame processing.
        
        Args:
            x: Node features of shape (batch, frames, nodes, features)
            adj: Adjacency matrix of shape (nodes, nodes)
            speech_mask: Optional speech mask of shape (batch, frames)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, num_frames, num_nodes, _ = x.shape
        
        # Process frames one at a time to minimize memory usage
        # This avoids creating large (batch*frames, heads, N, N) attention matrices
        frame_outputs = []
        
        for t in range(num_frames):
            frame_x = x[:, t, :, :]  # (batch, nodes, features)
            
            # Apply GAT layers
            for i, conv in enumerate(self.convs):
                frame_x = conv(frame_x, adj)
                if i < len(self.convs) - 1:
                    frame_x = F.elu(frame_x)
                    frame_x = self.dropout(frame_x)
            
            frame_outputs.append(frame_x)
            
            # Clear cache periodically to prevent memory buildup
            if (t + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Stack frames: (batch, frames, nodes, hidden_dim)
        x = torch.stack(frame_outputs, dim=1)
        
        # Spatial pooling: aggregate over nodes - using max to preserve discriminative signals
        x = x.max(dim=2)[0]  # (batch, frames, hidden_dim)
        
        # Temporal pooling with optional speech mask guidance
        if speech_mask is not None:
            # Use speech mask as attention weights (soft attention, not hard masking)
            attention_weights = torch.softmax(speech_mask * 10.0, dim=1)  # (batch, frames)
            attention_weights = attention_weights.unsqueeze(-1)  # (batch, frames, 1)
        else:
            attention_weights = None
        
        if self.temporal_pool == 'mean':
            if attention_weights is not None:
                # Weighted mean by attention
                weighted_sum = (x * attention_weights).sum(dim=1)  # (batch, hidden_dim)
                weights_sum = attention_weights.sum(dim=1).squeeze(-1)  # (batch,)
                x = weighted_sum / (weights_sum.unsqueeze(-1) + 1e-6)
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
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'temporal_pool': self.temporal_pool,
        })
        return config
