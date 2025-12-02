"""
Custom GNN layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GCNConv(nn.Module):
    """Graph Convolutional Network layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (..., nodes, in_features)
            adj: Adjacency matrix of shape (nodes, nodes)
            
        Returns:
            Output features of shape (..., nodes, out_features)
        """
        # Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
        adj = adj + torch.eye(adj.size(0), device=adj.device)  # Add self-loops
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        # Apply graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(norm, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class AdaptiveGCNConv(nn.Module):
    """
    Adaptive Graph Convolutional Network layer.
    
    Learns an adaptive adjacency matrix in addition to the fixed adjacency.
    This allows the model to discover task-specific graph structures that
    may not be present in the original topology.
    
    Why this improves lip reading:
    1. Fixed adjacency only captures anatomical connections (MediaPipe topology)
    2. Adaptive adjacency can learn task-specific connections (e.g., long-range
       dependencies between lip corners and jaw for certain visemes)
    3. Different words/visemes may require different graph structures
    4. Can discover non-local relationships (e.g., symmetry, co-articulation patterns)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        alpha: float = 0.5,
        bias: bool = True,
        init_adj: Optional[torch.Tensor] = None
    ):
        """
        Initialize Adaptive GCN layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_nodes: Number of nodes in the graph
            alpha: Weight for combining fixed and adaptive adjacency (0=only adaptive, 1=only fixed)
            bias: Whether to use bias
            init_adj: Initial adjacency matrix (optional, for warm start)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        
        # Feature transformation weight
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Learnable adaptive adjacency matrix
        # Initialize as identity + small random noise to encourage learning
        if init_adj is not None:
            # Use provided adjacency as initialization
            self.adaptive_adj = nn.Parameter(init_adj.clone().float())
        else:
            # Initialize as identity matrix (self-connections) with small random values
            self.adaptive_adj = nn.Parameter(
                torch.eye(num_nodes, dtype=torch.float32) + 
                0.01 * torch.randn(num_nodes, num_nodes)
            )
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        # Ensure adaptive adjacency is non-negative (use ReLU in forward)
        # Initialize to encourage learning while starting close to identity
        with torch.no_grad():
            self.adaptive_adj.data = torch.clamp(self.adaptive_adj.data, min=0.0)
    
    def forward(self, x: torch.Tensor, fixed_adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive adjacency.
        
        Args:
            x: Node features of shape (..., nodes, in_features)
            fixed_adj: Fixed adjacency matrix of shape (nodes, nodes)
            
        Returns:
            Output features of shape (..., nodes, out_features)
        """
        # Get adaptive adjacency (ensure non-negative)
        adaptive_adj = F.relu(self.adaptive_adj)
        
        # Normalize adaptive adjacency
        adaptive_adj = adaptive_adj + torch.eye(
            self.num_nodes, device=adaptive_adj.device
        ) * 0.1  # Add small self-connections
        deg_adaptive = adaptive_adj.sum(dim=1)
        deg_adaptive_inv_sqrt = torch.pow(deg_adaptive + 1e-6, -0.5)
        deg_adaptive_inv_sqrt[torch.isinf(deg_adaptive_inv_sqrt)] = 0.
        norm_adaptive = (
            deg_adaptive_inv_sqrt.unsqueeze(1) * adaptive_adj * 
            deg_adaptive_inv_sqrt.unsqueeze(0)
        )
        
        # Normalize fixed adjacency
        fixed_adj_norm = fixed_adj + torch.eye(
            fixed_adj.size(0), device=fixed_adj.device
        )
        deg_fixed = fixed_adj_norm.sum(dim=1)
        deg_fixed_inv_sqrt = torch.pow(deg_fixed + 1e-6, -0.5)
        deg_fixed_inv_sqrt[torch.isinf(deg_fixed_inv_sqrt)] = 0.
        norm_fixed = (
            deg_fixed_inv_sqrt.unsqueeze(1) * fixed_adj_norm * 
            deg_fixed_inv_sqrt.unsqueeze(0)
        )
        
        # Combine fixed and adaptive adjacency
        # alpha controls the trade-off: 0.5 means equal weight
        combined_adj = self.alpha * norm_fixed + (1 - self.alpha) * norm_adaptive
        
        # Apply graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(combined_adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GATConv(nn.Module):
    """Graph Attention Network layer."""
    
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
        
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (..., nodes, in_features)
            adj: Adjacency matrix of shape (nodes, nodes)
            
        Returns:
            Output features
        """
        # Get shape
        *batch_dims, N, _ = x.shape
        
        # Flatten batch dimensions
        x_flat = x.reshape(-1, N, self.in_features)
        batch_size = x_flat.size(0)
        
        # Multi-head attention
        h = torch.matmul(x_flat.unsqueeze(1), self.W)  # (batch, heads, N, out_features)
        
        # Memory-efficient attention computation using einsum
        # Instead of materializing full N×N×out_features tensors, compute scores directly
        # For each head: e[i,j] = LeakyReLU(a^T [Wh_i || Wh_j])
        # Split attention vector: a = [a_left, a_right] where each is (out_features,)
        
        # Split attention vector
        a_left = self.a[:, :self.out_features, :]  # (heads, out_features, 1)
        a_right = self.a[:, self.out_features:, :]  # (heads, out_features, 1)
        
        # Compute attention scores using einsum (avoids intermediate large tensors)
        # e[i,j] = LeakyReLU(a_left^T * h_i + a_right^T * h_j)
        # Using einsum: batch,heads,i,out @ heads,out -> batch,heads,i
        e_i = torch.einsum('bhio,ho->bhi', h, a_left.squeeze(-1))  # (batch, heads, N)
        e_j = torch.einsum('bhjo,ho->bhj', h, a_right.squeeze(-1))  # (batch, heads, N)
        
        # Broadcast and add: e[i,j] = e_i[i] + e_j[j]
        e = e_i.unsqueeze(-1) + e_j.unsqueeze(-2)  # (batch, heads, N, N)
        e = self.leakyrelu(e)  # (batch, heads, N, N)
        
        # Mask attention by adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0).unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention
        h_prime = torch.matmul(attention, h)  # (batch, heads, N, out_features)
        
        if self.concat:
            output = h_prime.reshape(batch_size, N, -1)
        else:
            output = h_prime.mean(dim=1)
        
        # Reshape back to original batch dimensions
        output = output.reshape(*batch_dims, N, -1)
        
        return output


class SAGEConv(nn.Module):
    """GraphSAGE layer."""
    
    def __init__(self, in_features: int, out_features: int, aggregator: str = 'mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()
    
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
        if self.aggregator == 'mean':
            # Compute degree
            deg = adj.sum(dim=1, keepdim=True)  # (nodes, 1)
            deg[deg == 0] = 1  # Avoid division by zero
            
            # Mean aggregation: matmul(adj, x) gives (..., nodes, features)
            # deg needs to broadcast: (nodes, 1) -> (1, nodes, 1) for (..., nodes, features)
            neigh_feat = torch.matmul(adj, x) / deg.unsqueeze(0)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        # Combine self and neighbor features
        self_feat = self.lin_self(x)
        neigh_feat = self.lin_neigh(neigh_feat)
        
        output = self_feat + neigh_feat
        
        return F.normalize(output, p=2, dim=-1)


class TemporalConv(nn.Module):
    """Temporal convolution layer (1D conv over time)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features of shape (batch, time, channels)
            
        Returns:
            Output of shape (batch, time, out_channels)
        """
        # Conv1d expects (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
