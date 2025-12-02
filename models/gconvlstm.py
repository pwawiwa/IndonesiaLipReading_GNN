"""
Graph Convolutional LSTM (GConvLSTM) model.
GConvLSTM combines graph convolutions with LSTM cells to jointly model
spatial and temporal dependencies in a unified recurrent architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import GCNConv


class GConvLSTMCell(nn.Module):
    """Graph Convolutional LSTM cell."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Graph convolutions for input, forget, output, and cell gates
        # Each gate uses GCN to aggregate spatial information
        self.gcn_ii = GCNConv(input_dim, hidden_dim, bias=False)
        self.gcn_if = GCNConv(input_dim, hidden_dim, bias=False)
        self.gcn_io = GCNConv(input_dim, hidden_dim, bias=False)
        self.gcn_ig = GCNConv(input_dim, hidden_dim, bias=False)
        
        self.gcn_hi = GCNConv(hidden_dim, hidden_dim, bias=False)
        self.gcn_hf = GCNConv(hidden_dim, hidden_dim, bias=False)
        self.gcn_ho = GCNConv(hidden_dim, hidden_dim, bias=False)
        self.gcn_hg = GCNConv(hidden_dim, hidden_dim, bias=False)
        
        # Bias terms
        self.bias_i = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_f = nn.Parameter(torch.ones(hidden_dim))  # Initialize forget gate bias to 1
        self.bias_o = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_g = nn.Parameter(torch.zeros(hidden_dim))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass of GConvLSTM cell.
        
        Args:
            x: Input features of shape (batch, nodes, input_dim)
            adj: Adjacency matrix of shape (nodes, nodes)
            h: Hidden state of shape (batch, nodes, hidden_dim)
            c: Cell state of shape (batch, nodes, hidden_dim)
            
        Returns:
            h_new: New hidden state (batch, nodes, hidden_dim)
            c_new: New cell state (batch, nodes, hidden_dim)
        """
        if h is None:
            h = torch.zeros(x.size(0), x.size(1), self.hidden_dim, device=x.device, dtype=x.dtype)
        if c is None:
            c = torch.zeros(x.size(0), x.size(1), self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Input gate
        i = torch.sigmoid(
            self.gcn_ii(x, adj) + self.gcn_hi(h, adj) + self.bias_i.unsqueeze(0).unsqueeze(0)
        )
        
        # Forget gate
        f = torch.sigmoid(
            self.gcn_if(x, adj) + self.gcn_hf(h, adj) + self.bias_f.unsqueeze(0).unsqueeze(0)
        )
        
        # Output gate
        o = torch.sigmoid(
            self.gcn_io(x, adj) + self.gcn_ho(h, adj) + self.bias_o.unsqueeze(0).unsqueeze(0)
        )
        
        # Cell gate
        g = torch.tanh(
            self.gcn_ig(x, adj) + self.gcn_hg(h, adj) + self.bias_g.unsqueeze(0).unsqueeze(0)
        )
        
        # Update cell state
        c_new = f * c + i * g
        c_new = self.dropout(c_new)
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class GConvLSTMModel(BaseGNNModel):
    """
    Graph Convolutional LSTM (GConvLSTM) model.
    
    GConvLSTM is a temporal GNN that integrates graph convolutions directly
    into LSTM cells, allowing the model to jointly learn spatial (via graph
    convolutions) and temporal (via LSTM recurrence) patterns in a unified
    recurrent architecture. This makes it particularly suitable for sequences
    of graph-structured data like facial landmarks over time.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        temporal_pool: str = 'last'
    ):
        """
        Initialize GConvLSTM model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GConvLSTM layers (stacked)
            dropout: Dropout rate
            temporal_pool: Temporal pooling method ('last', 'mean', 'max')
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_layers = num_layers
        self.temporal_pool = temporal_pool
        
        # Stacked GConvLSTM cells
        self.cells = nn.ModuleList()
        self.cells.append(GConvLSTMCell(in_features, hidden_dim, dropout))
        for _ in range(num_layers - 1):
            self.cells.append(GConvLSTMCell(hidden_dim, hidden_dim, dropout))
        
        # Dropout between layers
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
        
        # Initialize hidden and cell states for each layer
        h_states = [None] * self.num_layers
        c_states = [None] * self.num_layers
        
        # Process sequence frame by frame
        for t in range(num_frames):
            x_t = x[:, t, :, :]  # (batch, nodes, features)
            
            # Process through stacked GConvLSTM layers
            for layer_idx, cell in enumerate(self.cells):
                h_states[layer_idx], c_states[layer_idx] = cell(
                    x_t if layer_idx == 0 else h_states[layer_idx - 1],
                    adj,
                    h_states[layer_idx],
                    c_states[layer_idx]
                )
                if layer_idx < len(self.cells) - 1:
                    h_states[layer_idx] = self.dropout(h_states[layer_idx])
        
        # Get final hidden states from all layers
        # Use the last layer's hidden state
        h_final = h_states[-1]  # (batch, nodes, hidden_dim)
        
        # Spatial pooling (aggregate over nodes) - using max to preserve discriminative signals
        h_pooled = h_final.max(dim=1)[0]  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(h_pooled)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'temporal_pool': self.temporal_pool,
        })
        return config

