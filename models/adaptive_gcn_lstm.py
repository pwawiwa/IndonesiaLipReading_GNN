"""
Hybrid Adaptive GCN-LSTM model: Adaptive GCN for spatial, LSTM for temporal.
Combines learnable graph structure with temporal modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseGNNModel
from .layers import AdaptiveGCNConv


class AdaptiveGCNLSTMModel(BaseGNNModel):
    """
    Hybrid Adaptive GCN-LSTM model: Adaptive GCN for spatial, LSTM for temporal.
    
    Why Adaptive GCN LSTM improves over GIN LSTM:
    1. Adaptive graph learning: Discovers task-specific connections beyond fixed topology
    2. Long-range dependencies: Learns connections like lip corners ↔ jaw for certain visemes
    3. Co-articulation patterns: Captures relationships between lips, cheeks, and jaw
    4. Symmetry relationships: Learns left-right facial landmark correlations
    5. Task-specific structures: Different words/visemes may require different graph structures
    6. More flexible than fixed GIN topology: Adapts graph structure to lip reading task
    
    Comparison with GIN LSTM:
    - GIN LSTM: Uses fixed MediaPipe topology, powerful MLP-based aggregation
    - Adaptive GCN LSTM: Learns graph structure, combines fixed + adaptive adjacency
    - Both use LSTM for temporal modeling
    - Adaptive GCN can discover connections GIN cannot (not in original topology)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_nodes: int,
        num_gcn_layers: int = 2,
        num_lstm_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
        alpha: float = 0.5
    ):
        """
        Initialize Adaptive GCN-LSTM model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_nodes: Number of nodes in the graph (e.g., 468 for full face)
            num_gcn_layers: Number of Adaptive GCN layers
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            alpha: Weight for combining fixed and adaptive adjacency (0=only adaptive, 1=only fixed)
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        self.num_gcn_layers = num_gcn_layers
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.num_nodes = num_nodes
        self.alpha = alpha
        
        # Adaptive GCN layers (spatial)
        self.convs = nn.ModuleList()
        self.convs.append(AdaptiveGCNConv(in_features, hidden_dim, num_nodes, alpha))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(AdaptiveGCNConv(hidden_dim, hidden_dim, num_nodes, alpha))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LSTM (temporal)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, num_classes)
    
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
        
        # Spatial: Process each frame with Adaptive GCN
        # Reshape to (batch * frames, nodes, features)
        x = x.reshape(batch_size * num_frames, num_nodes, -1)
        
        # Apply Adaptive GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)  # Uses both fixed and adaptive adjacency
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Spatial pooling (aggregate nodes) - using mean to capture all information
        # Mean pooling performs better than max for lip reading (captures comprehensive spatial info)
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        x = x.mean(dim=2)  # (batch, frames, hidden_dim)
        
        # Learn from all frames (don't mask)
        # Temporal: LSTM processes all frames
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, frames, hidden_dim)
        
        # Use speech mask as attention on LSTM output (guidance, not masking)
        if speech_mask is not None:
            # speech_mask: (batch, frames)
            # Convert to attention weights
            attention_weights = torch.softmax(speech_mask * 10.0, dim=1)  # (batch, frames)
            attention_weights = attention_weights.unsqueeze(-1)  # (batch, frames, 1)
            
            # Weighted sum of LSTM outputs (all frames contribute, speech frames weighted more)
            h_weighted = (lstm_out * attention_weights).sum(dim=1) / (attention_weights.sum(dim=1) + 1e-6)
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
            'num_gcn_layers': self.num_gcn_layers,
            'num_lstm_layers': self.num_lstm_layers,
            'bidirectional': self.bidirectional,
            'num_nodes': self.num_nodes,
            'alpha': self.alpha,
        })
        return config

