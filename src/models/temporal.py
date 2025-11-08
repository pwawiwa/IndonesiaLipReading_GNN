"""
src/models/temporal.py
Temporal LSTM for processing sequential information
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM Network
    Processes temporal sequences of graph embeddings
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension per timestep
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(TemporalLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: Sequence input [batch_size, seq_len, input_dim]
            lengths: Actual sequence lengths [batch_size] (optional)
            
        Returns:
            output: LSTM outputs [batch_size, seq_len, output_dim]
            hidden: Final hidden state [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Unpack if needed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        
        # Apply layer norm
        lstm_out = self.layer_norm(lstm_out)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        
        return lstm_out, h_n


class TemporalLSTMClassifier(nn.Module):
    """
    Standalone temporal LSTM classifier
    Works on pre-extracted features (e.g., flattened landmarks)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension per timestep
            num_classes: Number of classes
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(TemporalLSTMClassifier, self).__init__()
        
        self.temporal_lstm = TemporalLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Classifier head
        lstm_output_dim = self.temporal_lstm.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data batch with x_temporal
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Get temporal features
        x_temporal = data.x_temporal  # Can be [total_nodes, seq_len, feat_dim] from PyG batching
        
        # Handle PyG batching
        if len(x_temporal.shape) == 3:
            # PyG batched format: [total_nodes, seq_len, feat_dim]
            total_nodes, seq_len, feat_dim = x_temporal.shape
            batch_size = data.num_graphs
            num_nodes = total_nodes // batch_size
            
            # Reshape to [batch_size, num_nodes, seq_len, feat_dim]
            x_temporal = x_temporal.view(batch_size, num_nodes, seq_len, feat_dim)
            # Transpose to [batch_size, seq_len, num_nodes, feat_dim]
            x_temporal = x_temporal.transpose(1, 2)
            
            batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        else:
            # Already in correct format
            batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        
        # Flatten spatial dimension
        x_temporal_flat = x_temporal.reshape(batch_size, seq_len, num_nodes * feat_dim)
        
        # Get sequence lengths
        lengths = data.num_frames.squeeze(-1)  # [batch_size]
        
        # LSTM forward
        _, hidden = self.temporal_lstm(x_temporal_flat, lengths)
        
        # Classify
        logits = self.classifier(hidden)
        
        return logits


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism
    Alternative to LSTM for temporal modeling
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TemporalAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            output: Transformer outputs [batch_size, seq_len, hidden_dim]
            pooled: Pooled output [batch_size, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)
        
        # Transformer
        output = self.transformer(x, src_key_padding_mask=mask)
        
        # Pool (mean over sequence)
        if mask is not None:
            # Masked mean
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = output.mean(dim=1)
        
        return output, pooled