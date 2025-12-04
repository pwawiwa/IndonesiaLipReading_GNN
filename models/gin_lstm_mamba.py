"""
Hybrid GIN-LSTM-Mamba model: GIN for spatial, LSTM for short-term temporal, Mamba for long-range temporal.
Sequential architecture: GNN → LSTM → Mamba → Classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")

from .base_model import BaseGNNModel
from .gin import GINConv


class GINLSTMMambaModel(BaseGNNModel):
    """
    Hybrid GIN-LSTM-Mamba model: GIN for spatial, LSTM for short-term temporal, Mamba for long-range temporal.
    
    Architecture: GNN (spatial) → LSTM (short-term) → Mamba (long-range) → Classifier
    
    Why LSTM + Mamba:
    1. LSTM excels at local temporal patterns (frame-to-frame lip movements)
    2. Mamba excels at long-range dependencies (co-articulation across frames)
    3. Sequential processing: LSTM output → Mamba input
    4. Natural hierarchy: local → global
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_gin_layers: int = 2,
        num_lstm_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
        eps: float = 0.0,
        train_eps: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        speech_mask_scale: float = 2.0,
        speech_mask_context: int = 0
    ):
        """
        Initialize GIN-LSTM-Mamba model.
        
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_gin_layers: Number of GIN layers
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            eps: Initial epsilon value for GIN aggregation
            train_eps: Whether to make epsilon trainable
            mamba_d_state: Mamba state dimension
            mamba_d_conv: Mamba convolution dimension
            mamba_expand: Mamba expansion factor
        """
        super().__init__(in_features, hidden_dim, num_classes, dropout)
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")
        
        self.num_gin_layers = num_gin_layers
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        
        # GIN layers (spatial)
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(in_features, hidden_dim, eps, train_eps))
        for _ in range(num_gin_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim, eps, train_eps))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LSTM (short-term temporal)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Mamba (long-range temporal)
        # Mamba processes LSTM output
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.mamba = Mamba(
            d_model=lstm_output_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand
        )
        
        # Layer normalization for better gradient flow
        self.ln_lstm = nn.LayerNorm(lstm_output_dim)
        self.ln_mamba = nn.LayerNorm(lstm_output_dim)
        
        # Dropout after Mamba
        self.dropout_mamba = nn.Dropout(dropout * 0.5)  # Lighter dropout after Mamba
        
        # Speech mask scaling factor (for combining with learned attention)
        self.speech_mask_scale = speech_mask_scale
        # Speech mask context: number of adjacent frames to include (dilation)
        # If > 0, applies weight to +/- N frames around speech_mask=1 frames
        # Helps account for co-articulation and imperfect mask accuracy
        self.speech_mask_context = speech_mask_context
        
        # Temporal Attention Pooling (learnable attention mechanism)
        # Learns which frames are important for classification
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(lstm_output_dim // 2, 1)  # Output: attention score per frame
        )
        
        # Classifier
        self.classifier = nn.Linear(lstm_output_dim, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
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
            x = F.relu(x)  # Always apply activation for better gradient flow
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        
        # Spatial pooling (aggregate nodes) - using mean to aggregate information from all nodes
        # Mean pooling performs better than max pooling for lip reading (captures comprehensive spatial info)
        x = x.reshape(batch_size, num_frames, num_nodes, self.hidden_dim)
        x = x.mean(dim=2)  # (batch, frames, hidden_dim)
        
        # Temporal: LSTM processes all frames (short-term patterns)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, frames, hidden_dim or 2*hidden_dim)
        
        # Mamba: Long-range dependencies (processes LSTM output)
        # Mamba expects (batch, seq_len, d_model)
        # Apply layer norm before Mamba for stability
        lstm_out_norm = self.ln_lstm(lstm_out)
        mamba_out = self.mamba(lstm_out_norm)  # (batch, frames, hidden_dim or 2*hidden_dim)
        
        # Residual connection: Add original LSTM output (not normalized) to Mamba output
        # This helps gradient flow and preserves information
        mamba_out = mamba_out + lstm_out
        
        # Layer norm after residual connection
        mamba_out = self.ln_mamba(mamba_out)
        mamba_out = self.dropout_mamba(mamba_out)
        
        # Temporal Attention Pooling (learnable attention mechanism)
        # Compute attention scores from features: (batch, frames, hidden_dim) -> (batch, frames, 1)
        attention_scores = self.temporal_attention(mamba_out)  # (batch, frames, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, frames)
        
        # Optionally combine with speech_mask (if provided)
        # This allows the model to use both learned attention and speech_mask guidance
        if speech_mask is not None:
            # Apply temporal dilation to speech_mask if context > 0
            # This includes adjacent frames (+/- N frames) around speech_mask=1
            # Helps account for co-articulation and imperfect mask accuracy
            if self.speech_mask_context > 0:
                # Create dilated mask: include +/- context frames
                dilated_mask = speech_mask.clone()
                for offset in range(1, self.speech_mask_context + 1):
                    # Shift forward (future frames)
                    forward_shift = torch.cat([
                        torch.zeros(speech_mask.shape[0], offset, device=speech_mask.device),
                        speech_mask[:, :-offset]
                    ], dim=1)
                    # Shift backward (past frames)
                    backward_shift = torch.cat([
                        speech_mask[:, offset:],
                        torch.zeros(speech_mask.shape[0], offset, device=speech_mask.device)
                    ], dim=1)
                    # Combine: any frame with speech OR adjacent frames
                    dilated_mask = torch.maximum(dilated_mask, forward_shift)
                    dilated_mask = torch.maximum(dilated_mask, backward_shift)
                speech_mask_weighted = dilated_mask
            else:
                speech_mask_weighted = speech_mask
            
            # Combine learned attention with speech_mask
            # speech_mask provides prior knowledge, learned attention adapts
            combined_scores = attention_scores + (speech_mask_weighted * self.speech_mask_scale)
        else:
            combined_scores = attention_scores
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(combined_scores, dim=1)  # (batch, frames)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, frames, 1)
        
        # Weighted sum of Mamba outputs using learned attention
        # This allows the model to focus on frames that are most discriminative
        h_weighted = (mamba_out * attention_weights).sum(dim=1)  # (batch, hidden_dim or 2*hidden_dim)
        
        # Classification
        logits = self.classifier(h_weighted)
        
        return logits
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Initialize temporal attention
        for module in self.temporal_attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_gin_layers': self.num_gin_layers,
            'num_lstm_layers': self.num_lstm_layers,
            'bidirectional': self.bidirectional,
            'eps': float(self.convs[0].eps.item()) if len(self.convs) > 0 else 0.0,
            'train_eps': self.convs[0].eps.requires_grad if len(self.convs) > 0 else False,
            'mamba_d_state': 16,
            'mamba_d_conv': 4,
            'mamba_expand': 2,
        })
        return config

