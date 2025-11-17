"""
Alternative model architectures for spatial-temporal lip reading
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GCNConv, GATConv
from typing import Optional

from .spatial import SpatialGCN
from .temporal import TemporalLSTM


class AlternativeSpatialGCN(nn.Module):
    """Alternative spatial GCN with attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 2  # Mean + Max pooling
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment [N]
        """
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        # Attention-based pooling
        att_weights = self.attention(x)  # [N, 1]
        att_weights = F.softmax(att_weights, dim=0)
        att_pool = (x * att_weights).sum(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Mean and max pooling
        mean_pool = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        max_pool = global_max_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Combine
        if batch is not None and len(batch) > 0:
            batch_size = batch.max().item() + 1
            att_pool = att_pool.repeat(batch_size, 1)
        else:
            batch_size = 1
        
        combined = torch.cat([mean_pool, max_pool], dim=1)  # [batch_size, hidden_dim*2]
        
        return combined


class AlternativeTemporalLSTM(nn.Module):
    """Alternative temporal model with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.5, bidirectional: bool = True, use_attention: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.output_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            lengths: Sequence lengths (optional)
        """
        # Pack if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply attention if enabled
        if self.use_attention:
            att_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
            att_weights = F.softmax(att_weights, dim=1)
            temporal_out = (lstm_out * att_weights).sum(dim=1)  # [batch, output_dim]
        else:
            # Use last hidden state
            if self.bidirectional:
                h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, output_dim]
            else:
                h_n = h_n[-1]  # [batch, output_dim]
            temporal_out = h_n
        
        temporal_out = self.layer_norm(temporal_out)
        
        return lstm_out, temporal_out


class ModelV2_Attention(nn.Module):
    """Model with attention-based spatial and temporal processing"""
    
    def __init__(self, input_dim: int, num_classes: int,
                 spatial_dim: int = 256, temporal_dim: int = 256,
                 spatial_layers: int = 3, temporal_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        
        self.spatial = AlternativeSpatialGCN(
            input_dim=input_dim,
            hidden_dim=spatial_dim,
            num_layers=spatial_layers,
            dropout=dropout
        )
        
        self.temporal = AlternativeTemporalLSTM(
            input_dim=self.spatial.output_dim,
            hidden_dim=temporal_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=True,
            use_attention=True
        )
        
        lstm_output_dim = self.temporal.output_dim
        classifier_dim = max(temporal_dim, lstm_output_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, classifier_dim),
            nn.BatchNorm1d(classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes)
        )
    
    def forward(self, data):
        """Forward pass"""
        x_temporal = data.x_temporal
        batch_size = data.num_graphs
        
        # Handle batching
        if isinstance(x_temporal, (list, tuple)):
            x_temporal = torch.stack(x_temporal, dim=0)
        elif len(x_temporal.shape) == 3:
            if hasattr(data, 'num_frames') and data.num_frames is not None:
                num_frames = data.num_frames
                if num_frames.dim() > 1:
                    num_frames = num_frames.squeeze(-1)
                seq_len = num_frames[0].item() if num_frames.dim() > 0 else num_frames.item()
                total_first, num_nodes, feat_dim = x_temporal.shape
                x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
            else:
                total_first, second_dim, feat_dim = x_temporal.shape
                if total_first % batch_size == 0:
                    seq_len = total_first // batch_size
                    num_nodes = second_dim
                    x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
                else:
                    raise ValueError(f"Cannot infer x_temporal shape")
        
        batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        
        # Get edge index
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            base_edges = data.edge_index
            if base_edges.shape[1] > 0:
                mask = (base_edges[0] < num_nodes) & (base_edges[1] < num_nodes)
                base_edges = base_edges[:, mask]
                if base_edges.shape[1] == 0:
                    base_edges = self._get_base_edges(num_nodes, x_temporal.device)
            else:
                base_edges = self._get_base_edges(num_nodes, x_temporal.device)
        else:
            base_edges = self._get_base_edges(num_nodes, x_temporal.device)
        
        # Process each timestep
        temporal_embeds = []
        for t in range(seq_len):
            x_t = x_temporal[:, t, :, :]  # [batch, nodes, feat]
            x_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
            
            # Create batched edges
            edge_batch = []
            for b in range(batch_size):
                offset = b * num_nodes
                edge_batch.append(base_edges + offset)
            edge_batch = torch.cat(edge_batch, dim=1)
            
            # Batch assignment
            batch_assign = torch.arange(batch_size, device=x_t.device).repeat_interleave(num_nodes)
            
            # Spatial processing
            graph_embed = self.spatial(x_flat, edge_batch, batch_assign)
            temporal_embeds.append(graph_embed)
        
        # Stack temporal
        temporal_seq = torch.stack(temporal_embeds, dim=1)  # [batch, seq, spatial_dim*2]
        
        # Apply speech mask
        if hasattr(data, 'speech_mask') and data.speech_mask is not None:
            if isinstance(data.speech_mask, (list, tuple)):
                speech_mask = torch.stack(data.speech_mask, dim=0)
            else:
                if len(data.speech_mask.shape) == 1:
                    speech_mask = data.speech_mask.unsqueeze(0)
                else:
                    speech_mask = data.speech_mask
            
            if speech_mask.shape[1] == seq_len:
                speech_mask = speech_mask.to(x_temporal.device)
                temporal_seq = temporal_seq * speech_mask.unsqueeze(-1)
        
        # Temporal processing
        lengths = None
        if hasattr(data, 'num_frames') and data.num_frames is not None:
            lengths = data.num_frames.to(x_temporal.device)
            if lengths.dim() > 1:
                lengths = lengths.squeeze(-1)
        
        _, temporal_out = self.temporal(temporal_seq, lengths=lengths)
        
        # Classify
        logits = self.classifier(temporal_out)
        
        return logits
    
    def _get_base_edges(self, num_nodes, device):
        """Build edge index"""
        edges = []
        k = 5
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


class ModelV3_ConvTemporal(nn.Module):
    """Model with 1D convolution for temporal processing instead of LSTM"""
    
    def __init__(self, input_dim: int, num_classes: int,
                 spatial_dim: int = 256, temporal_dim: int = 256,
                 spatial_layers: int = 3, temporal_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        
        self.spatial = SpatialGCN(
            input_dim=input_dim,
            hidden_dim=spatial_dim,
            num_layers=spatial_layers,
            dropout=dropout
        )
        
        # 1D Convolution for temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.spatial.output_dim, temporal_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        for _ in range(temporal_layers - 1):
            self.temporal_conv.append(nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1))
            self.temporal_conv.append(nn.BatchNorm1d(temporal_dim))
            self.temporal_conv.append(nn.ReLU())
            self.temporal_conv.append(nn.Dropout(dropout))
        
        # Global pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        classifier_dim = max(temporal_dim, temporal_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, classifier_dim),
            nn.BatchNorm1d(classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes)
        )
    
    def forward(self, data):
        """Forward pass"""
        x_temporal = data.x_temporal
        batch_size = data.num_graphs
        
        # Handle batching (same as ModelV2)
        if isinstance(x_temporal, (list, tuple)):
            x_temporal = torch.stack(x_temporal, dim=0)
        elif len(x_temporal.shape) == 3:
            if hasattr(data, 'num_frames') and data.num_frames is not None:
                num_frames = data.num_frames
                if num_frames.dim() > 1:
                    num_frames = num_frames.squeeze(-1)
                seq_len = num_frames[0].item() if num_frames.dim() > 0 else num_frames.item()
                total_first, num_nodes, feat_dim = x_temporal.shape
                x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
            else:
                total_first, second_dim, feat_dim = x_temporal.shape
                if total_first % batch_size == 0:
                    seq_len = total_first // batch_size
                    num_nodes = second_dim
                    x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
        
        batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        
        # Get edge index
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            base_edges = data.edge_index
            if base_edges.shape[1] > 0:
                mask = (base_edges[0] < num_nodes) & (base_edges[1] < num_nodes)
                base_edges = base_edges[:, mask]
                if base_edges.shape[1] == 0:
                    base_edges = self._get_base_edges(num_nodes, x_temporal.device)
            else:
                base_edges = self._get_base_edges(num_nodes, x_temporal.device)
        else:
            base_edges = self._get_base_edges(num_nodes, x_temporal.device)
        
        # Process each timestep
        temporal_embeds = []
        for t in range(seq_len):
            x_t = x_temporal[:, t, :, :]
            x_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
            
            edge_batch = []
            for b in range(batch_size):
                offset = b * num_nodes
                edge_batch.append(base_edges + offset)
            edge_batch = torch.cat(edge_batch, dim=1)
            
            batch_assign = torch.arange(batch_size, device=x_t.device).repeat_interleave(num_nodes)
            
            graph_embed = self.spatial(x_flat, edge_batch, batch_assign)
            temporal_embeds.append(graph_embed)
        
        # Stack: [batch, seq, spatial_dim*2]
        temporal_seq = torch.stack(temporal_embeds, dim=1)
        
        # Apply speech mask
        if hasattr(data, 'speech_mask') and data.speech_mask is not None:
            if isinstance(data.speech_mask, (list, tuple)):
                speech_mask = torch.stack(data.speech_mask, dim=0)
            else:
                if len(data.speech_mask.shape) == 1:
                    speech_mask = data.speech_mask.unsqueeze(0)
                else:
                    speech_mask = data.speech_mask
            
            if speech_mask.shape[1] == seq_len:
                speech_mask = speech_mask.to(x_temporal.device)
                temporal_seq = temporal_seq * speech_mask.unsqueeze(-1)
        
        # Conv1D expects [batch, channels, seq]
        temporal_seq = temporal_seq.transpose(1, 2)  # [batch, spatial_dim*2, seq]
        
        # Temporal convolution
        temporal_out = self.temporal_conv(temporal_seq)  # [batch, temporal_dim, seq]
        
        # Global pooling
        temporal_out = self.temporal_pool(temporal_out).squeeze(-1)  # [batch, temporal_dim]
        
        # Classify
        logits = self.classifier(temporal_out)
        
        return logits
    
    def _get_base_edges(self, num_nodes, device):
        """Build edge index"""
        edges = []
        k = 5
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

