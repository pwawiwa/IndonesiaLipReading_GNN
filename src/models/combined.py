"""
src/models/combined.py
Simplified Combined Spatial-Temporal Model
Reduced complexity to prevent overfitting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from .spatial import SpatialGCN
from .temporal import TemporalLSTM


class CombinedModel(nn.Module):
    """Simplified Combined Spatial-Temporal GNN for lip reading
    
    Reduced from 33M to ~6-8M parameters by:
    - Single temporal branch (LSTM only)
    - Reduced hidden dimensions
    - Simplified classifier
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 spatial_dim: int = 256, temporal_dim: int = 256,
                 spatial_layers: int = 3, temporal_layers: int = 2,
                 dropout: float = 0.5):
        """
        Args:
            input_dim: Input feature dimension per node
            num_classes: Number of classes
            spatial_dim: Spatial GCN hidden dimension (reduced from 384)
            temporal_dim: Temporal LSTM hidden dimension (reduced from 384)
            spatial_layers: Number of GCN layers (reduced from 4)
            temporal_layers: Number of LSTM layers
            dropout: Dropout rate (increased from 0.3)
        """
        super().__init__()
        
        # Spatial module (simplified)
        self.spatial = SpatialGCN(
            input_dim=input_dim,
            hidden_dim=spatial_dim,
            num_layers=spatial_layers,
            dropout=dropout
        )
        
        # Single temporal branch (LSTM only, removed Attention and Conv1D)
        self.temporal = TemporalLSTM(
            input_dim=self.spatial.output_dim,
            hidden_dim=temporal_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Simplified classifier (2 layers instead of 3)
        lstm_output_dim = self.temporal.output_dim  # temporal_dim * 2 (bidirectional)
        classifier_dim = max(temporal_dim, lstm_output_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, classifier_dim),
            nn.BatchNorm1d(classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes)
        )
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data batch
            
        Returns:
            logits [batch_size, num_classes]
        """
        # Extract data
        x_temporal = data.x_temporal
        batch_size = data.num_graphs
        
        # Handle PyG batching of x_temporal
        if isinstance(x_temporal, (list, tuple)):
            x_temporal = torch.stack(x_temporal, dim=0)  # [batch, T, N, feat]
        elif len(x_temporal.shape) == 3:
            if hasattr(data, 'num_frames') and data.num_frames is not None:
                num_frames = data.num_frames
                if num_frames.dim() > 1:
                    num_frames = num_frames.squeeze(-1)
                if num_frames.dim() == 0:
                    seq_len = num_frames.item()
                else:
                    seq_len = num_frames[0].item()
                
                total_first, num_nodes, feat_dim = x_temporal.shape
                x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
            else:
                total_first, second_dim, feat_dim = x_temporal.shape
                if total_first % batch_size == 0:
                    seq_len = total_first // batch_size
                    num_nodes = second_dim
                    x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
                else:
                    num_nodes = total_first // batch_size
                    if total_first % batch_size == 0:
                        seq_len = second_dim
                        x_temporal = x_temporal.view(batch_size, num_nodes, seq_len, feat_dim)
                        x_temporal = x_temporal.transpose(1, 2)  # [batch, T, N, feat]
                    else:
                        raise ValueError(f"Cannot infer x_temporal shape from {x_temporal.shape} with batch_size={batch_size}")
        
        batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        
        # Use edge_index from data if available, otherwise build k-NN
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
        
        # Process each timestep through spatial GCN
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
            
            # Process with spatial GCN
            graph_embed = self.spatial(x_flat, edge_batch, batch_assign)  # [batch, spatial_dim*2]
            temporal_embeds.append(graph_embed)
        
        # Stack temporal: [batch, seq, spatial_dim*2]
        temporal_seq = torch.stack(temporal_embeds, dim=1)
        
        # Apply speech mask if available
        if hasattr(data, 'speech_mask') and data.speech_mask is not None:
            if isinstance(data.speech_mask, (list, tuple)):
                speech_mask = torch.stack(data.speech_mask, dim=0)  # [batch, seq]
            else:
                if len(data.speech_mask.shape) == 1:
                    speech_mask = data.speech_mask.unsqueeze(0)
                else:
                    speech_mask = data.speech_mask
            
            if speech_mask.shape[1] == seq_len:
                speech_mask = speech_mask.to(x_temporal.device)
                temporal_seq = temporal_seq * speech_mask.unsqueeze(-1)  # [batch, seq, feat]
        
        # Sequence lengths for masking
        lengths = None
        if hasattr(data, 'num_frames') and data.num_frames is not None:
            lengths = data.num_frames.to(x_temporal.device)
            if lengths.dim() > 1:
                lengths = lengths.squeeze(-1)
        
        # Process with temporal LSTM (single branch)
        _, temporal_out = self.temporal(temporal_seq, lengths=lengths)  # [batch, temporal_dim*2]
        
        # Classify (simplified)
        logits = self.classifier(temporal_out)  # [batch, num_classes]
        
        return logits
    
    def _get_base_edges(self, num_nodes, device):
        """Build edge index for single graph (fallback)"""
        edges = []
        k = 5
        
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
