"""
src/models/combined.py
Simple Combined Spatial-Temporal Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from .spatial import SpatialGCN
from .temporal import TemporalLSTM, TemporalAttention


class CombinedModel(nn.Module):
    """Combined Spatial-Temporal GNN for lip reading"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 spatial_dim: int = 128, temporal_dim: int = 256,
                 spatial_layers: int = 3, temporal_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension per node
            num_classes: Number of classes
            spatial_dim: Spatial GCN hidden dimension
            temporal_dim: Temporal LSTM hidden dimension
            spatial_layers: Number of GCN layers
            temporal_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Spatial module
        self.spatial = SpatialGCN(
            input_dim=input_dim,
            hidden_dim=spatial_dim,
            num_layers=spatial_layers,
            dropout=dropout
        )
        
        # Temporal modules
        self.temporal = TemporalLSTM(
            input_dim=self.spatial.output_dim,
            hidden_dim=temporal_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=True
        )
        self.temporal_attn = TemporalAttention(
            input_dim=self.spatial.output_dim,
            hidden_dim=temporal_dim,
            num_heads=8,
            num_layers=max(1, temporal_layers - 1),
            dropout=dropout
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.spatial.output_dim, temporal_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_dim),
            nn.ReLU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        proj_dim = temporal_dim
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(self.temporal.output_dim),
            nn.Linear(self.temporal.output_dim, proj_dim)
        )
        self.attn_proj = nn.Sequential(
            nn.LayerNorm(self.temporal_attn.output_dim),
            nn.Linear(self.temporal_attn.output_dim, proj_dim)
        )
        self.conv_proj = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout to fusion gate
            nn.Linear(proj_dim, 3),
            nn.Softmax(dim=1)
        )

        fusion_dim = proj_dim * 4  # three branches + gated blend
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        # Classifier with stronger regularization
        mid_dim = max(temporal_dim * 2, fusion_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # Increased dropout in classifier
            nn.Linear(mid_dim, mid_dim // 2),
            nn.BatchNorm1d(mid_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # Additional dropout layer
            nn.Linear(mid_dim // 2, num_classes)
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
        # PyG DataLoader may keep x_temporal as a list or batch it
        x_temporal = data.x_temporal
        batch_size = data.num_graphs
        
        # Handle PyG batching of x_temporal
        # PyG typically keeps custom attributes as lists
        if isinstance(x_temporal, (list, tuple)):
            # Stack list of tensors: each is [T, N, feat_dim]
            x_temporal = torch.stack(x_temporal, dim=0)  # [batch, T, N, feat]
        elif len(x_temporal.shape) == 3:
            # PyG may have concatenated: need to reshape
            # Get sequence length from num_frames
            if hasattr(data, 'num_frames') and data.num_frames is not None:
                num_frames = data.num_frames
                if num_frames.dim() > 1:
                    num_frames = num_frames.squeeze(-1)
                if num_frames.dim() == 0:
                    seq_len = num_frames.item()
                else:
                    seq_len = num_frames[0].item()  # Use first sample's length
                
                total_first, num_nodes, feat_dim = x_temporal.shape
                # Reshape: [batch*T, N, feat] -> [batch, T, N, feat]
                x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
            else:
                # Fallback: try to infer from shape
                total_first, second_dim, feat_dim = x_temporal.shape
                # Try [batch*T, N, feat] format
                if total_first % batch_size == 0:
                    seq_len = total_first // batch_size
                    num_nodes = second_dim
                    x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, feat_dim)
                else:
                    # Try [batch*N, T, feat] format
                    num_nodes = total_first // batch_size
                    if total_first % batch_size == 0:
                        seq_len = second_dim
                        x_temporal = x_temporal.view(batch_size, num_nodes, seq_len, feat_dim)
                        x_temporal = x_temporal.transpose(1, 2)  # [batch, T, N, feat]
                    else:
                        raise ValueError(f"Cannot infer x_temporal shape from {x_temporal.shape} with batch_size={batch_size}")
        
        # Now x_temporal should be [batch, seq, nodes, feat]
        batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        
        # Build base edge index for one graph
        base_edges = self._get_base_edges(num_nodes, x_temporal.device)
        
        # Process each timestep through spatial GCN
        temporal_embeds = []
        
        for t in range(seq_len):
            # Get features at timestep t
            x_t = x_temporal[:, t, :, :]  # [batch, nodes, feat]
            
            # Flatten: [batch*nodes, feat]
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
        
        # Sequence lengths for masking
        lengths = None
        if hasattr(data, 'num_frames') and data.num_frames is not None:
            lengths = data.num_frames.to(x_temporal.device)
            if lengths.dim() > 1:
                lengths = lengths.squeeze(-1)
        
        # Process with temporal LSTM
        _, temporal_out = self.temporal(temporal_seq, lengths=lengths)  # [batch, temporal_dim*2]

        # Attention-based temporal context
        pad_mask = None
        if lengths is not None:
            max_len = temporal_seq.size(1)
            pad_mask = torch.arange(max_len, device=temporal_seq.device).unsqueeze(0) >= lengths.unsqueeze(1)
        _, attn_pooled = self.temporal_attn(temporal_seq, mask=pad_mask)  # [batch, temporal_dim]

        # Temporal convolutional context
        conv_seq = temporal_seq
        if pad_mask is not None:
            conv_seq = conv_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        conv_input = conv_seq.transpose(1, 2)  # [batch, feat, seq]
        conv_out = self.temporal_conv(conv_input).squeeze(-1)  # [batch, temporal_dim]

        # Project to common representation
        temporal_feat = self.temporal_proj(temporal_out)
        attn_feat = self.attn_proj(attn_pooled)
        conv_feat = self.conv_proj(conv_out)

        # Adaptive gating over branches
        gate_logits = torch.cat([temporal_feat, attn_feat, conv_feat], dim=1)
        gate = self.fusion_gate(gate_logits)  # [batch, 3]
        blended = (
            gate[:, 0:1] * temporal_feat +
            gate[:, 1:2] * attn_feat +
            gate[:, 2:3] * conv_feat
        )
        
        # Fuse temporal signals
        fusion = torch.cat([temporal_feat, attn_feat, conv_feat, blended], dim=1)
        fusion = self.fusion_norm(fusion)
        
        # Classify
        logits = self.classifier(fusion)  # [batch, num_classes]
        
        return logits
    
    def _get_base_edges(self, num_nodes, device):
        """Build edge index for single graph"""
        edges = []
        k = 5
        
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()