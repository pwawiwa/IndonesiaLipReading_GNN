"""
src/models/combined.py
Simple Combined Spatial-Temporal Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from .spatial import SpatialGCN
from .temporal import TemporalLSTM


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
        
        # Temporal module
        self.temporal = TemporalLSTM(
            input_dim=self.spatial.output_dim,
            hidden_dim=temporal_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal.output_dim, temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_classes)
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
        
        # Process with temporal LSTM
        # TemporalLSTM returns (lstm_out, hidden_state)
        _, temporal_out = self.temporal(temporal_seq)  # [batch, temporal_dim*2]
        
        # Classify
        logits = self.classifier(temporal_out)  # [batch, num_classes]
        
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