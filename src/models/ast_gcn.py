"""
AST-GCN: Attention-based Spatial-Temporal Graph Convolutional Network
V5 Model with attention mechanisms for both spatial and temporal processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree

from .temporal import TemporalLSTM


class AttentionSpatialGCN(nn.Module):
    """Attention-based Spatial GCN using Graph Attention Networks"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, 
                 num_heads: int = 4, dropout: float = 0.5):
        """
        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, 
                                     dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Final projection to reduce dimension
        self.final_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        self.dropout = dropout
        self.output_dim = hidden_dim * 2  # mean + max pooling
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connections [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Add self-loops for better connectivity
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Apply GAT layers with attention
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Project to final dimension
        x = self.final_proj(x)
        
        # Pool to graph level with attention-weighted pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        return torch.cat([x_mean, x_max], dim=1)


class AttentionTemporalLSTM(nn.Module):
    """Temporal LSTM with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.5, bidirectional: bool = True, use_attention: bool = True):
        """
        Args:
            input_dim: Input feature dimension per timestep
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
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
        self.dropout = nn.Dropout(dropout)
    
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
        temporal_out = self.dropout(temporal_out)
        
        return lstm_out, temporal_out


class ASTGCNModel(nn.Module):
    """AST-GCN: Attention-based Spatial-Temporal Graph Convolutional Network"""
    
    def __init__(self, input_dim: int, num_classes: int,
                 spatial_dim: int = 256, temporal_dim: int = 256,
                 spatial_layers: int = 3, temporal_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.5):
        """
        Args:
            input_dim: Input feature dimension per node
            num_classes: Number of classes
            spatial_dim: Spatial GAT hidden dimension
            temporal_dim: Temporal LSTM hidden dimension
            spatial_layers: Number of GAT layers
            temporal_layers: Number of LSTM layers
            num_heads: Number of attention heads for GAT
            dropout: Dropout rate
        """
        super().__init__()
        
        # Attention-based spatial module
        self.spatial = AttentionSpatialGCN(
            input_dim=input_dim,
            hidden_dim=spatial_dim,
            num_layers=spatial_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Attention-based temporal module
        self.temporal = AttentionTemporalLSTM(
            input_dim=self.spatial.output_dim,
            hidden_dim=temporal_dim,
            num_layers=temporal_layers,
            dropout=dropout,
            bidirectional=True,
            use_attention=True
        )
        
        # Classifier
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
            x_temporal = torch.stack(x_temporal, dim=0)
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
                    raise ValueError(f"Cannot infer x_temporal shape from {x_temporal.shape}")
        
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
        
        # Process each timestep through attention-based spatial GCN
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
            
            # Process with attention-based spatial GCN
            graph_embed = self.spatial(x_flat, edge_batch, batch_assign)  # [batch, spatial_dim*2]
            temporal_embeds.append(graph_embed)
        
        # Stack temporal: [batch, seq, spatial_dim*2]
        temporal_seq = torch.stack(temporal_embeds, dim=1)
        
        # Apply speech mask if available
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
        
        # Sequence lengths for masking
        lengths = None
        if hasattr(data, 'num_frames') and data.num_frames is not None:
            lengths = data.num_frames.to(x_temporal.device)
            if lengths.dim() > 1:
                lengths = lengths.squeeze(-1)
        
        # Process with attention-based temporal LSTM
        _, temporal_out = self.temporal(temporal_seq, lengths=lengths)  # [batch, temporal_dim*2]
        
        # Classify
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

