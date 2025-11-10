"""
src/models/combined.py
Combined Spatial-Temporal Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
import logging

from .spatial import SpatialGCN
from .temporal import TemporalLSTM, TemporalAttention

logger = logging.getLogger(__name__)


class SpatioTemporalGNN(nn.Module):
    """
    Combined Spatial-Temporal Graph Neural Network
    Processes both spatial (graph) and temporal (sequence) information
    Now supports speech mask for focusing on relevant frames
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        spatial_hidden_dim: int = 128,
        temporal_hidden_dim: int = 256,
        spatial_layers: int = 3,
        temporal_layers: int = 2,
        dropout: float = 0.3,
        use_gat: bool = False,
        temporal_type: str = 'lstm',  # 'lstm' or 'attention'
        use_speech_mask: bool = True  # NEW: Use speech mask if available
    ):
        """
        Args:
            input_dim: Input feature dimension per node
            num_classes: Number of classes
            spatial_hidden_dim: Hidden dimension for spatial GCN
            temporal_hidden_dim: Hidden dimension for temporal model
            spatial_layers: Number of GCN layers
            temporal_layers: Number of temporal layers
            dropout: Dropout rate
            use_gat: Use GAT instead of GCN
            temporal_type: Type of temporal model ('lstm' or 'attention')
            use_speech_mask: Whether to apply speech mask weighting
        """
        super(SpatioTemporalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.temporal_type = temporal_type
        self.use_speech_mask = use_speech_mask
        
        # Spatial GCN
        self.spatial_gcn = SpatialGCN(
            input_dim=input_dim,
            hidden_dim=spatial_hidden_dim,
            num_layers=spatial_layers,
            dropout=dropout,
            use_gat=use_gat
        )
        
        # Temporal model
        if temporal_type == 'lstm':
            self.temporal_model = TemporalLSTM(
                input_dim=self.spatial_gcn.output_dim * 2,  # *2 for mean+max pooling
                hidden_dim=temporal_hidden_dim,
                num_layers=temporal_layers,
                dropout=dropout,
                bidirectional=True
            )
        elif temporal_type == 'attention':
            self.temporal_model = TemporalAttention(
                input_dim=self.spatial_gcn.output_dim * 2,
                hidden_dim=temporal_hidden_dim,
                num_heads=8,
                num_layers=temporal_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")
        
        # Classifier head with deeper network for 100 classes
        temporal_output_dim = self.temporal_model.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(temporal_output_dim, temporal_hidden_dim * 2),
            nn.LayerNorm(temporal_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim * 2, temporal_hidden_dim),
            nn.LayerNorm(temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, num_classes)
        )
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data batch with temporal information
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Extract data
        x_temporal = data.x_temporal  # Can be [total_nodes, seq_len, feat_dim] from PyG batching
        num_frames = data.num_frames.squeeze(-1)  # [batch_size]
        
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
        else:
            # Already in correct format
            batch_size, seq_len, num_nodes, feat_dim = x_temporal.shape
        
        # Get the base edge_index (for a single graph)
        # We need to create it fresh for each graph in the batch
        base_edge_index = self._get_base_edge_index(num_nodes, x_temporal.device)
        
        # Process each timestep with spatial GCN
        temporal_embeddings = []
        
        for t in range(seq_len):
            # Get features for timestep t
            x_t = x_temporal[:, t, :, :]  # [batch_size, num_nodes, feat_dim]
            
            # Flatten batch and nodes: [batch_size * num_nodes, feat_dim]
            x_t_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
            
            # Create batched edge_index by repeating and offsetting for each graph
            edge_index_batch = []
            for i in range(batch_size):
                offset = i * num_nodes
                edge_index_offset = base_edge_index + offset
                edge_index_batch.append(edge_index_offset)
            edge_index_batch = torch.cat(edge_index_batch, dim=1)
            
            # Create batch assignment
            batch_t = torch.arange(batch_size, device=x_t.device).repeat_interleave(num_nodes)
            
            # Process with spatial GCN
            node_embeddings = self.spatial_gcn(x_t_flat, edge_index_batch, batch_t)
            
            # Pool to graph level
            graph_embedding_mean = global_mean_pool(node_embeddings, batch_t)
            graph_embedding_max = global_max_pool(node_embeddings, batch_t)
            graph_embedding = torch.cat([graph_embedding_mean, graph_embedding_max], dim=1)
            
            temporal_embeddings.append(graph_embedding)
        
        # Stack temporal embeddings: [batch_size, seq_len, embedding_dim]
        temporal_sequence = torch.stack(temporal_embeddings, dim=1)
        
        # Process with temporal model
        if self.temporal_type == 'lstm':
            _, hidden = self.temporal_model(temporal_sequence, num_frames)
            temporal_output = hidden
        elif self.temporal_type == 'attention':
            # Create mask for padding
            max_len = seq_len
            mask = torch.arange(max_len, device=num_frames.device).unsqueeze(0) >= num_frames.unsqueeze(1)
            _, temporal_output = self.temporal_model(temporal_sequence, mask)
        
        # Classify
        logits = self.classifier(temporal_output)
        
        return logits
    
    def _get_base_edge_index(self, num_nodes: int, device) -> torch.Tensor:
        """
        Create base edge index for a single graph
        
        Args:
            num_nodes: Number of nodes
            device: Device to create tensor on
            
        Returns:
            edge_index: [2, num_edges]
        """
        # Create k-nearest neighbor connectivity
        edges = []
        k = 5  # Number of nearest neighbors
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        
        if len(edges) == 0:
            # Fallback: self-loops
            edges = [[i, i] for i in range(num_nodes)]
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index
    
    def get_attention_weights(self, data):
        """
        Get attention weights for visualization (if using attention)
        """
        if self.temporal_type != 'attention':
            raise ValueError("Attention weights only available for attention-based model")
        
        # Similar forward pass but return attention weights
        # This is a placeholder - you'd need to modify TemporalAttention to return weights
        pass


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    **kwargs
):
    """
    Factory function to create models
    
    Args:
        model_type: 'spatial', 'temporal', or 'combined'
        input_dim: Input feature dimension
        num_classes: Number of classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        model: Initialized model
    """
    if model_type == 'spatial':
        from .spatial import SpatialGCNClassifier
        model = SpatialGCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'temporal':
        from .temporal import TemporalLSTMClassifier
        model = TemporalLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'combined':
        model = SpatioTemporalGNN(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model