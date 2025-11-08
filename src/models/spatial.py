"""
src/models/spatial.py
Spatial GCN for processing facial landmark graphs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
import logging

logger = logging.getLogger(__name__)


class SpatialGCN(nn.Module):
    """
    Spatial Graph Convolutional Network
    Processes spatial relationships between facial landmarks
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_gat: bool = False
    ):
        """
        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_gat: Use GAT instead of GCN
        """
        super(SpatialGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if use_gat:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            current_dim = hidden_dim * 4
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            current_dim = hidden_dim
        self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if use_gat:
                self.convs.append(GATConv(current_dim, hidden_dim, heads=4, concat=True))
                current_dim = hidden_dim * 4
            else:
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        self.output_dim = current_dim
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes] (optional)
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def forward_with_pooling(self, x, edge_index, batch):
        """
        Forward pass with graph-level pooling
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index, batch)
        
        # Pool to graph level
        graph_embedding_mean = global_mean_pool(node_embeddings, batch)
        graph_embedding_max = global_max_pool(node_embeddings, batch)
        
        # Concatenate both pooling strategies
        graph_embedding = torch.cat([graph_embedding_mean, graph_embedding_max], dim=1)
        
        return graph_embedding  # [batch_size, output_dim * 2]


class SpatialGCNClassifier(nn.Module):
    """
    Standalone spatial GCN classifier (without temporal modeling)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_gat: bool = False
    ):
        """
        Args:
            input_dim: Input feature dimension per node
            num_classes: Number of classes
            hidden_dim: Hidden dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_gat: Use GAT instead of GCN
        """
        super(SpatialGCNClassifier, self).__init__()
        
        self.spatial_gcn = SpatialGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat
        )
        
        # Classifier head
        gcn_output_dim = self.spatial_gcn.output_dim * 2  # *2 for mean+max pooling
        self.classifier = nn.Sequential(
            nn.Linear(gcn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data batch
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Extract graph data
        x = data.x  # [num_nodes, input_dim]
        edge_index = data.edge_index  # [2, num_edges]
        batch = data.batch  # [num_nodes]
        
        # Get graph embeddings
        graph_embedding = self.spatial_gcn.forward_with_pooling(x, edge_index, batch)
        
        # Classify
        logits = self.classifier(graph_embedding)
        
        return logits