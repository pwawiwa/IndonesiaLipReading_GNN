# Sequential GNN model with LSTM for lip reading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from datetime import datetime

class TemporalGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_gnn_layers=3, 
                 lstm_layers=2, dropout=0.3):
        super(TemporalGNN, self).__init__()
        
        # Model dimensions
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        
        # GNN layers with skip connections
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First GNN layer
        self.gnn_layers.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Additional GNN layers
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention layer for temporal weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data_list):
        """
        Process a list of graph snapshots (temporal sequence)
        data_list: List of Data objects, each representing a frame
        """
        batch_size = len(data_list)
        seq_length = len(data_list[0])  # Number of frames per sequence
        
        # Process each frame in the sequence
        temporal_embeddings = []
        
        for t in range(seq_length):
            # Get current frame data from all sequences
            x = torch.cat([seq[t].x for seq in data_list], dim=0)
            edge_index = torch.cat([seq[t].edge_index + i * x.size(0) 
                                  for i, seq in enumerate(data_list)], dim=1)
            batch = torch.cat([torch.full((seq[t].x.size(0),), i, 
                                        dtype=torch.long, device=x.device)
                             for i, seq in enumerate(data_list)])
            
            # Apply GNN layers with skip connections
            h = x
            for gnn, bn in zip(self.gnn_layers, self.batch_norms):
                h_new = F.relu(bn(gnn(h, edge_index)))
                h = h_new + h if h.shape == h_new.shape else h_new
                h = F.dropout(h, p=0.1, training=self.training)
            
            # Global pooling for each graph
            graph_embedding = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
            temporal_embeddings.append(graph_embedding)
        
        # Stack temporal embeddings
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)  # [batch_size, seq_length, hidden_dim]
        
        # Apply LSTM
        lstm_out, _ = self.lstm(temporal_embeddings)  # [batch_size, seq_length, hidden_dim*2]
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_length, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention to get final sequence representation
        weighted_sum = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim*2]
        
        # Final classification
        x = F.relu(self.fc1(weighted_sum))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def save_checkpoint(self, optimizer, epoch, loss, accuracy, path):
        """Save model checkpoint with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'hyperparameters': {
                'hidden_dim': self.hidden_dim,
                'num_gnn_layers': self.num_gnn_layers
            }
        }
        torch.save(checkpoint, path)