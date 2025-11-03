# Basic GNN model for lip reading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from datetime import datetime

class LipReadingGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(LipReadingGNN, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        
        # Temporal modeling
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # Final classification
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x, edge_index, batch):
        # Process each graph in the sequence
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling for each graph
        x = global_mean_pool(x, batch)
        
        # Reshape for LSTM
        batch_size = len(torch.unique(batch))
        seq_len = x.size(0) // batch_size
        x = x.view(batch_size, seq_len, -1)
        
        # Temporal modeling
        x, _ = self.lstm(x)
        
        # Take the final prediction
        x = self.fc(x[:, -1, :])
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
            'timestamp': timestamp
        }
        torch.save(checkpoint, path)