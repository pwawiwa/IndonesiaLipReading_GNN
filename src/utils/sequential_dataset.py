# Dataset utility for sequential lip reading data
import torch
from torch_geometric.data import Dataset
from pathlib import Path

class SequentialLipReadingDataset(Dataset):
    def __init__(self, root, split='train', transform=None, sequence_length=30):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Load class mapping
        self.classes = sorted(
            [d.name for d in self.root.parent.glob('*') if d.is_dir()]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all .pt files for the split
        self.samples = list(self.root.glob(f'{split}/**/*.pt'))
        
        super().__init__(root, transform)
    
    def len(self):
        return len(self.samples)
    
    def get(self, idx):
        # Load processed graph sequence
        sample_path = self.samples[idx]
        graphs = torch.load(sample_path)
        
        # Handle sequences
        if len(graphs) > self.sequence_length:
            # Take center sequence if too long
            start_idx = (len(graphs) - self.sequence_length) // 2
            graphs = graphs[start_idx:start_idx + self.sequence_length]
        elif len(graphs) < self.sequence_length:
            # Pad with last frame if too short
            last_frame = graphs[-1]
            padding = [last_frame] * (self.sequence_length - len(graphs))
            graphs.extend(padding)
            
        # Get label from parent directory name
        label = sample_path.parent.name
        label_idx = self.class_to_idx[label]
        
        # Apply transforms if any
        if self.transform is not None:
            graphs = self.transform(graphs)
        
        return graphs, label_idx

    def get_sequence_length(self):
        """Return the fixed sequence length used by the dataset."""
        return self.sequence_length