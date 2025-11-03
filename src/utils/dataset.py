# Dataset utility for lip reading data
import torch
from torch_geometric.data import Dataset
from pathlib import Path

class LipReadingDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
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
        
        # Get label from parent directory name
        label = sample_path.parent.name
        label_idx = self.class_to_idx[label]
        
        # Apply transforms if any
        if self.transform is not None:
            graphs = self.transform(graphs)
        
        return graphs, label_idx