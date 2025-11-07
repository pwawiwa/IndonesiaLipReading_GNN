import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm

class LipReadingDataset(Dataset):
    """
    PyG Dataset: converts .pt files (landmarks + features) to Data objects
    Automatically converts float16 -> float32 to avoid dtype issues.
    """
    def __init__(self, pt_file: str, transform=None, pre_transform=None):
        self.pt_file = Path(pt_file)
        self.raw_data = torch.load(self.pt_file)  # List of dicts
        # Map word labels to integers
        self.word2idx = {w: i for i, w in enumerate(
            sorted({sample['label'] for sample in self.raw_data})
        )}
        super().__init__(root=str(self.pt_file.parent), transform=transform, pre_transform=pre_transform)

    def len(self):
        return len(self.raw_data)

    def get(self, idx):
        sample = self.raw_data[idx]

        # Landmarks [T, N, 3]
        landmarks = sample['landmarks'].float()  # <-- convert to float32

        T, N, C = landmarks.shape
        x = landmarks.view(-1, C)  # flatten per frame

        # Edge indices: sequential connections per frame
        edge_index = []
        for t in range(T):
            offset = t * N
            for i in range(N - 1):
                edge_index.append([offset + i, offset + i + 1])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]

        # Features: AU + geometric
        feats_list = []
        features = sample.get('features', {})
        if 'action_units' in features:
            feats_list.append(features['action_units'].float().view(T, -1))
        if 'geometric' in features:
            feats_list.append(features['geometric'].float().view(T, -1))
        if feats_list:
            node_attr = torch.cat(feats_list, dim=1).repeat_interleave(N, dim=0)
        else:
            node_attr = None

        # Label
        y = torch.tensor(self.word2idx[sample['label']], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        if node_attr is not None:
            data.x = torch.cat([data.x, node_attr], dim=1)

        return data
