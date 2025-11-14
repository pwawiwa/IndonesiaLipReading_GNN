"""
src/dataset/dataset.py
Simple dataset loader for .pt files
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LipReadingDataset(Dataset):
    """Dataset for lip reading"""
    
    def __init__(self, pt_path: str, label_map: Dict = None):
        """
        Args:
            pt_path: Path to .pt file
            label_map: Label to index mapping (None = create new)
        """
        self.samples = torch.load(pt_path)
        
        # Build or use label mapping
        if label_map is None:
            unique_labels = sorted(set(s['label'] for s in self.samples))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map
        
        self.num_classes = len(self.label_map)
        
        logger.info(f"Loaded {len(self.samples)} samples with {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get tensors (already float32)
        landmarks = sample['landmarks']  # [T, N, 3]
        action_units = sample['action_units']  # [T, 18]
        geometric = sample['geometric']  # [T, 10]
        speech_mask = sample['speech_mask']  # [T]
        
        T, N, _ = landmarks.shape
        
        # Build edge index (from extractor)
        # Simple k-NN or anatomical connections
        edge_index = self._build_edge_index(N)
        
        # Create node features per timestep
        # Each node gets: [x, y, z] + broadcasted [AU, geometric]
        node_features_seq = []
        
        for t in range(T):
            # Node position: [N, 3]
            node_pos = landmarks[t]
            
            # Broadcast global features to all nodes
            au_broadcast = action_units[t].unsqueeze(0).repeat(N, 1)  # [N, 18]
            geo_broadcast = geometric[t].unsqueeze(0).repeat(N, 1)  # [N, 10]
            
            # Concatenate: [N, 3+18+10] = [N, 31]
            node_feat = torch.cat([node_pos, au_broadcast, geo_broadcast], dim=1)
            
            node_features_seq.append(node_feat)
        
        # Stack: [T, N, 31]
        node_features_seq = torch.stack(node_features_seq, dim=0)
        
        # Get label
        label_idx = self.label_map[sample['label']]
        
        # Create PyG Data object
        data = Data(
            x=node_features_seq[0],  # [N, 31] first frame for graph structure
            edge_index=edge_index,  # [2, E]
            y=torch.tensor([label_idx], dtype=torch.long),  # [1]
            # Temporal data
            x_temporal=node_features_seq,  # [T, N, 31]
            speech_mask=speech_mask,  # [T]
            num_frames=torch.tensor([T], dtype=torch.long),  # [1]
            video_id=sample['video_id'],
        )
        
        return data
    
    def _build_edge_index(self, num_nodes):
        """
        Build edge index for graph
        Simple k-NN in index space
        
        Args:
            num_nodes: Number of nodes
            
        Returns:
            [2, E] edge index
        """
        edges = []
        k = 5  # Connect to 5 nearest neighbors
        
        for i in range(num_nodes):
            for j in range(max(0, i - k), min(num_nodes, i + k + 1)):
                if i != j:
                    edges.append([i, j])
        
        if not edges:
            # Fallback: self-loops
            edges = [[i, i] for i in range(num_nodes)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index


def create_dataloaders(train_pt: str, val_pt: str, test_pt: str, 
                       batch_size: int = 32, num_workers: int = 4):
    """
    Create dataloaders for train/val/test
    
    Args:
        train_pt: Path to train.pt
        val_pt: Path to val.pt
        test_pt: Path to test.pt
        batch_size: Batch size
        num_workers: DataLoader workers
        
    Returns:
        train_loader, val_loader, test_loader, num_classes, label_map
    """
    # Create datasets
    train_dataset = LipReadingDataset(train_pt)
    label_map = train_dataset.label_map
    num_classes = train_dataset.num_classes
    
    val_dataset = LipReadingDataset(val_pt, label_map=label_map)
    test_dataset = LipReadingDataset(test_pt, label_map=label_map)
    
    # Create loaders
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, num_classes, label_map