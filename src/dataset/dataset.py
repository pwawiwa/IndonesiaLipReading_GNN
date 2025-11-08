"""
src/dataset/dataset.py
Dataset loader for .pt files into PyG Data objects
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LipReadingDataset(Dataset):
    """
    Dataset class for loading .pt files and converting to PyG Data objects
    """
    
    def __init__(self, pt_file_path: str, label_to_idx: Dict[str, int] = None):
        """
        Args:
            pt_file_path: Path to the .pt file
            label_to_idx: Dictionary mapping word labels to integer indices
        """
        self.pt_file_path = Path(pt_file_path)
        
        # Load data
        logger.info(f"Loading dataset from {self.pt_file_path}")
        self.samples = torch.load(self.pt_file_path)
        logger.info(f"Loaded {len(self.samples)} samples")
        
        # Build label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(list(set([s['label'] for s in self.samples])))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"Created label mapping with {len(self.label_to_idx)} classes")
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample as PyG Data object
        
        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: Graph connectivity [2, num_edges]
                - y: Label [1]
                - Additional features for temporal modeling
        """
        sample = self.samples[idx]
        
        # Convert to float32 for training
        landmarks = sample['landmarks'].float()  # [T, N, 3]
        features = {k: v.float() for k, v in sample['features'].items()}
        
        T, N, _ = landmarks.shape  # T: time steps, N: nodes (landmarks)
        
        # Build edge index (static graph structure)
        edge_index = self._build_edge_index(N)
        
        # Prepare node features: concatenate landmarks + geometric features per timestep
        # We'll create a temporal sequence of graphs
        node_features_list = []
        for t in range(T):
            # Node features at time t: [x, y, z] + geometric features
            node_feat = landmarks[t]  # [N, 3]
            
            # Add geometric features if available (broadcasted to all nodes)
            if 'geometric' in features:
                geom_feat = features['geometric'][t]  # [5]
                # Broadcast to all nodes
                geom_feat_expanded = geom_feat.unsqueeze(0).repeat(N, 1)  # [N, 5]
                node_feat = torch.cat([node_feat, geom_feat_expanded], dim=1)  # [N, 8]
            
            # Add action units if available
            if 'action_units' in features:
                au_feat = features['action_units'][t]  # [13]
                au_feat_expanded = au_feat.unsqueeze(0).repeat(N, 1)  # [N, 13]
                node_feat = torch.cat([node_feat, au_feat_expanded], dim=1)  # [N, 21]
            
            node_features_list.append(node_feat)
        
        # Stack temporal features [T, N, F]
        node_features_temporal = torch.stack(node_features_list, dim=0)
        
        # Get label
        label_str = sample['label']
        label_idx = self.label_to_idx[label_str]
        
        # Create PyG Data object
        # For temporal graphs, we'll use the first timestep as the main graph
        # and store the full temporal sequence
        data = Data(
            x=node_features_temporal[0],  # [N, F] - First timestep features
            edge_index=edge_index,  # [2, E]
            y=torch.tensor([label_idx], dtype=torch.long),  # [1]
            # Store temporal information
            x_temporal=node_features_temporal,  # [T, N, F]
            velocity=features.get('velocity', None),  # [T-1, N, 3]
            acceleration=features.get('acceleration', None),  # [T-2, N, 3]
            edges_features=features.get('edges', None),  # [T, E_feat]
            num_frames=torch.tensor([T], dtype=torch.long),  # [1]
            video_id=sample['video_id']
        )
        
        return data
    
    def _build_edge_index(self, num_nodes: int) -> torch.Tensor:
        """
        Build edge index for the facial landmark graph
        Creates a k-nearest neighbor graph in spatial domain
        
        Args:
            num_nodes: Number of nodes (landmarks)
            
        Returns:
            edge_index: [2, num_edges]
        """
        # For mouth-centric ROI, we'll use a predefined connectivity
        # based on anatomical structure
        edges = []
        
        # Connect each node to its k nearest neighbors (in index space)
        k = 5  # Number of nearest neighbors
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        
        # Also add some specific anatomical connections
        # (These are based on the ROI structure from facemesh_extractor.py)
        # You can refine this based on domain knowledge
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def get_label_mapping(self) -> Dict[str, int]:
        """Return label to index mapping"""
        return self.label_to_idx
    
    def get_num_classes(self) -> int:
        """Return number of classes"""
        return self.num_classes


def create_dataloaders(
    train_pt: str,
    val_pt: str,
    test_pt: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple:
    """
    Create train, val, test dataloaders
    
    Args:
        train_pt: Path to train.pt
        val_pt: Path to val.pt
        test_pt: Path to test.pt
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader, label_mapping, num_classes
    """
    from torch_geometric.loader import DataLoader
    
    # Create datasets
    train_dataset = LipReadingDataset(train_pt)
    label_mapping = train_dataset.get_label_mapping()
    num_classes = train_dataset.num_classes
    
    val_dataset = LipReadingDataset(val_pt, label_to_idx=label_mapping)
    test_dataset = LipReadingDataset(test_pt, label_to_idx=label_mapping)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    logger.info(f"  Num classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, label_mapping, num_classes