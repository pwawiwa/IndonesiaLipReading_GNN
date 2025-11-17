"""
src/dataset/dataset.py
Enhanced dataset loader with advanced features (v5)
Includes: Gabor filters, recurrence plots, FFT, multi-scale temporal, relative motion
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
# scipy imports removed - using torch operations only

logger = logging.getLogger(__name__)


class LipReadingDataset(Dataset):
    """Enhanced dataset for lip reading with advanced features"""
    
    def __init__(self, pt_path: str, label_map: Dict = None, use_advanced_features: bool = True):
        """
        Args:
            pt_path: Path to .pt file
            label_map: Label to index mapping (None = create new)
            use_advanced_features: Whether to compute advanced features (Gabor, FFT, etc.)
        """
        self.samples = torch.load(pt_path)
        self.use_advanced_features = use_advanced_features
        
        # Build or use label mapping
        if label_map is None:
            unique_labels = sorted(set(s['label'] for s in self.samples))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map
        
        self.num_classes = len(self.label_map)
        
        logger.info(f"Loaded {len(self.samples)} samples with {self.num_classes} classes")
        if use_advanced_features:
            logger.info("Advanced features enabled: Gabor, Recurrence, FFT, Multi-scale, Relative motion")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get tensors (already float32)
        landmarks = sample['landmarks']  # [T, N, 3]
        action_units = sample['action_units']  # [T, 18]
        geometric = sample['geometric']  # [T, 10 or 15] (backward compatible)
        speech_mask = sample['speech_mask']  # [T]
        
        T, N, _ = landmarks.shape
        
        # Expand geometric features if needed (from 10 to 15 with inner lip features)
        if geometric.shape[1] == 10:
            # Compute inner lip features from landmarks
            inner_lip_features = self._compute_inner_lip_features(landmarks)  # [T, 5]
            geometric = torch.cat([geometric, inner_lip_features], dim=1)  # [T, 15]
        
        # Get velocity and acceleration if available
        velocity = sample.get('velocity', None)  # [T-1, N, 3] or None
        acceleration = sample.get('acceleration', None)  # [T-2, N, 3] or None
        
        # Pad velocity and acceleration to match T frames
        # Use edge padding: repeat first/last frame
        if velocity is not None:
            if velocity.shape[0] == T - 1:
                # Pad with last frame
                velocity = torch.cat([velocity, velocity[-1:]], dim=0)  # [T, N, 3]
            elif velocity.shape[0] < T - 1:
                # Pad with zeros if too short
                padding = torch.zeros((T - velocity.shape[0], N, 3), dtype=velocity.dtype)
                velocity = torch.cat([velocity, padding], dim=0)  # [T, N, 3]
            elif velocity.shape[0] > T:
                # Truncate if too long
                velocity = velocity[:T]
        else:
            # No velocity: use zeros
            velocity = torch.zeros((T, N, 3), dtype=landmarks.dtype)
        
        if acceleration is not None:
            if acceleration.shape[0] == T - 2:
                # Pad with last frame (repeat twice to get to T)
                if acceleration.shape[0] > 0:
                    last_frame = acceleration[-1:]
                    acceleration = torch.cat([acceleration, last_frame, last_frame], dim=0)  # [T, N, 3]
                else:
                    acceleration = torch.zeros((T, N, 3), dtype=landmarks.dtype)
            elif acceleration.shape[0] < T - 2:
                # Pad with zeros
                padding = torch.zeros((T - acceleration.shape[0], N, 3), dtype=acceleration.dtype)
                acceleration = torch.cat([acceleration, padding], dim=0)  # [T, N, 3]
            elif acceleration.shape[0] > T:
                # Truncate
                acceleration = acceleration[:T]
        else:
            # No acceleration: use zeros
            acceleration = torch.zeros((T, N, 3), dtype=landmarks.dtype)
        
        # Use edge index from extractor if available, otherwise build k-NN
        if 'edge_index' in sample:
            edge_index = sample['edge_index']  # [2, E] - anatomical edges from extractor
            # Verify edge_index matches number of nodes
            if edge_index.max() >= N:
                # If edge_index has more nodes than current sample, rebuild
                edge_index = self._build_edge_index(N)
        else:
            # Fallback: build k-NN edges if not in sample
            edge_index = self._build_edge_index(N)
        
        # Create node features per timestep
        # Each node gets: [x, y, z] + [vx, vy, vz] + [ax, ay, az] + broadcasted [AU, geometric]
        node_features_seq = []
        
        for t in range(T):
            # Node position: [N, 3]
            node_pos = landmarks[t]
            
            # Motion features: [N, 3] each
            node_velocity = velocity[t]  # [N, 3]
            node_acceleration = acceleration[t]  # [N, 3]
            
            # Broadcast global features to all nodes
            au_broadcast = action_units[t].unsqueeze(0).repeat(N, 1)  # [N, 18]
            geo_broadcast = geometric[t].unsqueeze(0).repeat(N, 1)  # [N, 15]
            
            # Compute advanced features if enabled
            advanced_features = []
            if self.use_advanced_features:
                # Relative motion features (2 dims: upper-lower, left-right)
                rel_motion = self._compute_relative_motion(velocity, t, N)
                rel_motion_broadcast = rel_motion.unsqueeze(0).repeat(N, 1)  # [N, 2]
                advanced_features.append(rel_motion_broadcast)
                
                # Gabor features (simplified - 3 dims per node)
                gabor_feat = self._compute_gabor_features(node_pos, N)  # [N, 3]
                advanced_features.append(gabor_feat)
            
            # Base features: [N, 3+3+3+18+15] = [N, 42]
            # With advanced: [N, 42+2+3] = [N, 47]
            base_feat = torch.cat([
                node_pos,           # [N, 3]
                node_velocity,      # [N, 3]
                node_acceleration,  # [N, 3]
                au_broadcast,       # [N, 18]
                geo_broadcast       # [N, 15]
            ], dim=1)
            
            if advanced_features:
                node_feat = torch.cat([base_feat] + advanced_features, dim=1)
            else:
                node_feat = base_feat
            
            node_features_seq.append(node_feat)
        
        # Stack: [T, N, feat_dim]
        node_features_seq = torch.stack(node_features_seq, dim=0)
        feat_dim = node_features_seq.shape[2]
        
        # Compute sequence-level advanced features if enabled
        if self.use_advanced_features:
            # FFT features (per sequence)
            fft_features = self._compute_fft_features(landmarks, velocity)  # [T, feat_dim_fft]
            
            # Recurrence plot features (per sequence)
            recurrence_features = self._compute_recurrence_features(landmarks)  # [T, feat_dim_rec]
            
            # Multi-scale temporal features
            multiscale_features = self._compute_multiscale_temporal(landmarks, velocity)  # [T, feat_dim_ms]
            
            # Broadcast sequence-level features to nodes
            fft_broadcast = fft_features.unsqueeze(2).repeat(1, 1, N).transpose(1, 2)  # [T, N, feat_dim_fft]
            rec_broadcast = recurrence_features.unsqueeze(2).repeat(1, 1, N).transpose(1, 2)  # [T, N, feat_dim_rec]
            ms_broadcast = multiscale_features.unsqueeze(2).repeat(1, 1, N).transpose(1, 2)  # [T, N, feat_dim_ms]
            
            # Concatenate to node features
            node_features_seq = torch.cat([
                node_features_seq,
                fft_broadcast,
                rec_broadcast,
                ms_broadcast
            ], dim=2)  # [T, N, feat_dim + fft + rec + ms]
        
        # Get label
        label_idx = self.label_map[sample['label']]
        
        # Create PyG Data object
        data = Data(
            x=node_features_seq[0],  # [N, feat_dim] first frame for graph structure
            edge_index=edge_index,  # [2, E]
            y=torch.tensor([label_idx], dtype=torch.long),  # [1]
            # Temporal data
            x_temporal=node_features_seq,  # [T, N, feat_dim]
            speech_mask=speech_mask,  # [T]
            num_frames=torch.tensor([T], dtype=torch.long),  # [1]
            video_id=sample['video_id'],
        )
        
        return data
    
    def _compute_relative_motion(self, velocity: torch.Tensor, t: int, N: int) -> torch.Tensor:
        """Compute relative motion between key landmarks"""
        # Key landmark indices in MediaPipe: 13 (upper lip), 14 (lower lip), 61 (left corner), 291 (right corner)
        # These need to be mapped to ROI indices - approximate using first few nodes
        # Upper-lower relative motion (vertical)
        if t < velocity.shape[0] and N > 10:
            # Approximate: use nodes in upper and lower regions
            upper_nodes = velocity[t, :N//3, 1]  # Y component of upper region
            lower_nodes = velocity[t, 2*N//3:, 1]  # Y component of lower region
            rel_vertical = upper_nodes.mean() - lower_nodes.mean() if len(upper_nodes) > 0 and len(lower_nodes) > 0 else 0.0
        else:
            rel_vertical = 0.0
        
        # Left-right relative motion (horizontal)
        if t < velocity.shape[0] and N > 10:
            left_nodes = velocity[t, :N//2, 0]  # X component of left region
            right_nodes = velocity[t, N//2:, 0]  # X component of right region
            rel_horizontal = left_nodes.mean() - right_nodes.mean() if len(left_nodes) > 0 and len(right_nodes) > 0 else 0.0
        else:
            rel_horizontal = 0.0
        
        return torch.tensor([rel_vertical, rel_horizontal], dtype=velocity.dtype)
    
    def _compute_gabor_features(self, landmarks: torch.Tensor, N: int) -> torch.Tensor:
        """Compute simplified Gabor-like features from landmark positions"""
        # Simplified: use spatial frequency analysis on landmark positions
        # Compute local spatial patterns
        if N < 3:
            return torch.zeros((N, 3), dtype=landmarks.dtype)
        
        # Feature 1: Local spatial variance (roughness)
        local_var = torch.var(landmarks, dim=0, keepdim=True).repeat(N, 1)  # [N, 3]
        
        # Feature 2: Distance to centroid
        centroid = landmarks.mean(dim=0, keepdim=True)  # [1, 3]
        dist_to_centroid = torch.norm(landmarks - centroid, dim=1, keepdim=True)  # [N, 1]
        
        # Feature 3: Local gradient magnitude (spatial change)
        if N > 1:
            gradients = torch.diff(landmarks, dim=0)  # [N-1, 3]
            grad_mag = torch.norm(gradients, dim=1)  # [N-1]
            # Pad to match N
            grad_mag = torch.cat([grad_mag, grad_mag[-1:]], dim=0)  # [N]
            grad_mag = grad_mag.unsqueeze(1)  # [N, 1]
        else:
            grad_mag = torch.zeros((N, 1), dtype=landmarks.dtype)
        
        # Combine: [N, 3]
        gabor_feat = torch.cat([
            local_var[:, :1],  # X variance
            dist_to_centroid,  # Distance to center
            grad_mag  # Gradient magnitude
        ], dim=1)
        
        # Normalize
        gabor_feat = (gabor_feat - gabor_feat.min()) / (gabor_feat.max() - gabor_feat.min() + 1e-6)
        
        return gabor_feat
    
    def _compute_fft_features(self, landmarks: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Compute FFT features from landmark trajectories"""
        T, N, _ = landmarks.shape
        
        if T < 4:
            return torch.zeros((T, 3), dtype=landmarks.dtype)
        
        # Compute FFT for key trajectories
        # 1. Mouth opening (vertical distance between upper and lower regions)
        # Use equal-sized regions to avoid dimension mismatch
        upper_size = N // 3
        lower_start = N - upper_size
        upper_region = landmarks[:, :upper_size, :]  # [T, upper_size, 3]
        lower_region = landmarks[:, lower_start:, :]  # [T, upper_size, 3]
        mouth_opening = torch.norm(upper_region - lower_region, dim=2).mean(dim=1)  # [T]
        
        # 2. Mouth width (horizontal span)
        mouth_width = landmarks[:, :, 0].max(dim=1)[0] - landmarks[:, :, 0].min(dim=1)[0]  # [T]
        
        # Compute FFT
        fft_opening = torch.fft.fft(mouth_opening.float(), n=T)
        fft_width = torch.fft.fft(mouth_width.float(), n=T)
        
        # Extract dominant frequency and energy
        fft_size = T // 2 + 1
        fft_opening_mag = torch.abs(fft_opening[:fft_size])
        fft_width_mag = torch.abs(fft_width[:fft_size])
        
        # Dominant frequency index
        dom_freq_opening = torch.argmax(fft_opening_mag)
        dom_freq_width = torch.argmax(fft_width_mag)
        
        # Energy in low frequencies
        low_freq_size = max(1, T // 4)
        low_freq_energy = fft_opening_mag[:low_freq_size].sum()
        total_energy = fft_opening_mag.sum() + 1e-6
        
        # Normalize values
        dom_freq_opening_norm = dom_freq_opening.float() / max(1, fft_size - 1)
        dom_freq_width_norm = dom_freq_width.float() / max(1, fft_size - 1)
        low_freq_energy_norm = low_freq_energy / total_energy
        
        # Broadcast to all frames: [T, 3]
        fft_features = torch.stack([
            dom_freq_opening_norm.repeat(T),
            dom_freq_width_norm.repeat(T),
            low_freq_energy_norm.repeat(T)
        ], dim=1)
        
        return fft_features.to(landmarks.dtype)
    
    def _compute_recurrence_features(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute recurrence plot features"""
        T, N, _ = landmarks.shape
        
        if T < 3:
            return torch.zeros((T, 3), dtype=landmarks.dtype)
        
        # Flatten landmarks: [T, N*3]
        landmarks_flat = landmarks.reshape(T, -1)
        
        # Compute pairwise distances
        distances = torch.cdist(landmarks_flat, landmarks_flat)  # [T, T]
        
        # Recurrence features: diagonal recurrence, vertical recurrence, horizontal recurrence
        # Diagonal recurrence (self-similarity)
        diag_recurrence = torch.diagonal(distances, offset=0).mean()
        
        # Vertical recurrence (temporal similarity)
        vert_recurrence = distances.mean(dim=1)  # [T]
        
        # Horizontal recurrence (spatial similarity)
        horiz_recurrence = distances.mean(dim=0)  # [T]
        
        # Normalize
        diag_norm = diag_recurrence / (distances.max() + 1e-6)
        vert_norm = vert_recurrence / (vert_recurrence.max() + 1e-6)
        horiz_norm = horiz_recurrence / (horiz_recurrence.max() + 1e-6)
        
        # [T, 3]
        recurrence_features = torch.stack([
            diag_norm.repeat(T),
            vert_norm,
            horiz_norm
        ], dim=1)
        
        return recurrence_features
    
    def _compute_multiscale_temporal(self, landmarks: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale temporal features"""
        T, N, _ = landmarks.shape
        
        if T < 3:
            return torch.zeros((T, 3), dtype=landmarks.dtype)
        
        # Short-term (2-3 frames)
        short_term = []
        for t in range(T):
            if t >= 2:
                short_term.append(torch.norm(landmarks[t] - landmarks[t-2], dim=1).mean())
            else:
                short_term.append(torch.tensor(0.0, dtype=landmarks.dtype))
        short_term = torch.stack(short_term)  # [T]
        
        # Medium-term (5-7 frames)
        medium_term = []
        for t in range(T):
            if t >= 5:
                medium_term.append(torch.norm(landmarks[t] - landmarks[t-5], dim=1).mean())
            else:
                medium_term.append(torch.tensor(0.0, dtype=landmarks.dtype))
        medium_term = torch.stack(medium_term)  # [T]
        
        # Long-term (entire sequence mean)
        long_term = torch.norm(landmarks - landmarks.mean(dim=0, keepdim=True), dim=2).mean(dim=1)  # [T]
        
        # Normalize
        short_norm = short_term / (short_term.max() + 1e-6)
        medium_norm = medium_term / (medium_term.max() + 1e-6)
        long_norm = long_term / (long_term.max() + 1e-6)
        
        # [T, 3]
        multiscale_features = torch.stack([short_norm, medium_norm, long_norm], dim=1)
        
        return multiscale_features
    
    def _compute_inner_lip_features(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Compute inner lip features from landmarks (for backward compatibility)"""
        T, N, _ = landmarks.shape
        
        # Approximate inner lip features using landmark regions
        # Use equal-sized regions to avoid dimension mismatch
        upper_size = N // 3
        lower_start = N - upper_size
        upper_region = landmarks[:, :upper_size, :]  # [T, upper_size, 3]
        lower_region = landmarks[:, lower_start:, :]  # [T, upper_size, 3]
        left_region = landmarks[:, :N//2, :]  # [T, N//2, 3]
        right_region = landmarks[:, N//2:, :]  # [T, N//2, 3]
        
        # Inner lip height (vertical distance between upper and lower regions)
        inner_height = torch.norm(upper_region.mean(dim=1) - lower_region.mean(dim=1), dim=1)  # [T]
        
        # Inner lip width (horizontal distance between left and right regions)
        inner_width = torch.norm(left_region.mean(dim=1) - right_region.mean(dim=1), dim=1)  # [T]
        
        # Inner lip area
        inner_area = inner_width * inner_height  # [T]
        
        # Outer dimensions for ratios
        outer_width = landmarks[:, :, 0].max(dim=1)[0] - landmarks[:, :, 0].min(dim=1)[0]  # [T]
        outer_height = inner_height  # Use same as inner for simplicity
        
        # Ratios
        inner_outer_width_ratio = inner_width / (outer_width + 1e-6)
        inner_outer_height_ratio = inner_height / (outer_height + 1e-6)
        
        # Stack: [T, 5]
        inner_features = torch.stack([
            inner_height,
            inner_width,
            inner_area,
            inner_outer_width_ratio,
            inner_outer_height_ratio
        ], dim=1)
        
        # Normalize to [0, 1] per feature
        for i in range(inner_features.shape[1]):
            feat_min = inner_features[:, i].min()
            feat_max = inner_features[:, i].max()
            if feat_max > feat_min:
                inner_features[:, i] = (inner_features[:, i] - feat_min) / (feat_max - feat_min + 1e-6)
        
        return inner_features
    
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