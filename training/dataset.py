"""
Dataset and DataLoader for lip reading.
"""
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
from collections import Counter
import numpy as np


class LipReadingDataset(Dataset):
    """Dataset for lip reading from precomputed features.
    Each B level file contains cumulative features (no concatenation needed, 3D coordinates):
    - B0 file: B0 features only (3 features: X, Y, Z - 3D normalized coordinates)
    - B1 file: B0+B1 features (10 features: B0: 3 + B1: 7 = vx, vy, vz, speed, ax, ay, az - all 3D)
    - B2 file: B0+B1+B2 features (18 features: B0: 3 + B1: 7 + B2: 8 = MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening)
    - B3 file: B0+B1+B2+B3 features (22 features: B0: 3 + B1: 7 + B2: 8 + B3: 4 = AU25, AU26, AU12, AU27)
    Uses lazy loading to avoid loading entire file into memory at once.
    """
    
    def __init__(
        self,
        feature_file: str,
        transform: Optional[callable] = None,
        lazy_load: bool = True,
        feature_level: Optional[str] = None,
        feature_dir: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            feature_file: Path to feature .pt file (for backward compatibility)
            transform: Optional transform function
            lazy_load: If True, load videos on-demand. If False, load all at once.
            feature_level: Target feature level (B0, B1, B2, B3). If provided, loads incrementally.
            feature_dir: Base feature directory (e.g., 'data/features'). Required if feature_level is provided.
        """
        self.transform = transform
        self.lazy_load = lazy_load
        
        # Determine feature file to load (cumulative approach)
        if feature_level is not None and feature_dir is not None:
            # Cumulative loading mode: each B level file contains all features up to that level
            # B0 file has B0, B1 file has B0+B1, B2 file has B0+B1+B2, etc.
            self.feature_level = feature_level
            self.feature_dir = Path(feature_dir)
            
            # Load from single file (cumulative - no concatenation needed)
            self.feature_file = str(self.feature_dir / feature_level / Path(feature_file).name)
            if not Path(self.feature_file).exists():
                raise FileNotFoundError(f"Feature file not found: {self.feature_file}")
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data = torch.load(self.feature_file, map_location='cpu', weights_only=False)
            
            # No longer needed - single file contains all
            self.levels_to_load = None
            self.feature_files = None
        else:
            # Original mode: load single file
            self.feature_file = feature_file
            self.feature_level = None
            self.feature_files = None
            self.levels_to_load = None
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data = torch.load(feature_file, map_location='cpu', weights_only=False)
            self.feature_level = data.get('feature_level', 'B0')
        
        # Extract metadata
        self.split = data['split']
        self.partition = data['partition']
        self.adjacency = data['adjacency']
        self.word_to_label = data['word_to_label']
        
        # Build video list (just IDs, not data)
        self.video_ids = list(data['videos'].keys())
        
        # Build video_to_word mapping for class balancing
        # Word is stored directly in video data
        self.video_to_word = {}
        for vid_id, vid_data in data['videos'].items():
            if 'word' in vid_data:
                self.video_to_word[vid_id] = vid_data['word']
            else:
                # Fallback: find word from label
                label = vid_data['label']
                for word, word_label in self.word_to_label.items():
                    if word_label == label:
                        self.video_to_word[vid_id] = word
                        break
        
        if lazy_load:
            # Store file path and load videos on-demand
            self._data_cache = None
            self.videos = {}  # Empty dict - will load on-demand
        else:
            # Load all videos at once
            self._load_all_data()
    
    def _load_all_data(self):
        """Load all video data (for non-lazy mode)."""
        # Load single file (cumulative approach - no concatenation needed)
        # Suppress FutureWarning about weights_only
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._data_cache = torch.load(self.feature_file, map_location='cpu')
        self.videos = self._data_cache['videos']
    
    def _load_data(self):
        """Lazy load the full data file if not already loaded."""
        if self._data_cache is None:
            self._load_all_data()
    
    def _load_video_features(self, video_id: int) -> Dict:
        """
        Load features for a single video on-demand (true lazy loading).
        Loads from single file (cumulative approach - no concatenation needed).
        
        Each B level file contains cumulative features:
        - B0 file: B0 features only
        - B1 file: B0+B1 features
        - B2 file: B0+B1+B2 features
        - B3 file: B0+B1+B2+B3 features
        
        Features are kept as float16 in cache to save memory, converted to float32 in __getitem__.
        """
        # Load from single file (cumulative - no concatenation needed)
        if self._data_cache is None:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._data_cache = torch.load(self.feature_file, map_location='cpu', weights_only=False)
        video_data = self._data_cache['videos'][video_id].copy()
        # Keep as float16 in cache to save memory - convert to float32 in __getitem__
        return video_data
    
    def __len__(self) -> int:
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (features, speech_mask, adjacency, label)
        """
        video_id = self.video_ids[idx]
        
        # True lazy loading: load video on-demand
        if self.lazy_load:
            if not hasattr(self, '_file_cache'):
                self._file_cache = {}  # Cache loaded files to avoid reloading
            video_data = self._load_video_features(video_id)
        else:
            # Non-lazy: use pre-loaded data
            if self._data_cache is None:
                self._load_data()
            video_data = self.videos[video_id]
            # Features kept as float16 in cache to save memory - converted to float32 in __getitem__
        
        # Get features and label
        # Keep as float16 to save memory - will convert to float32 in collate_fn right before batching
        features = video_data['features']  # (frames, nodes, features) - kept as float16
        speech_mask = video_data['speech_mask']  # (frames,)
        label = video_data['label']  # int
        
        # Apply transform if provided (transform should handle float16 and return both features and speech_mask)
        if self.transform:
            # Transform may return (features, speech_mask) or just features
            result = self.transform(features, speech_mask)
            if isinstance(result, tuple) and len(result) == 2:
                features, speech_mask = result
            else:
                # Backward compatibility: transform only returns features
                features = result
        
        return features, speech_mask, self.adjacency, label
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.word_to_label)
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        # For lazy loading, load just one video to get dimensions
        if self.lazy_load:
            if not hasattr(self, '_file_cache'):
                self._file_cache = {}
            sample_video = self._load_video_features(self.video_ids[0])
            return sample_video['features'].shape[-1]
        else:
            if self._data_cache is None:
                self._load_data()
            sample_features = self.videos[self.video_ids[0]]['features']
            return sample_features.shape[-1]
    
    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        # For lazy loading, load just one video to get dimensions
        if self.lazy_load:
            if not hasattr(self, '_file_cache'):
                self._file_cache = {}
            sample_video = self._load_video_features(self.video_ids[0])
            return sample_video['features'].shape[1]
        else:
            if self._data_cache is None:
                self._load_data()
            sample_features = self.videos[self.video_ids[0]]['features']
            return sample_features.shape[1]


def collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader with variable-length sequences.
    Converts float16 to float32 here (right before batching) to minimize memory usage.
    Features are kept as float16 in dataset cache and __getitem__ to save memory.
    Only convert to float32 at the last moment for training accuracy.
    
    Args:
        batch: List of (features, speech_mask, adjacency, label)
        
    Returns:
        Tuple of (batched_features, batched_masks, adjacency, labels)
    """
    features_list, masks_list, adj_list, labels_list = zip(*batch)
    
    # Get max sequence length
    max_frames = max(f.shape[0] for f in features_list)
    num_nodes = features_list[0].shape[1]
    feature_dim = features_list[0].shape[2]
    
    # Pad sequences (create as float32 directly for training accuracy)
    batch_size = len(features_list)
    padded_features = torch.zeros(batch_size, max_frames, num_nodes, feature_dim, dtype=torch.float32)
    padded_masks = torch.zeros(batch_size, max_frames, dtype=torch.float32)
    
    for i, (feat, mask) in enumerate(zip(features_list, masks_list)):
        seq_len = feat.shape[0]
        # Convert float16 to float32 right before batching (minimizes memory usage)
        # Features are kept as float16 everywhere else to save memory
        if feat.dtype == torch.float16:
            feat = feat.float()  # Convert to float32 for training accuracy
        padded_features[i, :seq_len] = feat
        padded_masks[i, :seq_len] = mask
    
    # Adjacency is same for all samples
    adjacency = adj_list[0]
    
    # Labels
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    return padded_features, padded_masks, adjacency, labels


def create_balanced_sampler(
    dataset: LipReadingDataset,
    balance_factor: float = 1.0
) -> Optional[WeightedRandomSampler]:
    """
    Create WeightedRandomSampler to oversample minority classes only.
    Majority classes keep their original frequency (not reduced).
    
    Args:
        dataset: Dataset instance
        balance_factor: Balance factor (1.0 = full oversample to max, 0.5 = half oversample, etc.)
                       Controls how much to oversample minority classes.
                       Example: 0.5 means oversample minorities to max_count * 0.5
        
    Returns:
        WeightedRandomSampler or None if balancing not needed
    """
    # Count class frequencies
    class_counts = Counter()
    for video_id in dataset.video_ids:
        word = dataset.video_to_word.get(video_id)
        if word:
            label = dataset.word_to_label[word]
            class_counts[label] += 1
    
    if not class_counts:
        return None
    
    max_count = max(class_counts.values())
    num_classes = len(dataset.word_to_label)
    
    # Check if balancing is needed (if all classes have same count, no need)
    if len(set(class_counts.values())) == 1:
        return None
    
    # Create weights:
    # - Majority classes (count == max_count): weight = 1.0 (keep original frequency)
    # - Minority classes (count < max_count): weight = max_count / count (oversample)
    weights = []
    for idx in range(len(dataset.video_ids)):
        video_id = dataset.video_ids[idx]
        word = dataset.video_to_word.get(video_id)
        if word:
            label = dataset.word_to_label[word]
            count = class_counts[label]
            if count == max_count:
                # Majority class: keep original frequency (weight = 1.0)
                weight = 1.0
            else:
                # Minority class: oversample to match max_count
                # Apply balance_factor to control oversampling amount
                target_count = int(max_count * balance_factor)
                if target_count > count:
                    weight = target_count / count if count > 0 else 1.0
                else:
                    weight = 1.0  # Don't undersample if balance_factor is too low
            weights.append(weight)
        else:
            weights.append(1.0)
    
    # Calculate num_samples:
    # Original size (all classes at original frequency) + oversampled minorities
    original_size = len(dataset.video_ids)
    
    # Calculate additional samples needed for minority classes
    additional_samples = 0
    target_count = int(max_count * balance_factor)
    for label, count in class_counts.items():
        if count < max_count and target_count > count:
            # This minority class needs oversampling
            additional_samples += (target_count - count)
    
    num_samples = original_size + additional_samples
    
    # Ensure num_samples is at least original_size
    if num_samples < original_size:
        num_samples = original_size
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True  # Allow sampling same sample multiple times
    )
    
    return sampler


def get_dataloader(
    feature_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    feature_level: Optional[str] = None,
    feature_dir: Optional[str] = None,
    transform: Optional[callable] = None,
    balance_classes: bool = False,
    balance_factor: float = 1.0
) -> DataLoader:
    """
    Get DataLoader for feature file.
    
    Args:
        feature_file: Path to feature .pt file (filename only, e.g., 'full_train.pt')
        batch_size: Batch size
        shuffle: Whether to shuffle data (ignored if balance_classes=True)
        num_workers: Number of workers (default: 50% of CPU cores)
        pin_memory: Pin memory for faster GPU transfer
        feature_level: Target feature level (B0, B1, B2, B3). If provided, loads incrementally.
        feature_dir: Base feature directory (e.g., 'data/features'). Required if feature_level is provided.
        transform: Optional transform/augmentation function
        balance_classes: If True, use WeightedRandomSampler to balance classes to max count
        balance_factor: Balance factor (1.0 = full balance to max, 0.5 = half balance)
                        Reduces training time. Default: 1.0 (full balance)
        
    Returns:
        DataLoader instance
    """
    dataset = LipReadingDataset(
        feature_file,
        feature_level=feature_level,
        feature_dir=feature_dir,
        transform=transform
    )
    
    # Create balanced sampler if requested
    sampler = None
    if balance_classes:
        sampler = create_balanced_sampler(dataset, balance_factor=balance_factor)
        if sampler is not None:
            shuffle = False  # Can't shuffle when using sampler
    
    # Limit CPU usage and reduce memory: use fewer workers or 0 for memory-constrained systems
    # With num_workers > 0, each worker loads its own copy of data (multiplies memory usage)
    if num_workers is None:
        total_cores = os.cpu_count() or 1
        # Use 0 workers for memory-constrained systems, or 1-2 workers max
        # Set to 0 to avoid memory duplication across workers
        num_workers = 0  # Changed from 50% to 0 to reduce memory usage
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory if num_workers == 0 else False,  # pin_memory only useful with workers
        persistent_workers=False  # Don't keep workers alive (saves memory)
    )
    
    return dataloader

