"""
Dataset and DataLoader for lip reading.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os


class LipReadingDataset(Dataset):
    """Dataset for lip reading from precomputed features.
    Supports incremental loading: B0 loads B0 only, B1 loads B0+B1, B2 loads B0+B1+B2, etc.
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
        
        # Determine feature levels to load
        if feature_level is not None and feature_dir is not None:
            # Incremental loading mode
            self.feature_level = feature_level
            self.feature_dir = Path(feature_dir)
            
            # Determine which levels to load: B0, B0+B1, B0+B1+B2, etc.
            # B0 -> [B0], B1 -> [B0, B1], B2 -> [B0, B1, B2], etc.
            level_map = {'B0': 0, 'B1': 1, 'B2': 2, 'B3': 3}
            target_level = level_map.get(feature_level, 0)
            self.levels_to_load = [f'B{i}' for i in range(target_level + 1)]
            
            # Load metadata from B0 file (all files should have same metadata)
            b0_file = self.feature_dir / 'B0' / Path(feature_file).name
            if not b0_file.exists():
                raise FileNotFoundError(f"B0 file not found: {b0_file}")
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data = torch.load(b0_file, map_location='cpu')
            
            # Store file paths for all levels
            self.feature_files = {}
            for level in self.levels_to_load:
                level_file = self.feature_dir / level / Path(feature_file).name
                if not level_file.exists():
                    raise FileNotFoundError(f"Feature file not found: {level_file}")
                self.feature_files[level] = str(level_file)
            
            self.feature_file = feature_file  # Keep for backward compatibility
        else:
            # Original mode: load single file
            self.feature_file = feature_file
            self.feature_level = None
            self.feature_files = None
            self.levels_to_load = None
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data = torch.load(feature_file, map_location='cpu')
            self.feature_level = data.get('feature_level', 'B0')
        
        # Extract metadata
        self.split = data['split']
        self.partition = data['partition']
        self.adjacency = data['adjacency']
        self.word_to_label = data['word_to_label']
        
        # Build video list (just IDs, not data)
        self.video_ids = list(data['videos'].keys())
        
        if lazy_load:
            # Store file path(s) and load videos on-demand
            self._data_cache = None
            self._file_cache = {}  # Cache for loaded feature files (per level)
            self.videos = {}  # Empty dict - will load on-demand
        else:
            # Load all videos at once
            self._load_all_data()
    
    def _load_all_data(self):
        """Load all video data (for non-lazy mode)."""
        if self.levels_to_load:
            # Incremental loading: load and concatenate features from multiple levels
            all_videos = {}
            
            # Load each level (silent - no progress logging)
            # Suppress FutureWarning about weights_only
            import warnings
            level_data = {}
            for level in self.levels_to_load:
                level_file = self.feature_files[level]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    level_data[level] = torch.load(level_file, map_location='cpu')
            
            # Concatenate features for each video
            for video_id in self.video_ids:
                features_list = []
                for level in self.levels_to_load:
                    if video_id in level_data[level]['videos']:
                        # Keep as float16 in cache to save memory - convert to float32 in __getitem__
                        features = level_data[level]['videos'][video_id]['features']
                        features_list.append(features)
                
                if features_list:
                    # Concatenate features along feature dimension
                    concatenated_features = torch.cat(features_list, dim=-1)
                    
                    # Use metadata from first level (B0)
                    video_data = level_data[self.levels_to_load[0]]['videos'][video_id].copy()
                    video_data['features'] = concatenated_features
                    all_videos[video_id] = video_data
            
            self.videos = all_videos
            self._data_cache = level_data[self.levels_to_load[0]]  # Store B0 as cache
        else:
            # Original mode: load single file (silent - no progress logging)
            self._data_cache = torch.load(self.feature_file, map_location='cpu')
            self.videos = self._data_cache['videos']
    
    def _load_data(self):
        """Lazy load the full data file if not already loaded."""
        if self._data_cache is None:
            self._load_all_data()
    
    def _load_video_features(self, video_id: int) -> Dict:
        """
        Load features for a single video on-demand (true lazy loading).
        Loads each feature level file once and caches it, but only concatenates per video.
        
        NOTE: This still loads entire feature files into memory (all videos in split).
        For B3, this means 4 files (B0, B1, B2, B3) each containing all videos.
        Memory usage: ~2-3GB per feature file × 4 files = ~8-12GB for file cache (float16).
        Features are kept as float16 in cache to save memory, converted to float32 in __getitem__.
        """
        if self.levels_to_load:
            # Incremental loading: load video from each level and concatenate
            features_list = []
            video_data = None
            
            for level in self.levels_to_load:
                level_file = self.feature_files[level]
                
                # Load file into cache if not already loaded
                # Note: torch.load loads entire file, but we cache it to avoid reloading
                # Suppress FutureWarning about weights_only
                if level not in self._file_cache:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        self._file_cache[level] = torch.load(level_file, map_location='cpu')
                
                level_data = self._file_cache[level]
                if video_id in level_data['videos']:
                    if video_data is None:
                        # Use metadata from first level (B0)
                        video_data = level_data['videos'][video_id].copy()
                    # Extract features for this video
                    # Keep as float16 in cache to save memory - convert to float32 in __getitem__
                    features = level_data['videos'][video_id]['features']
                    features_list.append(features)
            
            if features_list:
                # Concatenate features along feature dimension
                # This creates a new tensor, but only for this one video
                concatenated_features = torch.cat(features_list, dim=-1)
                video_data['features'] = concatenated_features
            
            return video_data
        else:
            # Original mode: load from single file
            if self._data_cache is None:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    self._data_cache = torch.load(self.feature_file, map_location='cpu')
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
        
        # Apply transform if provided (transform should handle float16)
        if self.transform:
            features = self.transform(features)
        
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


def get_dataloader(
    feature_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    feature_level: Optional[str] = None,
    feature_dir: Optional[str] = None
) -> DataLoader:
    """
    Get DataLoader for feature file.
    
    Args:
        feature_file: Path to feature .pt file (filename only, e.g., 'full_train.pt')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers (default: 50% of CPU cores)
        pin_memory: Pin memory for faster GPU transfer
        feature_level: Target feature level (B0, B1, B2, B3). If provided, loads incrementally.
        feature_dir: Base feature directory (e.g., 'data/features'). Required if feature_level is provided.
        
    Returns:
        DataLoader instance
    """
    dataset = LipReadingDataset(
        feature_file,
        feature_level=feature_level,
        feature_dir=feature_dir
    )
    
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
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory if num_workers == 0 else False,  # pin_memory only useful with workers
        persistent_workers=False  # Don't keep workers alive (saves memory)
    )
    
    return dataloader

