# IDLRW Dataset handler for preprocessing
import os
from pathlib import Path
import cv2
import json
from tqdm import tqdm

class IDLRWDatasetHandler:
    def __init__(self, dataset_root):
        """
        Initialize IDLRW dataset handler
        Args:
            dataset_root (str): Path to IDLRW-DATASET folder
        """
        self.dataset_root = Path(dataset_root)
        self.words = self._get_word_list()
        
    def _get_word_list(self):
        """Get list of all word classes in dataset."""
        return [d.name for d in self.dataset_root.iterdir() 
                if d.is_dir() and not d.name.startswith('.')]
    
    def get_video_metadata(self, txt_path):
        """
        Parse metadata from .txt file
        Args:
            txt_path (Path): Path to metadata text file
        Returns:
            dict: Metadata dictionary
        """
        metadata = {}
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                    
        # Parse time information
        if 'Start' in metadata:
            start, end = metadata['Start'].split('End:')
            metadata['Start'] = float(start)
            metadata['End'] = float(end)
        
        if 'Duration' in metadata:
            metadata['Duration'] = float(metadata['Duration'].split()[0])
            
        return metadata
    
    def process_word_folder(self, word, output_dir=None, split=None):
        """
        Process all videos for a specific word
        Args:
            word (str): Word folder name
            output_dir (Path, optional): Output directory for processed data
            split (str, optional): Data split (train/val/test) if using splits
        """
        word_path = self.dataset_root / word
        if not word_path.exists():
            raise ValueError(f"Word folder {word} not found")
            
        # Create output directory structure if needed
        if output_dir:
            output_path = Path(output_dir)
            if split:
                output_path = output_path / split / word
            else:
                output_path = output_path / word
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all video-metadata pairs
        video_files = list(word_path.glob('*.mp4'))
        meta_files = list(word_path.glob('*.txt'))
        
        # Match video files with their metadata
        pairs = []
        for video_file in video_files:
            meta_file = word_path / f"{video_file.stem}.txt"
            if meta_file in meta_files:
                pairs.append((video_file, meta_file))
                
        return pairs
    
    def get_dataset_stats(self):
        """
        Get statistics about the dataset
        Returns:
            dict: Dataset statistics
        """
        stats = {
            'total_words': len(self.words),
            'words': {}
        }
        
        for word in tqdm(self.words, desc="Analyzing dataset"):
            word_path = self.dataset_root / word
            videos = list(word_path.glob('*.mp4'))
            metadata_files = list(word_path.glob('*.txt'))
            
            stats['words'][word] = {
                'num_videos': len(videos),
                'num_metadata': len(metadata_files),
                'has_split': (word_path / 'train').exists()
            }
            
        return stats
    
    def create_data_splits(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Create train/val/test splits for the dataset
        Args:
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            seed (int): Random seed for reproducibility
        """
        import numpy as np
        np.random.seed(seed)
        
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for word in self.words:
            word_path = self.dataset_root / word
            video_files = list(word_path.glob('*.mp4'))
            np.random.shuffle(video_files)
            
            n_total = len(video_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            splits['train'].extend([(f, word) for f in video_files[:n_train]])
            splits['val'].extend([(f, word) for f in video_files[n_train:n_train + n_val]])
            splits['test'].extend([(f, word) for f in video_files[n_train + n_val:]])
            
        return splits