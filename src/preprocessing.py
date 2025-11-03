# Main preprocessing script for Indonesian Lip Reading Dataset
import os
import sys
import argparse
import cv2
import torch
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.preprocessing.video_processor import VideoProcessor

def process_dataset(data_dir, output_dir, skip_processed=True):
    """Process all videos in the dataset directory."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Initialize video processor
    processor = VideoProcessor()
    
    # Get all word folders (ada, apa, etc.)
    word_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for word_folder in tqdm(word_folders, desc="Processing words"):
        # Create output word folder
        word_output_dir = output_dir / word_folder.name
        word_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos in the word folder
        video_files = list(word_folder.glob("**/*.mp4"))
        
        for video_path in tqdm(video_files, desc=f"Processing {word_folder.name}", leave=False):
            try:
                # Define output path for processed data
                relative_path = video_path.relative_to(word_folder)
                output_path = word_output_dir / relative_path.with_suffix('.pt')
                
                # Skip if already processed
                if skip_processed and output_path.exists():
                    continue
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process video into graph sequence
                graphs = processor.process_video(str(video_path))
                
                if graphs and len(graphs) > 0:
                    # Save processed data
                    torch.save(graphs, output_path)
                else:
                    print(f"Warning: No valid graphs generated for {video_path}")
                    
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")

def create_dataset_splits(processed_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Create train/val/test splits and save in organized structure."""
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    
    # Create split directories
    splits = ['train', 'val', 'test']
    split_dirs = {split: output_dir / split for split in splits}
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Process each word class
    for word_dir in processed_dir.iterdir():
        if not word_dir.is_dir():
            continue
            
        # Get all processed files for this word
        processed_files = list(word_dir.glob('**/*.pt'))
        if not processed_files:
            continue
            
        # Shuffle files
        np.random.shuffle(processed_files)
        
        # Calculate split indices
        n_files = len(processed_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        # Split files
        train_files = processed_files[:n_train]
        val_files = processed_files[n_train:n_train + n_val]
        test_files = processed_files[n_train + n_val:]
        
        # Save files to respective splits
        for files, split in zip([train_files, val_files, test_files], splits):
            if not files:
                continue
                
            # Create word directory in split
            split_word_dir = split_dirs[split] / word_dir.name
            split_word_dir.mkdir(exist_ok=True)
            
            # Copy files to split directory
            for src_path in files:
                dst_path = split_word_dir / src_path.name
                torch.save(torch.load(src_path), dst_path)

def main():
    parser = argparse.ArgumentParser(description='Preprocess Indonesian Lip Reading Dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory of the raw dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed data')
    parser.add_argument('--processed_dir', type=str,
                       help='Directory containing processed .pt files (for split creation)')
    parser.add_argument('--skip_processed', action='store_true',
                       help='Skip already processed videos')
    parser.add_argument('--create_splits', action='store_true',
                       help='Create train/val/test splits')
    args = parser.parse_args()
    
    if not args.create_splits:
        # Process raw videos into graph sequences
        process_dataset(args.data_dir, args.output_dir, args.skip_processed)
    else:
        # Create dataset splits
        if not args.processed_dir:
            print("Error: --processed_dir required for creating splits")
            return
        create_dataset_splits(args.processed_dir, args.output_dir)

if __name__ == '__main__':
    main()