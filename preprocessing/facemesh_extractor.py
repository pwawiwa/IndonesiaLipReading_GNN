"""
FaceMesh Landmark Extractor with SpeechMask Generation.

This script extracts facial landmarks from videos using MediaPipe FaceMesh,
generates speech masks from metadata, and saves aggregated .pt files per split.

Usage:
    python facemesh_extractor.py --partition lips --split train --out-dir data/extracted
"""
import argparse
import cv2
import mediapipe as mp
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, ensure_dir
from utils.meta_parser import parse_video_meta, generate_speech_mask
from preprocessing.mediapipe_nodes import get_partition_nodes, build_partition_adjacency


class FaceMeshExtractor:
    """Extract MediaPipe FaceMesh landmarks from videos."""
    
    def __init__(
        self,
        partition: str = 'lips',
        fps: int = 25,
        max_workers: Optional[int] = None,
        logger=None
    ):
        """
        Initialize extractor.
        
        Args:
            partition: 'lips', 'mouth', or 'full'
            fps: Target frames per second
            max_workers: Max parallel workers (default: 50% of CPU cores)
            logger: Logger instance
        """
        self.partition = partition
        self.fps = fps
        self.logger = logger or setup_logger('FaceMeshExtractor')
        
        # Limit CPU usage to ~50%
        if max_workers is None:
            total_cores = os.cpu_count() or 1
            self.max_workers = max(1, int(total_cores * 0.5))
        else:
            self.max_workers = max_workers
        
        self.logger.info(f"Initializing extractor with {self.max_workers} workers")
        
        # Get partition node indices
        self.nodes = get_partition_nodes(partition)
        self.n_nodes = len(self.nodes)
        
        # Build adjacency matrix and node mapping
        self.adjacency, self.node_mapping = build_partition_adjacency(partition)
        
        self.logger.info(f"Partition: {partition}")
        self.logger.info(f"Nodes: {self.n_nodes}")
        self.logger.info(f"Edges: {int((self.adjacency.sum() - self.adjacency.trace()).item()) // 2}")
    
    def extract_video_landmarks(
        self,
        video_path: Path,
        meta_path: Path
    ) -> Optional[Dict]:
        """
        Extract landmarks from a single video.
        
        Args:
            video_path: Path to video file
            meta_path: Path to metadata file
            
        Returns:
            Dictionary with extracted data or None if failed
        """
        try:
            # Parse metadata (without mask first - we need video duration)
            meta = parse_video_meta(str(meta_path))
            
            # Open video to get actual duration
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.warning(f"Failed to open video: {video_path}")
                return None
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate actual video duration (important: Duration field in meta.txt is speech segment duration, not video duration)
            video_duration = frame_count / original_fps if original_fps > 0 else 0
            
            # Generate speech mask with actual video duration
            speech_mask = generate_speech_mask(
                start=meta['start'],
                end=meta['end'],
                video_duration=video_duration,
                fps=self.fps
            )
            meta['speech_mask'] = speech_mask
            meta['num_frames'] = len(speech_mask)
            
            # Calculate frame sampling rate
            frame_sample_rate = original_fps / self.fps if original_fps > 0 else 1.0
            
            # Initialize MediaPipe FaceMesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            landmarks_list = []
            frame_idx = 0
            sample_frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at target FPS
                if frame_idx >= sample_frame_idx * frame_sample_rate:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with FaceMesh
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        # Get first face
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Extract coordinates for partition nodes
                        coords = np.zeros((self.n_nodes, 2), dtype=np.float32)
                        
                        for new_idx, orig_idx in enumerate(self.nodes):
                            landmark = face_landmarks.landmark[orig_idx]
                            # Normalize by resolution: convert to [0, 1] range
                            coords[new_idx, 0] = landmark.x
                            coords[new_idx, 1] = landmark.y
                        
                        landmarks_list.append(coords)
                    else:
                        # No face detected - use zeros
                        coords = np.zeros((self.n_nodes, 2), dtype=np.float32)
                        landmarks_list.append(coords)
                    
                    sample_frame_idx += 1
                
                frame_idx += 1
            
            cap.release()
            face_mesh.close()
            
            # Convert to tensor
            landmarks_tensor = torch.from_numpy(np.stack(landmarks_list, axis=0))
            
            # Verify frame count matches speech mask
            expected_frames = meta['num_frames']
            actual_frames = landmarks_tensor.shape[0]
            
            if actual_frames != expected_frames:
                # Adjust speech mask if needed
                if actual_frames < expected_frames:
                    self.logger.debug(
                        f"{video_path.name}: Truncating speech mask from {expected_frames} to {actual_frames} frames"
                    )
                    meta['speech_mask'] = meta['speech_mask'][:actual_frames]
                else:
                    # Pad speech mask with zeros for non-speech frames
                    self.logger.debug(
                        f"{video_path.name}: Video has {actual_frames} frames, speech duration is {expected_frames} frames. "
                        f"Padding speech mask with {actual_frames - expected_frames} zero frames."
                    )
                    padding = torch.zeros(actual_frames - expected_frames, dtype=torch.float32)
                    meta['speech_mask'] = torch.cat([meta['speech_mask'], padding])
                
                meta['num_frames'] = actual_frames
            
            # Extract word label from path
            word = video_path.parent.parent.name
            
            return {
                'video_path': str(video_path),
                'word': word,
                'landmarks': landmarks_tensor,  # Shape: (frames, n_nodes, 2)
                'speech_mask': meta['speech_mask'],  # Shape: (frames,)
                'meta': {
                    'vid_id': meta.get('vid_id', 0),
                    'vid_name': meta.get('vid_name', video_path.name),
                    'original_fps': original_fps,
                    'target_fps': self.fps,
                    'width': width,
                    'height': height,
                    'duration': meta['duration'],
                    'num_frames': meta['num_frames'],
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting {video_path}: {e}")
            return None
    
    def extract_split(
        self,
        dataset_root: Path,
        split: str,
        word_list: Optional[List[str]] = None
    ) -> Dict:
        """
        Extract landmarks for entire split (train/val/test).
        
        Args:
            dataset_root: Root directory of IDLRW-DATASET
            split: 'train', 'val', or 'test'
            word_list: Optional list of words to process (default: all words)
            
        Returns:
            Dictionary with aggregated data
        """
        self.logger.info(f"Processing split: {split}")
        
        # Collect all video paths for this split
        video_files = []
        
        if word_list is None:
            # Get all word directories
            word_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
        else:
            word_dirs = [dataset_root / word for word in word_list]
        
        for word_dir in word_dirs:
            split_dir = word_dir / split
            if split_dir.exists():
                videos = list(split_dir.glob("*.mp4"))
                video_files.extend(videos)
        
        self.logger.info(f"Found {len(video_files)} videos for {split} split")
        
        if len(video_files) == 0:
            self.logger.warning(f"No videos found for {split} split")
            return None
        
        # Process videos in parallel (or sequentially for full partition to avoid memory issues)
        results = []
        failed_count = 0
        
        # Use sequential processing for 'full' partition or when max_workers=1
        # to avoid ProcessPoolExecutor crashes with MediaPipe GPU conflicts
        if self.partition == 'full' or self.max_workers == 1:
            if self.max_workers == 1:
                self.logger.info("Using sequential processing (max_workers=1)")
            else:
                self.logger.info("Using sequential processing for 'full' partition (468 nodes)")
            for video_path in tqdm(video_files, desc=f"Extracting {split}"):
                meta_path = video_path.with_suffix('.txt')
                if meta_path.exists():
                    result = self.extract_video_landmarks(video_path, meta_path)
                    if result is not None:
                        results.append(result)
                    else:
                        failed_count += 1
                else:
                    self.logger.warning(f"Missing meta file: {meta_path}")
                    failed_count += 1
        else:
            # Use parallel processing for lips/mouth partitions
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                futures = {}
                for video_path in video_files:
                    meta_path = video_path.with_suffix('.txt')
                    if meta_path.exists():
                        future = executor.submit(
                            self.extract_video_landmarks,
                            video_path,
                            meta_path
                        )
                        futures[future] = video_path
                    else:
                        self.logger.warning(f"Missing meta file: {meta_path}")
                        failed_count += 1
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {split}"):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed_count += 1
        
        self.logger.info(f"Successfully extracted {len(results)} videos")
        if failed_count > 0:
            self.logger.warning(f"Failed to extract {failed_count} videos")
        
        # Build word-to-label mapping
        words = sorted(list(set([r['word'] for r in results])))
        word_to_label = {word: idx for idx, word in enumerate(words)}
        
        self.logger.info(f"Found {len(words)} unique words (classes)")
        
        # Aggregate results
        aggregated = {
            'split': split,
            'partition': self.partition,
            'n_nodes': self.n_nodes,
            'adjacency': self.adjacency,
            'node_mapping': self.node_mapping,
            'fps': self.fps,
            'word_to_label': word_to_label,
            'videos': {},
        }
        
        for result in results:
            video_id = Path(result['video_path']).stem
            word = result['word']
            label = word_to_label[word]
            
            aggregated['videos'][video_id] = {
                'video_path': result['video_path'],
                'word': word,
                'label': label,
                'landmarks': result['landmarks'],
                'speech_mask': result['speech_mask'],
                'meta': result['meta'],
            }
        
        return aggregated
    
    def save_split(self, data: Dict, output_path: Path) -> None:
        """
        Save aggregated split data to .pt file.
        
        Args:
            data: Aggregated data dictionary
            output_path: Path to save .pt file
        """
        ensure_dir(output_path.parent)
        torch.save(data, output_path)
        self.logger.info(f"Saved to: {output_path}")
        
        # Log file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"File size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Extract FaceMesh landmarks')
    parser.add_argument('--partition', type=str, required=True,
                        choices=['lips', 'mouth', 'full'],
                        help='Partition type')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/member2/tomoooo/IndonesiaLipReading_GNN/data/IDLRW-DATASET',
                        help='Dataset root directory')
    parser.add_argument('--out-dir', type=str,
                        default='/home/member2/tomoooo/IndonesiaLipReading_GNN/data/extracted',
                        help='Output directory')
    parser.add_argument('--fps', type=int, default=25,
                        help='Target FPS')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Max parallel workers (default: 50%% of CPUs)')
    
    args = parser.parse_args()
    
    # Setup logger
    log_dir = Path(args.out_dir) / args.partition / 'logs'
    ensure_dir(log_dir)
    log_file = log_dir / f'extract_{args.split}.log'
    logger = setup_logger('FaceMeshExtractor', log_file=str(log_file))
    
    logger.info("=" * 60)
    logger.info("FACEMESH LANDMARK EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Partition: {args.partition}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Output dir: {args.out_dir}")
    logger.info(f"FPS: {args.fps}")
    logger.info("=" * 60)
    
    # Initialize extractor
    extractor = FaceMeshExtractor(
        partition=args.partition,
        fps=args.fps,
        max_workers=args.max_workers,
        logger=logger
    )
    
    # Extract split
    dataset_root = Path(args.dataset_root)
    data = extractor.extract_split(dataset_root, args.split)
    
    if data is None:
        logger.error("Extraction failed")
        return
    
    # Save to file
    output_dir = Path(args.out_dir) / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.partition}_{args.split}.pt"
    extractor.save_split(data, output_path)
    
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

