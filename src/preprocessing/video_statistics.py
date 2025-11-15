"""
video_statistics.py
Analyze video statistics: FPS, duration, frame count, etc.
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_video(video_path: Path) -> Dict:
    """
    Analyze a single video file
    
    Returns:
        Dictionary with video statistics or None if analysis fails
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0.0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height,
            'fourcc': fourcc,
            'file_size_mb': video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0.0
        }
    except Exception as e:
        logger.warning(f"Error analyzing {video_path.name}: {e}")
        return None


def analyze_dataset(dataset_root: Path, output_path: Path = None):
    """
    Analyze all videos in the dataset
    
    Args:
        dataset_root: Root directory (e.g., data/IDLRW-DATASET)
        output_path: Optional path to save JSON report
    """
    logger.info(f"\n{'='*70}")
    logger.info("VIDEO STATISTICS ANALYSIS")
    logger.info(f"{'='*70}\n")
    
    # Collect all videos
    video_files = []
    for word_dir in sorted(dataset_root.iterdir()):
        if not word_dir.is_dir():
            continue
        
        for split in ['train', 'val', 'test']:
            split_dir = word_dir / split
            if split_dir.exists():
                for video_path in sorted(split_dir.glob("*.mp4")):
                    video_files.append((video_path, word_dir.name, split))
    
    logger.info(f"Found {len(video_files)} videos to analyze\n")
    
    # Analyze videos
    stats_by_split = defaultdict(list)
    all_stats = []
    failed = 0
    
    for video_path, word, split in tqdm(video_files, desc="Analyzing videos"):
        stat = analyze_video(video_path)
        if stat is not None:
            stat['word'] = word
            stat['split'] = split
            stat['filename'] = video_path.name
            stats_by_split[split].append(stat)
            all_stats.append(stat)
        else:
            failed += 1
    
    # Calculate statistics
    def calc_stats(stats_list: List[Dict], name: str):
        """Calculate statistics for a list of video stats"""
        if not stats_list:
            return {}
        
        fps_values = [s['fps'] for s in stats_list if s['fps'] > 0]
        durations = [s['duration'] for s in stats_list if s['duration'] > 0]
        frame_counts = [s['frame_count'] for s in stats_list if s['frame_count'] > 0]
        file_sizes = [s['file_size_mb'] for s in stats_list if s['file_size_mb'] > 0]
        widths = [s['width'] for s in stats_list if s['width'] > 0]
        heights = [s['height'] for s in stats_list if s['height'] > 0]
        
        stats = {
            'total_videos': len(stats_list),
            'fps': {
                'mean': np.mean(fps_values) if fps_values else 0,
                'median': np.median(fps_values) if fps_values else 0,
                'std': np.std(fps_values) if fps_values else 0,
                'min': np.min(fps_values) if fps_values else 0,
                'max': np.max(fps_values) if fps_values else 0,
                'unique_values': sorted(set([round(f, 2) for f in fps_values])) if fps_values else []
            },
            'duration_seconds': {
                'mean': np.mean(durations) if durations else 0,
                'median': np.median(durations) if durations else 0,
                'std': np.std(durations) if durations else 0,
                'min': np.min(durations) if durations else 0,
                'max': np.max(durations) if durations else 0,
                'total_hours': sum(durations) / 3600 if durations else 0
            },
            'frame_count': {
                'mean': np.mean(frame_counts) if frame_counts else 0,
                'median': np.median(frame_counts) if frame_counts else 0,
                'std': np.std(frame_counts) if frame_counts else 0,
                'min': np.min(frame_counts) if frame_counts else 0,
                'max': np.max(frame_counts) if frame_counts else 0,
                'total_frames': sum(frame_counts) if frame_counts else 0
            },
            'file_size_mb': {
                'mean': np.mean(file_sizes) if file_sizes else 0,
                'median': np.median(file_sizes) if file_sizes else 0,
                'std': np.std(file_sizes) if file_sizes else 0,
                'min': np.min(file_sizes) if file_sizes else 0,
                'max': np.max(file_sizes) if file_sizes else 0,
                'total_gb': sum(file_sizes) / 1024 if file_sizes else 0
            },
            'resolution': {
                'width': {
                    'mean': np.mean(widths) if widths else 0,
                    'unique': sorted(set(widths)) if widths else []
                },
                'height': {
                    'mean': np.mean(heights) if heights else 0,
                    'unique': sorted(set(heights)) if heights else []
                },
                'common_resolutions': list(set([(w, h) for w, h in zip(widths, heights)])) if widths and heights else []
            }
        }
        
        return stats
    
    # Calculate per-split statistics
    split_stats = {}
    for split in ['train', 'val', 'test']:
        if stats_by_split[split]:
            split_stats[split] = calc_stats(stats_by_split[split], split)
    
    # Calculate overall statistics
    overall_stats = calc_stats(all_stats, 'overall')
    
    # Print report
    logger.info(f"\n{'='*70}")
    logger.info("OVERALL STATISTICS")
    logger.info(f"{'='*70}\n")
    
    logger.info(f"Total Videos: {overall_stats['total_videos']}")
    logger.info(f"Failed Analyses: {failed}")
    logger.info(f"Success Rate: {(overall_stats['total_videos'] / len(video_files) * 100):.1f}%\n")
    
    logger.info("FPS Statistics:")
    fps_stats = overall_stats['fps']
    logger.info(f"  Mean: {fps_stats['mean']:.2f} fps")
    logger.info(f"  Median: {fps_stats['median']:.2f} fps")
    logger.info(f"  Std Dev: {fps_stats['std']:.2f} fps")
    logger.info(f"  Range: {fps_stats['min']:.2f} - {fps_stats['max']:.2f} fps")
    logger.info(f"  Unique FPS values: {fps_stats['unique_values']}\n")
    
    logger.info("Duration Statistics:")
    dur_stats = overall_stats['duration_seconds']
    logger.info(f"  Mean: {dur_stats['mean']:.2f} seconds ({dur_stats['mean']:.2f}s)")
    logger.info(f"  Median: {dur_stats['median']:.2f} seconds")
    logger.info(f"  Std Dev: {dur_stats['std']:.2f} seconds")
    logger.info(f"  Range: {dur_stats['min']:.2f}s - {dur_stats['max']:.2f}s")
    logger.info(f"  Total Duration: {dur_stats['total_hours']:.2f} hours\n")
    
    logger.info("Frame Count Statistics:")
    frame_stats = overall_stats['frame_count']
    logger.info(f"  Mean: {frame_stats['mean']:.0f} frames")
    logger.info(f"  Median: {frame_stats['median']:.0f} frames")
    logger.info(f"  Std Dev: {frame_stats['std']:.0f} frames")
    logger.info(f"  Range: {frame_stats['min']:.0f} - {frame_stats['max']:.0f} frames")
    logger.info(f"  Total Frames: {frame_stats['total_frames']:,} frames\n")
    
    logger.info("File Size Statistics:")
    size_stats = overall_stats['file_size_mb']
    logger.info(f"  Mean: {size_stats['mean']:.2f} MB")
    logger.info(f"  Median: {size_stats['median']:.2f} MB")
    logger.info(f"  Std Dev: {size_stats['std']:.2f} MB")
    logger.info(f"  Range: {size_stats['min']:.2f} - {size_stats['max']:.2f} MB")
    logger.info(f"  Total Size: {size_stats['total_gb']:.2f} GB\n")
    
    logger.info("Resolution Statistics:")
    res_stats = overall_stats['resolution']
    logger.info(f"  Mean Width: {res_stats['width']['mean']:.0f} px")
    logger.info(f"  Mean Height: {res_stats['height']['mean']:.0f} px")
    logger.info(f"  Common Resolutions: {res_stats['common_resolutions']}\n")
    
    # Per-split statistics
    for split in ['train', 'val', 'test']:
        if split in split_stats:
            logger.info(f"\n{'='*70}")
            logger.info(f"{split.upper()} SPLIT STATISTICS")
            logger.info(f"{'='*70}\n")
            
            stats = split_stats[split]
            logger.info(f"Total Videos: {stats['total_videos']}")
            logger.info(f"FPS: {stats['fps']['mean']:.2f} ± {stats['fps']['std']:.2f} (range: {stats['fps']['min']:.2f}-{stats['fps']['max']:.2f})")
            logger.info(f"Duration: {stats['duration_seconds']['mean']:.2f}s ± {stats['duration_seconds']['std']:.2f}s (range: {stats['duration_seconds']['min']:.2f}s-{stats['duration_seconds']['max']:.2f}s)")
            logger.info(f"Frames: {stats['frame_count']['mean']:.0f} ± {stats['frame_count']['std']:.0f} (range: {stats['frame_count']['min']:.0f}-{stats['frame_count']['max']:.0f})")
            logger.info(f"Total Duration: {stats['duration_seconds']['total_hours']:.2f} hours")
    
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        return obj
    
    # Create full report
    report = {
        'overall': overall_stats,
        'by_split': split_stats,
        'failed_analyses': failed,
        'total_videos_found': len(video_files)
    }
    
    # Convert numpy types to native Python types
    report = convert_to_serializable(report)
    
    # Save to JSON
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n{'='*70}")
        logger.info(f"Report saved to: {output_path}")
        logger.info(f"{'='*70}\n")
    
    return report


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video statistics')
    parser.add_argument('--dataset_root', type=str, 
                       default='data/IDLRW-DATASET',
                       help='Root directory of dataset')
    parser.add_argument('--output', type=str,
                       default='reports/video_statistics.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return
    
    analyze_dataset(dataset_root, output_path)
    logger.info("✅ Analysis complete!")


if __name__ == "__main__":
    main()

