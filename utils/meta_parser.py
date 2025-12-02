"""
Video metadata parser with SpeechMask generation.
"""
import re
import torch
from pathlib import Path
from typing import Dict, Optional
import math


def parse_video_meta(meta_path: str) -> Dict:
    """
    Parse video metadata file.
    
    Expected format:
        VidID: 0
        VidName: matanajwa20212_100_125_4.mp4
        ChannelId: 
        Start: 0.12 End: 0.88
        Duration: 0.76 seconds
    
    Args:
        meta_path: Path to metadata .txt file
        
    Returns:
        Dictionary with parsed metadata including speech_mask tensor
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        content = f.read().strip()
    
    meta = {}
    
    # Parse VidID
    vid_match = re.search(r'VidID:\s*(\d+)', content)
    if vid_match:
        meta['vid_id'] = int(vid_match.group(1))
    
    # Parse VidName
    name_match = re.search(r'VidName:\s*(.+)', content)
    if name_match:
        meta['vid_name'] = name_match.group(1).strip()
    
    # Parse ChannelId (optional, may be empty)
    channel_match = re.search(r'ChannelId:\s*(.+)?', content)
    meta['channel_id'] = channel_match.group(1).strip() if (channel_match and channel_match.group(1)) else ""
    
    # Parse Start and End
    start_end_match = re.search(r'Start:\s*([\d.]+)\s+End:\s*([\d.]+)', content)
    if start_end_match:
        meta['start'] = float(start_end_match.group(1))
        meta['end'] = float(start_end_match.group(2))
    else:
        raise ValueError(f"Could not parse Start/End from {meta_path}")
    
    # Parse Duration
    duration_match = re.search(r'Duration:\s*([\d.]+)', content)
    if duration_match:
        meta['duration'] = float(duration_match.group(1))
    else:
        raise ValueError(f"Could not parse Duration from {meta_path}")
    
    return meta


def generate_speech_mask(
    start: float,
    end: float,
    video_duration: float,
    fps: int = 25
) -> torch.Tensor:
    """
    Generate speech mask from start/end timestamps.
    
    IMPORTANT: The 'Duration' field in meta.txt is the speech segment duration (end-start),
    NOT the full video duration. This function requires the actual video duration.
    
    Logic:
    - Start/End are absolute timestamps in seconds (relative to video start)
    - Frames are inclusive [start_frame, end_frame]
    - Use floor for rounding (e.g., 0.15 * 25 = 3.75 → frame 3)
    
    Args:
        start: Start time in seconds (absolute, relative to video start)
        end: End time in seconds (absolute, relative to video start)
        video_duration: Total video duration in seconds (NOT the Duration field from meta.txt)
        fps: Frames per second (default: 25)
        
    Returns:
        Binary mask tensor of shape (num_frames,) where 1=speech, 0=non-speech
    """
    # Calculate total frames from actual video duration
    num_frames = math.floor(video_duration * fps)
    
    # Calculate speech frame range (inclusive)
    start_frame = math.floor(start * fps)
    end_frame = math.floor(end * fps)
    
    # Ensure frames are within valid range
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(0, min(end_frame, num_frames - 1))
    
    # Create mask (all zeros initially)
    mask = torch.zeros(num_frames, dtype=torch.float32)
    
    # Set speech frames to 1 (inclusive range)
    if start_frame <= end_frame:
        mask[start_frame:end_frame + 1] = 1.0
    
    return mask


def parse_video_meta_with_mask(meta_path: str, fps: int = 25, video_duration: Optional[float] = None) -> Dict:
    """
    Parse metadata and generate speech mask.
    
    WARNING: The 'Duration' field in meta.txt is the speech segment duration (end-start),
    NOT the full video duration. If video_duration is not provided, this function will
    generate a mask based on the maximum needed frame (end_frame), which may be incorrect.
    It's recommended to provide video_duration from the actual video file.
    
    Args:
        meta_path: Path to metadata .txt file
        fps: Frames per second (default: 25)
        video_duration: Actual video duration in seconds (if None, uses end time as estimate)
        
    Returns:
        Dictionary with parsed metadata and speech_mask tensor
    """
    meta = parse_video_meta(meta_path)
    
    # If video_duration not provided, estimate from end time (add small buffer)
    # This is a fallback - ideally video_duration should be provided
    if video_duration is None:
        # Use end time + small buffer (0.1s) as estimate
        estimated_duration = meta['end'] + 0.1
        video_duration = estimated_duration
    
    # Generate speech mask using actual video duration
    speech_mask = generate_speech_mask(
        start=meta['start'],
        end=meta['end'],
        video_duration=video_duration,
        fps=fps
    )
    
    meta['speech_mask'] = speech_mask
    meta['num_frames'] = len(speech_mask)
    
    return meta

