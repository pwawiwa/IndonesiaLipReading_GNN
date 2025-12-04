"""
Verify speech mask calculation is correct.
Tests that:
1. Speech mask uses video duration (not Duration field from meta.txt)
2. Speech mask correctly identifies frames based on Start/End timestamps
3. Duration field is the speech segment duration (end-start), not video duration
"""
import torch
import cv2
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.meta_parser import parse_video_meta, generate_speech_mask


def verify_speech_mask(video_path: Path, meta_path: Path, fps: int = 25):
    """
    Verify speech mask calculation for a video.
    
    Args:
        video_path: Path to video file
        meta_path: Path to metadata file
        fps: Target FPS
    """
    print("="*80)
    print("SPEECH MASK VERIFICATION")
    print("="*80)
    
    # 1. Parse metadata
    print("\n1. Parsing metadata...")
    meta = parse_video_meta(str(meta_path))
    start = meta['start']
    end = meta['end']
    duration_from_meta = meta['duration']
    
    print(f"   Start: {start:.4f} seconds")
    print(f"   End: {end:.4f} seconds")
    print(f"   Duration (from meta.txt): {duration_from_meta:.4f} seconds")
    print(f"   Expected speech segment duration: {end - start:.4f} seconds")
    
    # Verify Duration field matches (end - start)
    expected_duration = end - start
    if abs(duration_from_meta - expected_duration) < 0.01:
        print(f"   ✓ Duration field matches (end - start) = {expected_duration:.4f}")
    else:
        print(f"   ⚠ WARNING: Duration field ({duration_from_meta:.4f}) doesn't match (end-start)={expected_duration:.4f}")
    
    # 2. Get actual video duration
    print("\n2. Getting actual video duration...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"   ✗ Failed to open video: {video_path}")
        return None
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / original_fps if original_fps > 0 else 0
    cap.release()
    
    print(f"   Video FPS: {original_fps:.2f}")
    print(f"   Video frame count: {frame_count}")
    print(f"   Video duration: {video_duration:.4f} seconds")
    print(f"   ✓ Using VIDEO duration ({video_duration:.4f}s), NOT Duration field ({duration_from_meta:.4f}s)")
    
    # 3. Generate speech mask
    print("\n3. Generating speech mask...")
    speech_mask = generate_speech_mask(
        start=start,
        end=end,
        video_duration=video_duration,
        fps=fps
    )
    
    num_frames = len(speech_mask)
    speech_frames = int(speech_mask.sum().item())
    non_speech_frames = num_frames - speech_frames
    
    print(f"   Total frames (at {fps} FPS): {num_frames}")
    print(f"   Speech frames (mask=1): {speech_frames}")
    print(f"   Non-speech frames (mask=0): {non_speech_frames}")
    
    # 4. Calculate expected frame range
    print("\n4. Frame range calculation...")
    start_frame = math.floor(start * fps)
    end_frame = math.floor(end * fps)
    expected_speech_frames = end_frame - start_frame + 1  # Inclusive
    
    print(f"   Start time: {start:.4f}s → Frame {start_frame} (floor({start} * {fps}) = {start * fps:.2f})")
    print(f"   End time: {end:.4f}s → Frame {end_frame} (floor({end} * {fps}) = {end * fps:.2f})")
    print(f"   Expected speech frames: {expected_speech_frames} (frames {start_frame} to {end_frame}, inclusive)")
    
    if speech_frames == expected_speech_frames:
        print(f"   ✓ Speech frame count matches expected: {speech_frames}")
    else:
        print(f"   ⚠ WARNING: Speech frame count ({speech_frames}) doesn't match expected ({expected_speech_frames})")
    
    # 5. Show frame-by-frame breakdown
    print("\n5. Frame-by-frame breakdown (first 10 and last 10 frames):")
    print("   Frame | Time (s) | Speech Mask | Status")
    print("   " + "-"*50)
    
    for frame_idx in list(range(min(10, num_frames))) + list(range(max(0, num_frames - 10), num_frames)):
        time_sec = frame_idx / fps
        mask_val = speech_mask[frame_idx].item()
        status = "SPEECH ON" if mask_val > 0.5 else "SPEECH OFF"
        
        # Check if frame should be in speech range
        in_range = start_frame <= frame_idx <= end_frame
        expected_status = "SPEECH ON" if in_range else "SPEECH OFF"
        
        match = "✓" if (mask_val > 0.5) == in_range else "✗"
        
        print(f"   {frame_idx:5d} | {time_sec:7.4f} | {mask_val:11.1f} | {status:10s} {match}")
    
    # 6. Verify all frames
    print("\n6. Verification summary:")
    correct_frames = 0
    for frame_idx in range(num_frames):
        time_sec = frame_idx / fps
        mask_val = speech_mask[frame_idx].item()
        should_be_speech = start <= time_sec <= end
        is_speech = mask_val > 0.5
        
        # Account for frame discretization
        # Frame represents time range [frame_idx/fps, (frame_idx+1)/fps)
        frame_start_time = frame_idx / fps
        frame_end_time = (frame_idx + 1) / fps
        
        # Frame should be speech if it overlaps with [start, end]
        should_be_speech = not (frame_end_time <= start or frame_start_time >= end)
        
        if (should_be_speech and is_speech) or (not should_be_speech and not is_speech):
            correct_frames += 1
    
    accuracy = correct_frames / num_frames * 100
    print(f"   Correct frames: {correct_frames}/{num_frames} ({accuracy:.1f}%)")
    
    if accuracy == 100.0:
        print("   ✓ All frames correctly labeled!")
    else:
        print(f"   ⚠ Some frames may be incorrectly labeled")
    
    return {
        'start': start,
        'end': end,
        'duration_meta': duration_from_meta,
        'video_duration': video_duration,
        'speech_mask': speech_mask,
        'num_frames': num_frames,
        'speech_frames': speech_frames
    }


def main():
    """Test with sample videos from the dataset."""
    dataset_root = Path("data/IDLRW-DATASET")
    
    if not dataset_root.exists():
        print(f"Dataset not found at {dataset_root}")
        print("Testing with sample_mouth.pt instead...")
        
        # Test with sample_mouth.pt
        sample_file = Path("data/extracted/mouth/sample_mouth.pt")
        if sample_file.exists():
            data = torch.load(sample_file, map_location='cpu', weights_only=False)
            print("\n" + "="*80)
            print("VERIFYING SAMPLE DATA")
            print("="*80)
            
            for vid_id, vid_data in list(data['videos'].items())[:2]:
                print(f"\n\nVideo: {vid_id}")
                landmarks = vid_data['landmarks']
                speech_mask = vid_data['speech_mask']
                meta = vid_data.get('meta', {})
                
                print(f"  Total frames: {landmarks.shape[0]}")
                print(f"  Speech ON frames: {int(speech_mask.sum().item())}")
                print(f"  Speech OFF frames: {int((1 - speech_mask).sum().item())}")
                print(f"  Video duration (from meta): {meta.get('duration', 'N/A')}")
                print(f"  Original FPS: {meta.get('original_fps', 'N/A')}")
                print(f"  Target FPS: {meta.get('target_fps', 'N/A')}")
                
                print(f"\n  Speech mask values:")
                print(f"    {speech_mask.tolist()}")
                
                # Show which frames are ON
                speech_on_frames = torch.where(speech_mask > 0.5)[0].tolist()
                print(f"  Speech ON at frames: {speech_on_frames}")
        return
    
    # Find a test video
    test_videos = []
    for word_dir in dataset_root.iterdir():
        if word_dir.is_dir():
            for split_dir in word_dir.iterdir():
                if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
                    videos = list(split_dir.glob("*.mp4"))
                    if videos:
                        test_videos.append((videos[0], videos[0].with_suffix('.txt')))
                        if len(test_videos) >= 2:
                            break
            if len(test_videos) >= 2:
                break
    
    if not test_videos:
        print("No test videos found")
        return
    
    print(f"Found {len(test_videos)} test videos\n")
    
    for video_path, meta_path in test_videos[:2]:
        if meta_path.exists():
            result = verify_speech_mask(video_path, meta_path)
            print("\n")


if __name__ == '__main__':
    main()

