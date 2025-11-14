"""
sample_output.py
Test extractor on a single video and print detailed output
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from facemesh_extractor import FaceMeshExtractor


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test extractor on a single video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--word', type=str, default='test', help='Word label')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"TESTING EXTRACTOR")
    print(f"{'='*80}")
    print(f"Video: {args.video}")
    print(f"Word: {args.word}\n")
    
    # Create extractor
    extractor = FaceMeshExtractor(num_workers=1)
    
    # Extract
    result = extractor.extract_video(Path(args.video), args.word)
    
    if result is None:
        print("❌ EXTRACTION FAILED!")
        print("Possible reasons:")
        print("  - Video file corrupt or unreadable")
        print("  - Face detection rate < 70%")
        print("  - No valid frames extracted")
        return
    
    print("✅ EXTRACTION SUCCESSFUL!\n")
    
    # Print all keys
    print("[Keys in result]")
    for key in result.keys():
        print(f"  - {key}")
    
    print(f"\n[Shapes]")
    print(f"  video_id: '{result['video_id']}'")
    print(f"  label: '{result['label']}'")
    print(f"  landmarks: {tuple(result['landmarks'].shape)} (T, N, 3)")
    print(f"  action_units: {tuple(result['action_units'].shape)} (T, 18)")
    print(f"  geometric: {tuple(result['geometric'].shape)} (T, 10)")
    print(f"  speech_mask: {tuple(result['speech_mask'].shape)} (T)")
    
    print(f"\n[Metadata]")
    print(f"  num_frames: {result['num_frames']}")
    print(f"  detection_rate: {result['detection_rate']:.1%}")
    print(f"  num_landmarks: {result['landmarks'].shape[1]}")
    
    print(f"\n[Extractor Info]")
    print(f"  ROI landmarks: {len(extractor.ROI_INDICES)}")
    print(f"  Edge connections: {extractor.edge_index.shape[1]}")
    
    print(f"\n[Data Ranges]")
    lm = result['landmarks']
    print(f"  landmarks: min={lm.min():.4f}, max={lm.max():.4f}, mean={lm.mean():.4f}")
    
    au = result['action_units']
    print(f"  action_units: min={au.min():.4f}, max={au.max():.4f}, mean={au.mean():.4f}")
    
    geo = result['geometric']
    print(f"  geometric: min={geo.min():.4f}, max={geo.max():.4f}, mean={geo.mean():.4f}")
    
    print(f"\n[Speech Mask]")
    mask = result['speech_mask']
    speech_frames = mask.sum().item()
    total_frames = len(mask)
    print(f"  Speech frames: {speech_frames:.0f} / {total_frames}")
    print(f"  Speech ratio: {speech_frames/total_frames:.1%}")
    print(f"  First 30 frames: {mask[:30].tolist()}")
    
    print(f"\n[Sample Features - Frame 0]")
    print(f"  First 3 landmarks: {lm[0, :3].tolist()}")
    print(f"  Action units: {au[0].tolist()}")
    print(f"  Geometric: {geo[0].tolist()}")
    
    print(f"\n[Verification]")
    print(f"  ✓ All values in [0, 1]: {(lm >= 0).all() and (lm <= 1).all() and (au >= 0).all() and (au <= 1).all()}")
    print(f"  ✓ No NaN values: {not lm.isnan().any() and not au.isnan().any()}")
    print(f"  ✓ Speech mask binary: {set(mask.tolist()).issubset({0.0, 1.0})}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()