"""
Quick script to verify dataset loading
"""
import torch
from pathlib import Path
from dataset.dataset import LipReadingDataset

def check_dataset():
    """Check if dataset loads correctly"""
    data_dir = Path('data/processed')
    
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    # Check train dataset
    train_pt = data_dir / 'train.pt'
    if not train_pt.exists():
        print(f"❌ Train file not found: {train_pt}")
        return
    
    print(f"\n📁 Loading: {train_pt}")
    train_dataset = LipReadingDataset(str(train_pt))
    
    print(f"✅ Loaded {len(train_dataset)} samples")
    print(f"✅ Number of classes: {train_dataset.num_classes}")
    print(f"✅ Label map: {len(train_dataset.label_map)} labels")
    
    # Check a sample
    print(f"\n📊 Sample 0:")
    sample = train_dataset[0]
    print(f"  - x shape: {sample.x.shape}")
    print(f"  - x_temporal shape: {sample.x_temporal.shape}")
    print(f"  - edge_index shape: {sample.edge_index.shape}")
    print(f"  - speech_mask shape: {sample.speech_mask.shape}")
    print(f"  - num_frames: {sample.num_frames.item()}")
    print(f"  - label: {sample.y.item()}")
    print(f"  - video_id: {sample.video_id}")
    
    # Check data ranges
    print(f"\n📈 Data Ranges:")
    print(f"  - x: min={sample.x.min():.4f}, max={sample.x.max():.4f}")
    print(f"  - x_temporal: min={sample.x_temporal.min():.4f}, max={sample.x_temporal.max():.4f}")
    print(f"  - speech_mask: min={sample.speech_mask.min():.4f}, max={sample.speech_mask.max():.4f}")
    print(f"  - speech_mask unique: {torch.unique(sample.speech_mask).tolist()}")
    
    # Check edge index
    print(f"\n🔗 Edge Index:")
    print(f"  - Shape: {sample.edge_index.shape}")
    print(f"  - Num edges: {sample.edge_index.shape[1]}")
    print(f"  - Edge range: [{sample.edge_index.min()}, {sample.edge_index.max()}]")
    print(f"  - Num nodes: {sample.x.shape[0]}")
    
    # Check if speech mask is used
    print(f"\n⚠️  Speech Mask Usage:")
    speech_frames = (sample.speech_mask > 0.5).sum().item()
    total_frames = len(sample.speech_mask)
    print(f"  - Speech frames: {speech_frames}/{total_frames} ({100*speech_frames/total_frames:.1f}%)")
    print(f"  - ⚠️  NOTE: Speech mask is NOT used in model forward pass!")
    
    # Check edge index type
    print(f"\n🔗 Edge Index Type:")
    print(f"  - ⚠️  NOTE: Using k-NN edges (k=5), NOT anatomical edges from extractor!")
    
    print(f"\n{'='*60}")
    print("✅ Dataset loading verification complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    check_dataset()



