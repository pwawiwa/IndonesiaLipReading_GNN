# src/preprocessing/check_pt.py
import torch
from pathlib import Path
import numpy as np

def inspect_pt_file(pt_path):
    """
    Inspect a single .pt file and print summary
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        print(f"❌ File not found: {pt_path}")
        return

    data = torch.load(pt_path)
    print(f"\n📂 Inspecting: {pt_path}")
    print(f"Total samples: {len(data)}")

    # Check the first sample for structure
    sample = data[0]
    print(f"Keys in sample: {list(sample.keys())}")

    # Landmarks
    landmarks = sample.get('landmarks', None)
    if landmarks is not None:
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Landmarks dtype: {landmarks.dtype}")
        print(f"Landmarks stats: min={landmarks.min().item():.4f}, max={landmarks.max().item():.4f}, mean={landmarks.mean().item():.4f}")

    # Features
    features = sample.get('features', None)
    if features is not None:
        print(f"Features keys: {list(features.keys())}")
        for k, v in features.items():
            arr = v.numpy()
            print(f"  {k}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    # Metadata
    metadata = sample.get('metadata', None)
    if metadata is not None:
        print(f"Metadata: {metadata}")

    # Label
    label = sample.get('label', None)
    if label is not None:
        print(f"Label: {label}")

def main():
    # Adjust paths if needed
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    for split in ["train.pt", "val.pt", "test.pt"]:
        pt_file = processed_dir / split
        inspect_pt_file(pt_file)

if __name__ == "__main__":
    main()
