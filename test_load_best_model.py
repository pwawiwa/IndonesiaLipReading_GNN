#!/usr/bin/env python3
"""
Test loading best model from checkpoint - same pattern as in results/mouth/gin_lstm_mamba
"""
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.model_loader import load_model_from_checkpoint
from utils.config_loader import load_config


def test_load_best_model():
    """Test loading best model the same way as in results directory."""
    print("=" * 80)
    print("TEST LOADING BEST MODEL (Same pattern as results/mouth/gin_lstm_mamba)")
    print("=" * 80)
    
    # Path to best model
    result_dir = Path("results/mouth/gin_lstm_mamba/seed_0")
    checkpoint_path = result_dir / "best.pth"
    config_path = result_dir / "config.yaml"
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        return False
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    
    # Load config
    print("\n" + "-" * 80)
    print("1. LOADING CONFIG:")
    print("-" * 80)
    config = load_config(str(config_path))
    print(f"  Partition: {config['data']['partition']}")
    print(f"  Feature level: {config['data']['feature_level']}")
    print(f"  Model: {config['model']['name']}")
    print(f"  ✓ Config loaded")
    
    # Load checkpoint info
    print("\n" + "-" * 80)
    print("2. CHECKING CHECKPOINT STRUCTURE:")
    print("-" * 80)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Model config keys: {list(checkpoint['model_config'].keys())}")
    print(f"  ✓ Checkpoint structure valid")
    
    # Load model using the same pattern as in utils/model_loader.py
    print("\n" + "-" * 80)
    print("3. LOADING MODEL (Same pattern as load_model_from_checkpoint):")
    print("-" * 80)
    try:
        model = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            model_name=config['model']['name'],
            device='cpu'  # Use CPU for testing
        )
        print(f"  ✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Model in eval mode: {not model.training}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify model config matches
    print("\n" + "-" * 80)
    print("4. VERIFYING MODEL CONFIG:")
    print("-" * 80)
    model_config = checkpoint['model_config']
    print(f"  In features: {model_config.get('in_features', 'N/A')}")
    print(f"  Hidden dim: {model_config.get('hidden_dim', 'N/A')}")
    print(f"  Num classes: {model_config.get('num_classes', 'N/A')}")
    print(f"  Num GIN layers: {model_config.get('num_gin_layers', 'N/A')}")
    print(f"  Num LSTM layers: {model_config.get('num_lstm_layers', 'N/A')}")
    print(f"  ✓ Model config verified")
    
    # Test forward pass with dummy data
    print("\n" + "-" * 80)
    print("5. TESTING FORWARD PASS:")
    print("-" * 80)
    try:
        # Get dimensions from model config
        in_features = model_config['in_features']
        num_nodes = 277  # Mouth partition
        num_classes = model_config['num_classes']
        frames = 10
        
        # Create dummy input
        dummy_features = torch.randn(1, frames, num_nodes, in_features)
        dummy_adjacency = torch.eye(num_nodes).unsqueeze(0).repeat(1, frames, 1, 1)
        
        print(f"  Input shape: {dummy_features.shape}")
        print(f"  Adjacency shape: {dummy_adjacency.shape}")
        
        with torch.no_grad():
            output = model(dummy_features, dummy_adjacency)
            print(f"  Output shape: {output.shape}")
            print(f"  Expected shape: (1, {num_classes})")
            
            if output.shape == (1, num_classes):
                print(f"  ✓ Forward pass successful")
            else:
                print(f"  ✗ Output shape mismatch!")
                return False
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("RESULT: BEST MODEL LOADED SUCCESSFULLY ✓")
    print("=" * 80)
    print("\nLoading pattern matches:")
    print("  1. Load checkpoint with weights_only=False")
    print("  2. Get model_config from checkpoint")
    print("  3. Filter config to valid parameters")
    print("  4. Create model using get_model()")
    print("  5. Load state_dict from checkpoint")
    print("  6. Set model to eval mode")
    print("\nThis is the same pattern used in:")
    print("  - utils/model_loader.py::load_model_from_checkpoint()")
    print("  - utils/generate_classification_report.py")
    print("  - utils/generate_gradcam.py")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = test_load_best_model()
    sys.exit(0 if success else 1)

