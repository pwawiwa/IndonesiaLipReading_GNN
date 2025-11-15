#!/usr/bin/env python3
"""Inspect model parameters from checkpoint files"""
import torch
from pathlib import Path
import json

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and extract parameters"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n📦 Checkpoint Keys: {list(ckpt.keys())}")
    
    # Extract metadata
    if 'epoch' in ckpt:
        print(f"\n📅 Epoch: {ckpt['epoch']}")
    
    if 'val_acc' in ckpt:
        print(f"✅ Best Val Accuracy: {ckpt['val_acc']:.6f}")
    
    if 'val_f1' in ckpt:
        print(f"✅ Best Val F1: {ckpt['val_f1']:.6f}")
    
    # Optimizer parameters
    if 'optimizer_state_dict' in ckpt:
        opt_state = ckpt['optimizer_state_dict']
        if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
            pg = opt_state['param_groups'][0]
            print(f"\n⚙️  Optimizer Parameters:")
            print(f"   Learning Rate: {pg.get('lr', 'N/A')}")
            print(f"   Initial LR: {pg.get('initial_lr', 'N/A')}")
            print(f"   Weight Decay: {pg.get('weight_decay', 'N/A')}")
            print(f"   Betas: {pg.get('betas', 'N/A')}")
    
    # Model architecture (infer from state dict)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print(f"\n🏗️  Model Architecture (inferred from state dict):")
        
        # Try to infer dimensions from layer names
        spatial_dim = None
        temporal_dim = None
        num_classes = None
        dropout = None
        
        for key in state_dict.keys():
            if 'spatial' in key.lower():
                if 'weight' in key and len(state_dict[key].shape) >= 2:
                    dim = state_dict[key].shape[0]
                    if spatial_dim is None or dim > spatial_dim:
                        spatial_dim = dim
            if 'temporal' in key.lower() or 'lstm' in key.lower():
                if 'weight' in key and len(state_dict[key].shape) >= 2:
                    dim = state_dict[key].shape[0]
                    if temporal_dim is None or dim > temporal_dim:
                        temporal_dim = dim
            if 'classifier' in key.lower():
                if 'weight' in key and len(state_dict[key].shape) >= 2:
                    # Last layer of classifier should have num_classes
                    if state_dict[key].shape[0] < 1000:  # Reasonable upper bound
                        num_classes = state_dict[key].shape[0]
        
        # Check for specific layer dimensions
        if 'spatial.gcn_layers.0.weight' in state_dict:
            spatial_dim = state_dict['spatial.gcn_layers.0.weight'].shape[0]
        if 'temporal.lstm.weight_ih_l0' in state_dict:
            # LSTM input dimension
            lstm_input = state_dict['temporal.lstm.weight_ih_l0'].shape[1]
            # Hidden dimension is typically 1/4 of weight_ih (for 4 gates)
            temporal_dim = state_dict['temporal.lstm.weight_ih_l0'].shape[0] // 4
        if 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].shape[0]
        
        if spatial_dim:
            print(f"   Spatial Dimension: {spatial_dim}")
        if temporal_dim:
            print(f"   Temporal Dimension: {temporal_dim}")
        if num_classes:
            print(f"   Number of Classes: {num_classes}")
        
        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values())
        trainable_params = total_params  # All saved params are trainable
        print(f"   Total Parameters: {total_params:,}")
    
    # Training history
    if 'history' in ckpt:
        history = ckpt['history']
        print(f"\n📊 Training History:")
        if 'train_loss' in history:
            print(f"   Training epochs: {len(history['train_loss'])}")
            if len(history['train_loss']) > 0:
                print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        if 'val_acc' in history:
            if len(history['val_acc']) > 0:
                print(f"   Best val acc: {max(history['val_acc']):.6f}")
                print(f"   Final val acc: {history['val_acc'][-1]:.6f}")

if __name__ == "__main__":
    # Check for config files
    config_files = [
        'outputs/combined/config.json',
        'outputs/combined/training_config.json',
        'outputs/combined/params.json'
    ]
    
    print("🔍 Searching for config files...")
    found_config = False
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"\n✅ Found config file: {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("\n📋 Training Configuration:")
            for key, value in config.items():
                print(f"   {key}: {value}")
            found_config = True
    
    if not found_config:
        print("   No config files found in outputs/combined/")
    
    # Inspect checkpoints
    inspect_checkpoint('outputs/combined/best_model.pth')
    inspect_checkpoint('outputs/combined/final_model.pth')
    
    # Also check training script for default config
    print(f"\n{'='*60}")
    print("📝 Default Config from train.py:")
    print(f"{'='*60}")
    print("""
    Based on src/train.py, the default training parameters are:
    
    Model Architecture:
      - spatial_dim: 256
      - temporal_dim: 512
      - spatial_layers: 3
      - temporal_layers: 2
      - dropout: 0.4
    
    Training:
      - batch_size: 64
      - learning_rate: 1e-3 (0.001)
      - num_epochs: 100
      - weight_decay: 1e-4 (0.0001)
      - optimizer: AdamW
      - scheduler: CosineAnnealingLR
      - label_smoothing: 0.0
      - gradient_clip_norm: 1.0
    """)

