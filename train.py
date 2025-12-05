"""
Main training script for Indonesia Lip Reading GNN.

Usage:
    python train.py --config configs/gcn_lips_B0.yaml
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import numpy as np

from utils import load_config, setup_logger, log_system_info, ensure_dir
from utils.model_loader import filter_model_config
from models import get_model
from training import get_dataloader, Trainer
from training.augmentations import create_augmentation_pipeline


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for lip reading')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup paths
    partition = config['data']['partition']
    feature_level = config['data']['feature_level']
    model_name = config['model']['name']
    
    # Output directory
    output_dir = Path(config['training']['output_dir']) / partition / feature_level / model_name / f"seed_{args.seed}"
    ensure_dir(output_dir)
    
    # Setup logger
    logger = setup_logger('Train', log_file=str(output_dir / 'train.log'))
    
    logger.info("=" * 60)
    logger.info("INDONESIA LIP READING GNN - TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)
    
    # Log system info
    system_info = log_system_info(logger)
    
    # Load data
    logger.info("Loading data...")
    feature_dir = Path(config['data']['feature_dir'])
    
    # Use incremental loading: B0 loads B0 only, B1 loads B0+B1, etc.
    train_file = f"{partition}_train.pt"
    val_file = f"{partition}_val.pt"
    test_file = f"{partition}_test.pt"
    
    logger.info(f"Feature level: {feature_level}")
    level_map = {'B0': 0, 'B1': 1, 'B2': 2, 'B3': 3}
    target_level = level_map.get(feature_level, 0)
    levels_to_load = [f'B{i}' for i in range(target_level + 1)]
    logger.info(f"Incremental loading: Will load {' + '.join(levels_to_load)}")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Val file: {val_file}")
    logger.info(f"Test file: {test_file}")
    
    # Create augmentation pipeline (only for training)
    augmentation_config = config.get('augmentation', {})
    train_transform = create_augmentation_pipeline({
        **augmentation_config,
        'feature_level': feature_level
    })
    
    if train_transform is not None:
        logger.info("Augmentation enabled for training")
        aug_list = []
        for aug in train_transform.augmentations:
            aug_list.append(f"{aug.__class__.__name__} (p={aug.p})")
        logger.info(f"  Augmentations: {', '.join(aug_list)}")
    else:
        logger.info("No augmentation (disabled or not configured)")
    
    # Create data loaders with incremental loading
    # Enable class balancing: oversample minority classes to match max class count
    balance_classes = config['training'].get('balance_classes', False)
    balance_factor = config['training'].get('balance_factor', 1.0)  # 1.0 = full balance, 0.5 = half balance
    
    train_loader = get_dataloader(
        train_file,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', None),
        feature_level=feature_level,
        feature_dir=str(feature_dir),
        transform=train_transform,  # Augmentation only for training
        balance_classes=balance_classes,  # Balance classes to max count
        balance_factor=balance_factor  # Balance factor (1.0 = full, 0.5 = half, etc.)
    )
    
    if balance_classes:
        logger.info(f"Class balancing enabled: oversampling minority classes (factor={balance_factor:.2f})")
        if train_loader.sampler:
            logger.info(f"  Balanced epoch size: {train_loader.sampler.num_samples:,} samples "
                       f"(original: {len(train_loader.dataset):,}, "
                       f"ratio: {train_loader.sampler.num_samples / len(train_loader.dataset):.2f}x)")
    
    val_loader = get_dataloader(
        val_file,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', None),
        feature_level=feature_level,
        feature_dir=str(feature_dir),
        transform=None  # No augmentation for validation
    )
    
    test_loader = get_dataloader(
        test_file,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', None),
        feature_level=feature_level,
        feature_dir=str(feature_dir),
        transform=None  # No augmentation for test
    )
    
    # Get dataset info
    dataset = train_loader.dataset
    num_classes = dataset.get_num_classes()
    in_features = dataset.get_feature_dim()
    num_nodes = dataset.get_num_nodes()  # Get number of nodes from dataset
    
    logger.info(f"Classes: {num_classes}")
    logger.info(f"Input features: {in_features}")
    logger.info(f"Number of nodes: {num_nodes}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Class weights and focal loss are disabled per user request
    use_class_weights = False
    use_focal_loss = False
    class_weights = None
    
    # Build model
    logger.info("Building model...")
    model_config = config['model']
    model_config['params']['in_features'] = in_features
    model_config['params']['num_classes'] = num_classes
    # Auto-fill num_nodes for adaptive models if not provided
    if model_name in ['adaptive_gcn', 'adaptive_gcn_lstm', 'adaptive_gcn_lstm_mamba'] and 'num_nodes' not in model_config['params']:
        model_config['params']['num_nodes'] = num_nodes
        logger.info(f"Auto-filled num_nodes: {num_nodes}")
    
    # Filter model params to only include valid parameters for the model
    filtered_params = filter_model_config(model_name, model_config['params'])
    
    model = get_model(model_name, **filtered_params)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    # Loss function - standard CrossEntropyLoss (no class weights or focal loss)
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"Using label smoothing: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")
    
    # Optimizer
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 1e-4)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Optimizer: {optimizer_name}, LR: {lr}")
    
    # Scheduler (optional)
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler_config = config['training']['scheduler']
        scheduler_name = scheduler_config['name'].lower()
        
        if scheduler_name == 'steplr':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
        elif scheduler_name == 'cosine':
            T_max = config['training']['epochs']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            logger.info(f"Scheduler: CosineAnnealingLR (T_max={T_max})")
        elif scheduler_name == 'reduceonplateau':
            mode = scheduler_config.get('mode', 'min')
            factor = float(scheduler_config.get('factor', 0.5))
            patience = int(scheduler_config.get('patience', 5))
            min_lr = scheduler_config.get('min_lr', 1e-6)
            # Convert min_lr to float if it's a string (e.g., "1e-6" from YAML)
            if isinstance(min_lr, str):
                min_lr = float(min_lr)
            else:
                min_lr = float(min_lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
            )
            logger.info(f"Scheduler: ReduceLROnPlateau (mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr})")
        elif scheduler_name == 'cosinewarmrestarts':
            T_0 = scheduler_config.get('T_0', 10)  # First restart period
            T_mult = scheduler_config.get('T_mult', 2)  # Period multiplier
            eta_min = scheduler_config.get('eta_min', 1e-6)  # Minimum learning rate
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            logger.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
        elif scheduler_name == 'warmup':
            # Warmup + CosineAnnealingLR
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            T_max = config['training']['epochs'] - warmup_epochs
            eta_min = float(scheduler_config.get('eta_min', 1e-6))
            if isinstance(eta_min, str):
                eta_min = float(eta_min)
            
            # Create a lambda scheduler for warmup
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup: gradually increase from 0 to 1
                    return (epoch + 1) / warmup_epochs
                else:
                    # Cosine annealing after warmup
                    adjusted_epoch = epoch - warmup_epochs
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * adjusted_epoch / T_max))
                    # Scale from eta_min/lr to 1.0
                    return eta_min / lr + (1 - eta_min / lr) * cosine_factor
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logger.info(f"Scheduler: Warmup ({warmup_epochs} epochs) + CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
    
    # Define callback to save metadata after each epoch
    def save_metadata_callback(epoch, save_dir, metrics):
        """Save run metadata and summary after each epoch."""
        try:
            # Save run metadata (incremental - updates each epoch)
            run_meta = {
                'partition': partition,
                'feature_level': feature_level,
                'model': model_name,
                'seed': args.seed,
                'config': config,
                'system_info': system_info,
                'current_epoch': epoch,
                'best_val_acc': metrics['best_val_acc'],
                'best_epoch': metrics['best_epoch'],
                'current_train_loss': metrics['train_loss'],
                'current_train_acc': metrics['train_acc'],
                'current_val_loss': metrics['val_loss'],
                'current_val_acc': metrics['val_acc'],
                'num_parameters': model.count_parameters(),
                'training_in_progress': True,  # Flag to indicate training not complete
            }
            
            torch.save(run_meta, save_dir / 'run_meta.pt')
            
            # Write summary (incremental - updates each epoch)
            with open(save_dir / 'summary.txt', 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("TRAINING SUMMARY (IN PROGRESS)\n")
                f.write("=" * 60 + "\n")
                f.write(f"Partition: {partition}\n")
                f.write(f"Feature Level: {feature_level}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Seed: {args.seed}\n")
                f.write(f"Parameters: {model.count_parameters():,}\n")
                f.write(f"\n")
                f.write(f"Current Epoch: {epoch}\n")
                f.write(f"Best Val Accuracy: {metrics['best_val_acc']:.2f}% (epoch {metrics['best_epoch']})\n")
                f.write(f"Current Train Loss: {metrics['train_loss']:.4f}\n")
                f.write(f"Current Train Acc: {metrics['train_acc']:.2f}%\n")
                f.write(f"Current Val Loss: {metrics['val_loss']:.4f}\n")
                f.write(f"Current Val Acc: {metrics['val_acc']:.2f}%\n")
                f.write("=" * 60 + "\n")
                f.write("Note: Training in progress. Final results will be updated upon completion.\n")
        except Exception as e:
            logger.warning(f"Failed to save metadata at epoch {epoch}: {e}")
    
    # Create trainer with callback
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
        epoch_callback=save_metadata_callback
    )
    
    # Train with early stopping and gradient clipping
    early_stopping_patience = config['training'].get('early_stopping_patience', None)
    gradient_clip = config['training'].get('gradient_clip', None)
    
    history = trainer.train(
        num_epochs=config['training']['epochs'],
        save_dir=output_dir,
        early_stopping_patience=early_stopping_patience,
        gradient_clip=gradient_clip
    )
    
    # Save history
    trainer.save_history(output_dir / 'history.pt')
    logger.info(f"Saved training history to {output_dir / 'history.pt'}")
    
    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)
    
    # Load best model (use strict=False to handle missing keys from older checkpoints)
    checkpoint = torch.load(output_dir / 'best.pth', weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint (will use default initialization): {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")
    
    # Test
    trainer.model = model
    trainer.val_loader = test_loader
    test_loss, test_acc = trainer.validate()
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save final run metadata (overwrites incremental version with final results)
    run_meta = {
        'partition': partition,
        'feature_level': feature_level,
        'model': model_name,
        'seed': args.seed,
        'config': config,
        'system_info': system_info,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'num_parameters': model.count_parameters(),
        'training_in_progress': False,  # Training complete
    }
    
    torch.save(run_meta, output_dir / 'run_meta.pt')
    logger.info(f"Saved final run metadata to {output_dir / 'run_meta.pt'}")
    
    # Write final summary (overwrites incremental version)
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Partition: {partition}\n")
        f.write(f"Feature Level: {feature_level}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Parameters: {model.count_parameters():,}\n")
        f.write(f"\n")
        f.write(f"Best Val Accuracy: {trainer.best_val_acc:.2f}% (epoch {trainer.best_epoch})\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write("=" * 60 + "\n")


if __name__ == '__main__':
    main()

