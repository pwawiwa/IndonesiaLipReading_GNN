"""
src/train.py
Simple training script without early stopping
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import logging
from datetime import datetime
from typing import Optional

from dataset.dataset import create_dataloaders
from models.combined import CombinedModel
from models.ast_gcn import ASTGCNModel
from utils.log_dashboard import start_log_server

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Attach a file handler so training logs persist to disk."""
    log_path = Path(log_path)
    existing = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
    ]
    if existing:
        return log_path
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"🔒 File logging enabled at {log_path}")
    return log_path


class Trainer:
    """Simple trainer"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr, num_epochs, save_dir, label_map, 
                 checkpoint_interval: int = 10, resume: bool = False,
                 resume_checkpoint: Optional[str] = None,
                 epochs_per_run: Optional[int] = None,
                 label_smoothing: float = 0.0,
                 early_stopping_patience: int = 50,
                 early_stopping_min_delta: float = 0.001,
                 weight_decay: float = 1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.resume_checkpoint = resume_checkpoint
        self.epochs_per_run = epochs_per_run
        self.label_smoothing = label_smoothing
        self.start_epoch = 1
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.patience_counter = 0
        self.best_val_acc_for_patience = 0.0
        self.weight_decay = weight_decay
        
        # Optimizer with stronger weight decay to reduce overfitting
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr/100
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Enable AMP for faster training
        self.use_amp = device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Resume if requested
        if self.resume:
            self._resume_from_checkpoint(self.resume_checkpoint)
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch.y.squeeze())
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Stats
            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y.squeeze()).sum().item()
            total += batch.num_graphs
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{correct/total:.4f}'
            })
        
        return total_loss / total, correct / total
    
    def validate(self):
        """Validate"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch)
                        loss = self.criterion(logits, batch.y.squeeze())
                else:
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch.y.squeeze())
                
                total_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y.squeeze()).sum().item()
                total += batch.num_graphs
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
        
        # Calculate F1
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return total_loss / total, correct / total, f1
    
    def train_all(self):
        """Train for all epochs"""
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING START")
        logger.info(f"{'='*60}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        logger.info(f"Classes: {len(self.label_map)}")
        if self.resume and self.start_epoch > 1:
            logger.info(f"Resuming from epoch {self.start_epoch}")
        segment_msg = ""
        run_end_epoch = self.num_epochs
        if self.epochs_per_run is not None:
            run_end_epoch = min(self.num_epochs, self.start_epoch + self.epochs_per_run - 1)
            segment_msg = f"This run will cover epochs {self.start_epoch}-{run_end_epoch}."
        logger.info(f"{'='*60}\n")
        if segment_msg:
            logger.info(segment_msg)
        
        best_val_acc = max(self.history['val_acc']) if self.history['val_acc'] else 0
        best_val_f1 = max(self.history['val_f1']) if self.history['val_f1'] else 0
        
        for epoch in range(self.start_epoch, run_end_epoch + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            logger.info(f"{'-'*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Scheduler
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Log
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            gen_gap = val_loss - train_loss
            logger.info(f"LR: {lr:.6f} | Generalization Gap (val-train loss): {gen_gap:.4f}")
            
            # Track best
            save_best = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, self.save_dir / 'best_model.pth')
                logger.info(f"✓ Saved best model (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
                save_best = True
            
            # Early stopping logic
            if val_acc > self.best_val_acc_for_patience + self.early_stopping_min_delta:
                self.best_val_acc_for_patience = val_acc
                self.patience_counter = 0
                logger.info(f"✓ Validation improved, resetting patience counter")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ No improvement for {self.patience_counter}/{self.early_stopping_patience} epochs")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"\n{'='*60}")
                    logger.info("EARLY STOPPING TRIGGERED")
                    logger.info(f"{'='*60}")
                    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
                    logger.info(f"Best Val F1: {best_val_f1:.4f}")
                    logger.info(f"Stopped at epoch {epoch}")
                    logger.info(f"{'='*60}\n")
                    break

            # Periodic checkpoints to allow resuming/analysis mid-training
            should_checkpoint = (epoch % self.checkpoint_interval == 0) or save_best or (epoch == run_end_epoch)
            if should_checkpoint:
                self._save_checkpoint(epoch, is_best=save_best)
        
        logger.info(f"\n{'='*60}")
        if run_end_epoch >= self.num_epochs:
            logger.info(f"TRAINING COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Best Val Acc: {best_val_acc:.4f}")
            logger.info(f"Best Val F1: {best_val_f1:.4f}")
            logger.info(f"{'='*60}\n")
            
            # Save final
            torch.save({
                'epoch': self.num_epochs,
                'model_state_dict': self.model.state_dict(),
                'history': self.history,
            }, self.save_dir / 'final_model.pth')
            
            # Save final history and visualization
            self._persist_history(self.save_dir / 'history.json')
            self.plot_history()
        else:
            logger.info(f"SEGMENT COMPLETE (epochs {self.start_epoch}-{run_end_epoch}).")
            logger.info("Resume training later to continue toward full 1000 epochs.")
        
        # Persist latest history regardless of completion
        self._persist_history(self.save_dir / 'history.json')

    def _persist_history(self, target_path: Path):
        """Persist current training history to disk"""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save checkpoint with current training state.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this checkpoint corresponds to a new best model
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_acc': max(self.history['val_acc']) if self.history['val_acc'] else 0.0,
            'best_val_f1': max(self.history['val_f1']) if self.history['val_f1'] else 0.0,
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(state, checkpoint_path)
        torch.save(state, self.checkpoint_dir / 'latest_checkpoint.pth')
        
        if is_best:
            torch.save(state, self.checkpoint_dir / 'best_checkpoint.pth')
        
        # Persist latest history & viz for progress reporting
        self._persist_history(self.checkpoint_dir / 'history_latest.json')
        self.plot_history(self.checkpoint_dir / 'training_curves_latest.png')
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _resume_from_checkpoint(self, resume_checkpoint: Optional[str]):
        """
        Resume optimizer/model/history state from checkpoint.
        """
        ckpt_path = Path(resume_checkpoint) if resume_checkpoint else self.checkpoint_dir / 'latest_checkpoint.pth'
        if not ckpt_path.exists():
            logger.info(f"No checkpoint found at {ckpt_path}, starting fresh.")
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        loaded_history = checkpoint.get('history', self.history)
        # Ensure all keys exist
        for key in self.history.keys():
            loaded_history.setdefault(key, [])
        self.history = loaded_history
        
        last_epoch = checkpoint.get('epoch', 0)
        self.start_epoch = min(last_epoch + 1, self.num_epochs)
        
        logger.info(f"Resumed training from {ckpt_path} at epoch {self.start_epoch}")
    
    def plot_history(self, output_path: Optional[Path] = None):
        """Plot training curves and optionally override output path"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[2].plot(epochs, self.history['val_f1'], 'g-', label='Val F1', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('F1 Score', fontsize=12)
        axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if output_path is None:
            viz_dir = Path('reports/viz')
            viz_dir.mkdir(parents=True, exist_ok=True)
            output_path = viz_dir / 'training_curves.png'
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {output_path}")
        plt.close()
    
    def test(self):
        """Test and generate report"""
        logger.info(f"\n{'='*60}")
        logger.info("TESTING")
        logger.info(f"{'='*60}\n")
        
        # Load best model
        checkpoint = torch.load(self.save_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        logger.info(f"Best Val Acc: {checkpoint['val_acc']:.4f}")
        logger.info(f"Best Val F1: {checkpoint['val_f1']:.4f}\n")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch)
                else:
                    logits = self.model(batch)
                
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        logger.info(f"Test Accuracy: {acc:.4f}")
        logger.info(f"Test F1 (Macro): {f1_macro:.4f}")
        logger.info(f"Test F1 (Weighted): {f1_weighted:.4f}")
        
        # Classification report
        label_names = [self.idx_to_label[i] for i in range(len(self.label_map))]
        report = classification_report(
            all_labels, all_preds, 
            target_names=label_names, 
            digits=4,
            zero_division=0
        )
        
        # Save report
        report_dir = Path('reports')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f'classification_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Combined Spatial-Temporal GNN\n")
            f.write(f"Test Samples: {len(all_labels)}\n")
            f.write(f"Num Classes: {len(self.label_map)}\n\n")
            f.write(f"{'='*80}\n")
            f.write(f"OVERALL METRICS\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Accuracy:              {acc:.4f}\n")
            f.write(f"F1 Score (Macro):      {f1_macro:.4f}\n")
            f.write(f"F1 Score (Weighted):   {f1_weighted:.4f}\n")
            f.write(f"Precision (Macro):     {precision_macro:.4f}\n")
            f.write(f"Recall (Macro):        {recall_macro:.4f}\n\n")
            f.write(f"{'='*80}\n")
            f.write(f"PER-CLASS METRICS\n")
            f.write(f"{'='*80}\n\n")
            f.write(report)
        
        logger.info(f"\nSaved report to {report_file}")
        logger.info(f"\n{report}")
        
        return acc, f1_macro, f1_weighted


def main():
    # Load Configuration (defaults to v5)
    import importlib.util
    import sys
    
    # Check if config file is specified via command line
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        config_name = sys.argv[1]
    else:
        config_name = 'v5.py'  # Default to v5
    
    config_path = Path(__file__).parent.parent / 'configs' / config_name
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
        logger.info(f"✅ Loaded config from {config_path}")
    else:
        # Fallback to V4 balanced config (hardcoded)
        logger.warning(f"Config file not found at {config_path}, using hardcoded V4 config")
        config = {
            'data_dir': Path('data/processed'),
            'batch_size': 32,
            'num_workers': 8,
            'lr': 7.5e-4,  # Between V2 (1e-3) and V3 (5e-4)
            'num_epochs': 1000,
            'spatial_dim': 192,  # Between V2 (256) and V3 (128)
            'temporal_dim': 192,
            'spatial_layers': 2,
            'temporal_layers': 2,
            'dropout': 0.6,  # Between V2 (0.5) and V3 (0.7)
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': 'outputs/v4',
            'checkpoint_interval': 100,
            'early_stopping_patience': 200,  # Very patient
            'early_stopping_min_delta': 0.0005,
            'label_smoothing': 0.15,  # Between V2 (0.1) and V3 (0.2)
            'weight_decay': 5e-4,  # Between V2 (2e-4) and V3 (1e-3)
            'enable_log_server': False,
            'log_server': {
                'host': '0.0.0.0',
                'port': 8080,
                'entries': 10,
                'refresh_minutes': 5,
            },
        }
    
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION")
    logger.info("="*60)
    for k, v in config.items():
        logger.info(f"{k:20s}: {v}")
    logger.info("="*60)
    
    log_path = Path(config['save_dir']) / 'training.log'
    setup_file_logging(log_path)
    log_server_info = None
    if config.get('enable_log_server', False):
        log_cfg = config.get('log_server', {})
        host = log_cfg.get('host', '127.0.0.1')
        port = log_cfg.get('port', 8080)
        entries = log_cfg.get('entries', 10)
        refresh_minutes = log_cfg.get('refresh_minutes', 5)
        server, thread = start_log_server(
            log_path=log_path,
            host=host,
            port=port,
            entries=entries,
            refresh_minutes=refresh_minutes
        )
        log_server_info = (server, thread)
        logger.info(f"🌐 Log dashboard available at http://{host}:{port}/ (auto-refresh every {refresh_minutes} min)")
    
    # Load data
    train_loader, val_loader, test_loader, num_classes, label_map = create_dataloaders(
        train_pt=str(config['data_dir'] / 'train.pt'),
        val_pt=str(config['data_dir'] / 'val.pt'),
        test_pt=str(config['data_dir'] / 'test.pt'),
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Get input dim
    sample = next(iter(train_loader))
    input_dim = sample.x.shape[1]
    
    logger.info(f"\nData Info:")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Num classes: {num_classes}")
    logger.info(f"  Sample batch shape: {sample.x.shape}")
    
    # Create model based on config
    model_type = config.get('model_type', 'combined')
    if model_type == 'ast_gcn':
        logger.info("Using AST-GCN Model (Attention-based Spatial-Temporal)")
        model = ASTGCNModel(
            input_dim=input_dim,
            num_classes=num_classes,
            spatial_dim=config.get('spatial_dim', 256),
            temporal_dim=config.get('temporal_dim', 256),
            spatial_layers=config.get('spatial_layers', 3),
            temporal_layers=config.get('temporal_layers', 2),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.5)
        )
    else:
        logger.info("Using Combined Model (Simplified)")
        model = CombinedModel(
            input_dim=input_dim,
            num_classes=num_classes,
            spatial_dim=config.get('spatial_dim', 256),
            temporal_dim=config.get('temporal_dim', 256),
            spatial_layers=config.get('spatial_layers', 3),
            temporal_layers=config.get('temporal_layers', 2),
            dropout=config.get('dropout', 0.5)
        )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=torch.device(config['device']),
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir'],
        label_map=label_map,
        checkpoint_interval=config['checkpoint_interval'],
        label_smoothing=config.get('label_smoothing', 0.0),
        early_stopping_patience=config.get('early_stopping_patience', 50),
        early_stopping_min_delta=config.get('early_stopping_min_delta', 0.001),
        weight_decay=config.get('weight_decay', 1e-3)
    )
    
    trainer.train_all()
    trainer.test()
    
    logger.info("\n✅ ALL COMPLETE!")


if __name__ == "__main__":
    main()