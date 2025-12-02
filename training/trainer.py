"""
Trainer class for model training and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class Trainer:
    """Trainer for GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        logger = None,
        epoch_callback: Optional[callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to use
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.epoch_callback = epoch_callback
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopping_patience = None
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': [],
            'lr': []
        }
    
    def train_epoch(self, gradient_clip: Optional[float] = None) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            gradient_clip: Maximum gradient norm for clipping (optional)
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Disable tqdm progress bar - only log at epoch completion
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", disable=True, miniters=len(self.train_loader))
        
        for batch_idx, (features, speech_mask, adj, labels) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            speech_mask = speech_mask.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features, adj, speech_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (if specified)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Disable tqdm progress bar - only log at epoch completion
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]", disable=True, miniters=len(self.val_loader))
        
        for batch_idx, (features, speech_mask, adj, labels) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            speech_mask = speech_mask.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(features, adj, speech_mask)
            loss = self.criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        save_dir: Path,
        early_stopping_patience: Optional[int] = None,
        gradient_clip: Optional[float] = None
    ) -> Dict:
        """
        Train model for specified epochs.
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = early_stopping_patience
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("STARTING TRAINING")
            self.logger.info("=" * 60)
            self.logger.info(f"Epochs: {num_epochs}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
            if early_stopping_patience:
                self.logger.info(f"Early stopping patience: {early_stopping_patience}")
            if gradient_clip:
                self.logger.info(f"Gradient clipping: {gradient_clip}")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(gradient_clip=gradient_clip)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Save best model (based on validation loss for early stopping)
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                improved = True
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'model_config': self.model.get_config(),
                }
                
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(checkpoint, save_dir / 'best.pth')
                
                if self.logger:
                    self.logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                if self.early_stopping_patience and self.patience_counter >= self.early_stopping_patience:
                    if self.logger:
                        self.logger.info(f"Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                        self.logger.info(f"Best model was at epoch {self.best_epoch} (val_loss: {self.best_val_loss:.4f}, val_acc: {self.best_val_acc:.2f}%)")
                    break
            
            # Save latest model after each epoch (for ability to stop anytime)
            latest_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'model_config': self.model.get_config(),
                'best_val_acc': self.best_val_acc,
                'best_epoch': self.best_epoch,
            }
            
            if self.scheduler:
                latest_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(latest_checkpoint, save_dir / 'latest_model.pth')
            
            # Save history after each epoch (for real-time monitoring)
            self.save_history(save_dir / 'history.pt')
            
            # Generate visualization after each epoch
            try:
                self._generate_loss_history_plot(save_dir)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to generate loss history plot: {e}")
            
            # Call callback if provided (for saving metadata incrementally)
            if hasattr(self, 'epoch_callback') and self.epoch_callback:
                try:
                    self.epoch_callback(epoch, save_dir, {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'best_val_acc': self.best_val_acc,
                        'best_epoch': self.best_epoch,
                    })
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Epoch callback failed: {e}")
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("TRAINING COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Best val accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        return self.history
    
    def save_history(self, save_path: Path) -> None:
        """Save training history."""
        torch.save(self.history, save_path)
    
    def _generate_loss_history_plot(self, save_dir: Path) -> None:
        """Generate loss history plot after each epoch."""
        if len(self.history['train_loss']) == 0:
            return
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            
            # Plot loss
            if 'train_loss' in self.history and 'val_loss' in self.history:
                axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
                axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
                axes[0].set_xlabel('Epoch', fontsize=12)
                axes[0].set_ylabel('Loss', fontsize=12)
                axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)
            
            # Plot accuracy
            if 'train_acc' in self.history and 'val_acc' in self.history:
                axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
                axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
                axes[1].set_xlabel('Epoch', fontsize=12)
                axes[1].set_ylabel('Accuracy (%)', fontsize=12)
                axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            output_file = save_dir / 'loss_history.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Silently fail if plotting fails (non-critical)
            pass

