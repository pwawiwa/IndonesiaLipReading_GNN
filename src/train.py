"""
src/train.py
Main training script with comprehensive tracking and visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

from dataset import create_dataloaders
from models import create_model
from eval import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value: float) -> bool:
        """
        Check if should stop
        
        Args:
            value: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True
        
        return False


class Trainer:
    """Main training class"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        model_name: str = "model",
        save_dir: Path = Path("outputs")
    ):
        """
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            device: Device
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            model_name: Name for saving
            save_dir: Directory for outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.save_dir = save_dir
        
        # Create save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.save_dir / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler with warmup
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Warmup + Cosine Annealing
        self.warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=learning_rate * 0.01
        )
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=8
        )
        
        # Loss function with class weights for imbalanced data
        self.criterion = self._create_weighted_loss(train_loader)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        # Evaluator
        num_classes = train_loader.dataset.num_classes
        class_names = list(train_loader.dataset.label_to_idx.keys())
        self.evaluator = Evaluator(num_classes, class_names)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"Trainer initialized. Outputs will be saved to {self.run_dir}")
    
    def _create_weighted_loss(self, train_loader) -> nn.Module:
        """
        Create weighted cross-entropy loss for imbalanced classes
        
        Args:
            train_loader: Training dataloader
            
        Returns:
            Weighted CrossEntropyLoss
        """
        logger.info("Computing class weights for imbalanced dataset...")
        
        # Count samples per class
        class_counts = torch.zeros(train_loader.dataset.num_classes)
        for data in train_loader.dataset:
            class_counts[data.y.item()] += 1
        
        # Compute weights: inverse of frequency
        # Use effective number of samples for better weighting
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        logger.info(f"Class weight statistics:")
        logger.info(f"  Min weight: {weights.min():.4f}")
        logger.info(f"  Max weight: {weights.max():.4f}")
        logger.info(f"  Mean weight: {weights.mean():.4f}")
        logger.info(f"  Samples per class - Min: {class_counts.min():.0f}, Max: {class_counts.max():.0f}")
        
        return nn.CrossEntropyLoss(weight=weights.to(self.device))
        
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation
        accumulation_steps = 2
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for i, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y.squeeze(-1))
            loss = loss / accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps * batch.num_graphs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch.y.squeeze(-1)).sum().item()
            total += batch.num_graphs
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'acc': correct / total
            })
        
        # Final update if there are remaining gradients
        if (i + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        metrics = {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
        
        return metrics
    
    def validate(self) -> Dict:
        """Validate on validation set"""
        metrics = self.evaluator.evaluate(
            self.model,
            self.val_loader,
            self.device,
            self.criterion
        )
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(epoch + val_metrics['loss'] / 100)  # Cosine annealing
            self.plateau_scheduler.step(val_metrics['loss'])  # Plateau reduction
            new_lr = self.optimizer.param_groups[0]['lr']
            if abs(old_lr - new_lr) > 1e-8:
                logger.info(f"Learning rate: {old_lr:.6f} → {new_lr:.6f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                logger.info(f"✓ New best model saved (Val Loss: {val_metrics['loss']:.4f})")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth', epoch, val_metrics)
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best model at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.4f}")
        
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        save_path = self.run_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.run_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train Acc', marker='o')
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val Acc', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['val_f1'], label='Val F1 (Macro)', marker='s', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], label='Learning Rate', marker='o', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.run_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {plot_path}")
        plt.close()
    
    def test(self):
        """Test on test set"""
        logger.info("\n" + "="*80)
        logger.info("Testing on test set...")
        logger.info("="*80)
        
        # Load best model
        best_model_path = self.run_dir / "best_model.pth"
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate
        test_metrics = self.evaluator.evaluate(
            self.model,
            self.test_loader,
            self.device,
            self.criterion
        )
        
        # Print metrics
        self.evaluator.print_metrics(test_metrics, title="Test Set Results")
        
        # Save test metrics
        metrics_path = self.run_dir / "test_metrics.json"
        # Convert numpy arrays to lists for JSON serialization
        test_metrics_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in test_metrics.items()
            if k not in ['classification_report', 'per_class']
        }
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics_serializable, f, indent=4)
        logger.info(f"Test metrics saved to {metrics_path}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(test_metrics['confusion_matrix'])
        
        # Find worst and best cases
        worst_cases, best_cases = self.evaluator.find_worst_and_best_cases(
            self.model,
            self.test_loader,
            self.device,
            top_k=5
        )
        
        # Save cases
        self.save_cases(worst_cases, best_cases)
        
        return test_metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.evaluator.class_names,
            yticklabels=self.evaluator.class_names
        )
        
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.run_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {plot_path}")
        plt.close()
    
    def save_cases(self, worst_cases: List, best_cases: List):
        """Save worst and best cases"""
        cases_path = self.run_dir / "cases_analysis.json"
        
        cases = {
            'worst_cases': worst_cases,
            'best_cases': best_cases
        }
        
        with open(cases_path, 'w') as f:
            json.dump(cases, f, indent=4)
        
        logger.info(f"Cases analysis saved to {cases_path}")
        
        # Print summary
        logger.info("\n[Worst Cases - High Confidence Errors]")
        for i, case in enumerate(worst_cases, 1):
            true_label = self.evaluator.class_names[case['true_label']]
            pred_label = self.evaluator.class_names[case['pred_label']]
            logger.info(
                f"{i}. {case['video_id']}: "
                f"True={true_label}, Pred={pred_label}, "
                f"Confidence={case['confidence']:.4f}"
            )
        
        logger.info("\n[Best Cases - High Confidence Correct]")
        for i, case in enumerate(best_cases, 1):
            true_label = self.evaluator.class_names[case['true_label']]
            logger.info(
                f"{i}. {case['video_id']}: "
                f"True={true_label}, "
                f"Confidence={case['confidence']:.4f}"
            )


def main():
    """Main function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'data_dir': Path('data/processed'),
        'model_type': 'combined',  # 'spatial', 'temporal', or 'combined'
        'batch_size': 32,  # Increased for better gradient estimates
        'num_workers': 4,
        'learning_rate': 5e-4,  # Reduced for stability
        'weight_decay': 1e-3,  # Increased regularization
        'num_epochs': 100,
        'early_stopping_patience': 20,  # Increased patience
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Model hyperparameters - Increased capacity for 100 classes
        'spatial_hidden_dim': 256,  # Increased
        'temporal_hidden_dim': 512,  # Increased
        'spatial_layers': 4,  # More layers
        'temporal_layers': 3,  # More layers
        'dropout': 0.5,  # Increased dropout
        'use_gat': False,
        'temporal_type': 'lstm'  # 'lstm' or 'attention'
    }
    
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, label_mapping, num_classes = create_dataloaders(
        train_pt=str(config['data_dir'] / 'train.pt'),
        val_pt=str(config['data_dir'] / 'val.pt'),
        test_pt=str(config['data_dir'] / 'test.pt'),
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        num_classes=num_classes,
        spatial_hidden_dim=config['spatial_hidden_dim'],
        temporal_hidden_dim=config['temporal_hidden_dim'],
        spatial_layers=config['spatial_layers'],
        temporal_layers=config['temporal_layers'],
        dropout=config['dropout'],
        use_gat=config['use_gat'],
        temporal_type=config.get('temporal_type', 'lstm')
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=torch.device(config['device']),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        model_name=config['model_type'],
        save_dir=Path('outputs')
    )
    
    # Train
    trainer.train()
    
    # Test
    trainer.test()
    
    logger.info("\n✓ Training and evaluation completed!")
    logger.info(f"Results saved to: {trainer.run_dir}")


if __name__ == "__main__":
    main()