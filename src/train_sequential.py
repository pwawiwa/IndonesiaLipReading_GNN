# Training script for sequential lip reading GNN
import torch
from pathlib import Path
from torch_geometric.data import DataLoader
from models.sequence_gnn import TemporalGNN
from utils.sequential_dataset import SequentialLipReadingDataset
from utils.trainer import Trainer

def main():
    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    batch_size = 16  # Smaller batch size due to sequence processing
    learning_rate = 0.001
    hidden_dim = 128
    sequence_length = 30  # Fixed sequence length
    
    # Setup data
    train_dataset = SequentialLipReadingDataset(
        root='data/processed',
        split='train',
        sequence_length=sequence_length
    )
    val_dataset = SequentialLipReadingDataset(
        root='data/processed',
        split='val',
        sequence_length=sequence_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TemporalGNN(
        num_node_features=3,  # x, y, z coordinates
        hidden_dim=hidden_dim,
        num_classes=len(train_dataset.classes),
        num_gnn_layers=3,
        lstm_layers=2,
        dropout=0.3
    ).to(device)
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Path('checkpoints')
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )

if __name__ == '__main__':
    main()