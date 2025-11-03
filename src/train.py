# Training script for lip reading GNN
import torch
from pathlib import Path
from torch_geometric.data import DataLoader
from models.lip_reading_gnn import LipReadingGNN
from utils.dataset import LipReadingDataset
from utils.trainer import Trainer

def main():
    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # Setup data
    train_dataset = LipReadingDataset(root='data/processed', split='train')
    val_dataset = LipReadingDataset(root='data/processed', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LipReadingGNN(
        num_node_features=3,  # x, y, z coordinates
        num_classes=len(train_dataset.classes)
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
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