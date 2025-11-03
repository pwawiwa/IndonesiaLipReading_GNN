# Training utility class
import torch
from pathlib import Path
from datetime import datetime

class Trainer:
    def __init__(self, model, optimizer, device, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = torch.nn.functional.nll_loss(out, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = out.max(1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
            
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.functional.nll_loss(out, batch.y)
                
                total_loss += loss.item()
                pred = out.max(1)[1]
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
                
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_checkpoint(
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    accuracy=val_acc,
                    path=self.checkpoint_dir / f'best_model_{datetime.now():%Y%m%d_%H%M%S}.pt'
                )
            
            # Print progress
            print(f'Epoch {epoch:03d}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')