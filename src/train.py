import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from dataset.lipreading_dataset import LipReadingDataset
from gnn.lipreading_model import GraphTemporalGNN


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device, epoch, split="Val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# --- Penambahan Kelas EarlyStopping ---
class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="limit number of samples per split (debugging)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait for improvement before early stopping")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    landmark_root = "data/landmarks"

    print("Processing...")
    train_ds = LipReadingDataset(root=landmark_root, split="train")
    val_ds = LipReadingDataset(root=landmark_root, split="val")

    if args.subset:
        train_ds = train_ds[:args.subset]
        val_ds = val_ds[:args.subset]

    all_labels = [d.y.item() for d in train_ds]
    num_classes = int(max(all_labels) + 1)
    print("Num classes detected:", num_classes)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = GraphTemporalGNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # --- Inisialisasi Early Stopping ---
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path='best_model_checkpoint.pt')

    print("Start training...")
    for epoch in range(1, 21):  # Ubah jumlah epoch menjadi lebih banyak, misal 100
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, split="Val")

        print(f"[Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # --- Cek kondisi early stopping ---
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    # --- Muat kembali model terbaik ---
    print("Loading the best model found during training...")
    model.load_state_dict(torch.load('best_model_checkpoint_6okt25.pt'))
    
    print("Training finished.")

if __name__ == "__main__":
    main()