import argparse
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import importlib
import torch.nn.functional as F
from torch import nn

# ---------- TRAIN LOOP ----------
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return correct / total

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="Train Lip Reading GNN")
    parser.add_argument("--model", type=str, default="gcn", help="model module name in models/")
    parser.add_argument("--train_file", type=str, default="data/graphs/train_graphs.pt")
    parser.add_argument("--val_file", type=str, default="data/graphs/val_graphs.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------- Load dataset ----------
    print("🔹 Loading graphs...")
    train_graphs = torch.load(args.train_file)
    val_graphs = torch.load(args.val_file)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)

    # ---------- Import model dynamically ----------
    model_module = importlib.import_module(f"models.{args.model}")
    model = model_module.get_model()  # each model.py must implement get_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        loss = train(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{args.model}.pt")
            print(f"💾 Best model saved at epoch {epoch}")

    print("🏁 Training complete. Best val acc:", best_val_acc)

if __name__=="__main__":
    main()
