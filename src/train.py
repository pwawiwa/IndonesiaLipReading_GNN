import torch
from torch_geometric.loader import DataLoader
from dataset import LipReadingDataset
from models import GCN  # or GAT
from eval import evaluate
from pathlib import Path
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_dir = Path(__file__).parent.parent / "data" / "processed"

    train_dataset = LipReadingDataset(str(processed_dir / "train.pt"))
    val_dataset = LipReadingDataset(str(processed_dir / "val.pt"))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Infer input channels from first sample
    sample = train_dataset[0]
    in_channels = sample.x.shape[1]
    num_classes = len(train_dataset.word2idx)

    model = GCN(in_channels, hidden_channels=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
