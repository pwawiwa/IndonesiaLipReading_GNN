import torch
from torch.utils.data import DataLoader
from dataset import IDLRWDataset
from models import BaseGCN
from eval import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    train_ds = IDLRWDataset("data/processed/train.pt")
    val_ds = IDLRWDataset("data/processed/val.pt")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    
    # Create model (adjust input/output channels as needed)
    # For now, features flattened vector size example:
    sample_landmarks, sample_features, _ = train_ds[0]
    in_channels = sample_features.shape[0]
    hidden_channels = 128
    out_channels = 100  # number of words / labels (adjust)
    
    model = BaseGCN(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()  # assumes labels mapped to int
    
    # Training loop
    for epoch in range(5):  # small epochs for test
        model.train()
        for landmarks, features, labels in train_loader:
            features = features.to(device)
            # labels = labels_tensor.to(device) # map string -> int first
            
            optimizer.zero_grad()
            # placeholder: edge_index None
            out = model(features, edge_index=None)
            # loss = criterion(out, labels)
            # loss.backward()
            # optimizer.step()
        
        print(f"Epoch {epoch} done")
        
        # Evaluate
        evaluate(model, val_loader, device=device)

if __name__ == "__main__":
    main()
