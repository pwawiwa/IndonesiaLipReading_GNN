import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path


class LipReadingGraphDataset(Dataset):
    """
    Loads spatio-temporal facemesh graphs from .pt files created by GraphBuilderST.
    Each sample is a torch_geometric.data.Data object containing:
        x: [T*468, 3]
        edge_index: [2, E]
        num_frames: int
        video_name: str
    """

    def __init__(self, root_dir, split="train", transform=None, pre_transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.graph_path = self.root_dir / f"{split}_st_graphs.pt"
        self.graphs = torch.load(self.graph_path, weights_only=False)
        super().__init__(root=str(self.root_dir), transform=transform, pre_transform=pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


def get_dataloaders(graph_root="data/graphs", batch_size=8, shuffle=True):
    train_ds = LipReadingGraphDataset(graph_root, "train")
    val_ds = LipReadingGraphDataset(graph_root, "val")
    test_ds = LipReadingGraphDataset(graph_root, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick check
    train_loader, val_loader, test_loader = get_dataloaders()
    for batch in train_loader:
        print(batch)
        break
