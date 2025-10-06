import torch
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

data, slices = torch.load(
    "/Users/wirahutomo/Projects/TA_IDLR_GNN/data/landmarks/processed/train.pt",
    weights_only=False
)

print("Data object:", data)
print("Slices keys:", slices.keys())

# Ambil semua labels dari data.y pakai slices
labels = data.y.tolist()
print("Total samples:", len(labels))
print("Unique labels:", len(set(labels)))
print("Label range:", min(labels), "to", max(labels))
