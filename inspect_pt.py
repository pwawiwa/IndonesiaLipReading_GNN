
import torch

# Replace with the path to your .pt file
pt_file = "/Users/wirahutomo/Projects/TA_IDLR_GNN/preprocessed/train/ada_00001.pt"

# Allow loading full object
data = torch.load(pt_file, weights_only=False)

print("Keys in .pt file:", data.keys())
print("Word label:", data['word'])
print("Landmark tensor shape:", data['nodes'].shape)
print("First frame landmarks:\n", data['nodes'][0])

