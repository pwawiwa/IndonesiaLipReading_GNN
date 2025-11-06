import torch
data = torch.load("data/processed/train.pt", weights_only=False)
print(data[0])
