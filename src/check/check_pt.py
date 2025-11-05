import torch

# Load the processed facemesh data
data = torch.load("data/processed/train.pt", weights_only=False)


# Now inspect it
print(f"Total videos: {len(data)}")

sample = data[0]
print("Keys:", sample.keys())
print("Video name:", sample["video"])
print("Landmarks shape:", sample["landmarks"].shape)
