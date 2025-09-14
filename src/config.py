import os

# Base paths
DATA_DIR = "data/IDLWR-DATASET"
OUTPUT_DIR = "outputs"
LANDMARKS_DIR = os.path.join(OUTPUT_DIR, "landmarks")

# Dataset settings
IMG_SIZE = 224
FPS = 25
NUM_CLASSES = 100
MAX_SEQ_LEN = 25  # pad/truncate to 25 frames (≈1 sec)

# Training settings
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 50
DEVICE = "mps"  # for Mac M1 GPU
