```markdown
# TA_IDLR_GNN

**Indonesian Lip Reading using Graph Neural Networks (GNN)**

---

## Project Overview

This project implements a lip reading system using Graph Temporal GNNs on facial landmarks extracted from videos. The system processes video frames, extracts face and lip landmarks using MediaPipe FaceMesh, and trains a GNN to classify spoken words.

---

## Directory Structure

```

TA_IDLR_GNN/
├── data/
│   ├── IDLRW-DATASET         # Original video dataset
│   ├── landmarks/            # Extracted landmarks
│   │   ├── processed/        # Processed .pt files for PyG
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── idlrw-env/            # Environment (optional)
│   └── preprocessed/         # Optional preprocessing outputs
├── src/
│   ├── configs/
│   │   └── paths.yaml        # Dataset paths
│   ├── dataset/
│   │   ├── extract_landmarks.py
│   │   ├── facemesh_test.py
│   │   ├── lipreading_dataset.py
│   │   ├── preview_landmarks.py
│   │   └── utils_video.py
│   ├── gnn/
│   │   ├── facemesh_topology.py
│   │   └── lipreading_model.py
│   └── models/
│       └── base_gnn.py
│   └── train.py
├── baseline_model.pth        # Optional pretrained weights
├── README.md
└── requirements.txt

````

---

## Setup

```bash
# Create virtual environment
python -m venv lipreading-gnn
source lipreading-gnn/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## Preprocessing

1. Extract facial landmarks from videos:

```bash
python src/dataset/extract_landmarks.py --split train
python src/dataset/extract_landmarks.py --split val
python src/dataset/extract_landmarks.py --split test
```

2. Landmarks are saved as `.npz` files per video, and later collated into `.pt` files by `LipReadingDataset` for PyTorch Geometric.

---

## Training

```bash
python src/train.py --subset 100  # optional: limit samples for debugging
```

* Uses `GraphTemporalGNN` for temporal and spatial modeling of facial landmarks.
* Default batch size: 4, learning rate: 1e-3.
* Outputs training and validation loss/accuracy per epoch.

---

## Notes

* Focuses on mouth/lip region for lipreading tasks.
* Eye/other facial AU features can be added for broader affective analysis, but not required for lipreading.
* Consider using single pickle/pt files per split for efficiency in future preprocessing.

---

## License

This project is open-source for research and educational purposes.

```

I can also make it **more visually appealing with badges, usage examples, and training tips** if you want.  

Do you want me to do that next?
```
