# Project Folder Structure

## Overview
Organized folder structure with consistent versioning for models, configs, and outputs.

## Directory Structure

```
IndonesiaLipReading_GNN/
├── configs/              # Model configurations (versioned)
│   ├── v1.py            # (if exists)
│   ├── v2.py            # (if exists)
│   ├── v3.py            # Simplified model
│   ├── v4.py            # Balanced model
│   └── v5.py            # AST-GCN with advanced features
│
├── data/                # Dataset and processed data
│   ├── IDLRW-DATASET/   # Raw video dataset
│   ├── processed/       # V1 processed data
│   └── processed_v2/    # V2 processed data (current)
│
├── outputs/             # Training outputs (versioned)
│   ├── v1/              # V1 model outputs
│   ├── v2/              # V2 model outputs
│   ├── v3/              # V3 model outputs
│   ├── v4/              # V4 model outputs
│   └── v5/              # V5 AST-GCN outputs
│       ├── best_model.pth
│       ├── checkpoints/
│       ├── training.log
│       ├── train_stdout.log
│       └── train_stderr.log
│
├── src/                 # Source code
│   ├── models/          # Model architectures
│   │   ├── combined.py  # V1-V4: Combined model
│   │   ├── ast_gcn.py   # V5: AST-GCN model
│   │   ├── spatial.py   # Spatial GCN
│   │   └── temporal.py  # Temporal LSTM
│   ├── dataset/         # Dataset loaders
│   ├── preprocessing/   # Feature extraction
│   ├── debug/           # Debugging tools
│   ├── train.py         # Training script
│   └── eval.py          # Evaluation script
│
├── scripts/             # Utility scripts
│   ├── run_v5_training.sh    # Run V5 training (survives disconnection)
│   ├── stop_training.sh      # Stop all training
│   └── ...
│
└── docs/                 # Documentation
    └── reports/         # Analysis reports
```

## Version Naming Convention

### Configs
- `configs/v{N}.py` - Configuration for version N
- Example: `v5.py` = V5 AST-GCN configuration

### Outputs
- `outputs/v{N}/` - Output directory for version N
- Contains: models, checkpoints, logs

### Models
- `src/models/combined.py` - V1-V4: Combined spatial-temporal model
- `src/models/ast_gcn.py` - V5: Attention-based AST-GCN model

## Data Versions

- `data/processed/` - V1 processed data (10 geometric features)
- `data/processed_v2/` - V2 processed data (15 geometric features + motion)

## Usage

### Run Training
```bash
# V5 (default)
python src/train.py

# Specific version
python src/train.py v4.py

# With background script (survives disconnection)
./scripts/run_v5_training.sh
```

### Check Outputs
```bash
# View training log
tail -f outputs/v5/training.log

# Check model
ls outputs/v5/
```

## Version History

- **V1**: Initial model
- **V2**: Overfitting issues
- **V3**: Simplified with strong regularization
- **V4**: Balanced between V2 and V3
- **V5**: AST-GCN with advanced features (Gabor, FFT, Recurrence, Multi-scale, Relative motion)


