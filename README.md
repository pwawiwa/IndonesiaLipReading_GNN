# Indonesia Lip Reading with GNN

## Project Structure

```
IndonesiaLipReading_GNN/
├── analysis/              # Analysis and debugging documents
│   ├── DEBUGGING_ANALYSIS.md
│   ├── PREPROCESSING_CHECKLIST.md
│   ├── MODEL_COMPLEXITY.md
│   └── checkpoints/       # Checkpoint analysis files
├── archive/              # Archived old outputs and files
│   ├── old_outputs/
│   └── old_reports/
├── data/                  # Dataset and processed data
│   ├── IDLRW-DATASET/    # Raw video dataset
│   ├── processed/        # Processed .pt files (v1)
│   └── processed_v2/    # Processed .pt files (v2)
├── docs/                  # Documentation and reports
├── outputs/              # Training outputs
│   └── combined/        # Current training run
├── reports/              # Evaluation reports
├── scripts/              # Utility scripts
└── src/                  # Source code
    ├── dataset/          # Dataset loading
    ├── models/           # Model architectures
    ├── preprocessing/    # Feature extraction
    ├── utils/            # Utilities
    ├── train.py          # Training script
    └── eval.py           # Evaluation script
```

## Current Status

### ⚠️ Debugging Phase

**Issues Identified:**
1. **Severe Overfitting**: Train acc 91% vs Val acc 11%
2. **Model Too Complex**: 33M parameters
3. **Speech Mask Not Used**: Extracted but not applied
4. **Edge Connections Mismatch**: Anatomical edges lost

See `analysis/DEBUGGING_ANALYSIS.md` for detailed analysis.

## Quick Start

### Preprocessing
```bash
cd src/preprocessing
python facemesh_extractor.py
```

### Training
```bash
cd src
python train.py
```

### Evaluation
```bash
cd src
python eval.py
```

## Key Files

- `analysis/DEBUGGING_ANALYSIS.md` - Complete debugging analysis
- `analysis/MODEL_COMPLEXITY.md` - Model architecture analysis
- `analysis/PREPROCESSING_CHECKLIST.md` - Preprocessing verification
- `src/train.py` - Main training script
- `src/models/combined.py` - Combined spatial-temporal model

## Next Steps

1. ✅ Analysis complete
2. ⏳ Simplify model architecture
3. ⏳ Fix speech mask usage
4. ⏳ Fix edge connections
5. ⏳ Retrain with fixes



