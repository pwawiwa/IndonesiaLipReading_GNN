# Full Face Pipeline - Focused Approach

## Overview

This pipeline focuses exclusively on **full face extraction (468 MediaPipe nodes)** with comprehensive feature engineering (B0-B4) and model evaluation.

## Pipeline Strategy

```
1. Extraction → MediaPipe Default Topology
   ├─ All 468 MediaPipe FaceMesh nodes
   ├─ FACEMESH_TESSELATION (complete mesh, ~900+ edges)
   └─ Run: bash scripts/extract_full_face.sh

2. Feature Engineering (B0-B4) → per .cursorrules
   ├─ B0: Raw normalized coordinates
   ├─ B1: B0 + velocity
   ├─ B2: B1 + acceleration + speed
   ├─ B3: B2 + geometric features
   └─ B4: B3 + AU features + PCA + motion energy

3. Model Training (5 models per feature set)
   ├─ GCN
   ├─ GAT
   ├─ GraphSAGE
   ├─ ST-GCN
   └─ GNN-LSTM

4. GradCAM Evaluation (automatic after each model)
   ├─ Node importance visualization
   ├─ Per-class aggregated heatmaps
   ├─ Sample visualizations
   └─ Summary statistics saved

5. Storage Efficiency
   └─ Delete features after training each set
```

## Total Scenarios

**25 training runs:**
- 1 partition (full face)
- 5 feature sets (B0-B4)
- 5 models per feature set
- = 1 × 5 × 5 = **25 runs**

## Usage

### Step 1: Extract Full Face (if not done or updating topology)

```bash
cd /home/member2/tomoooo/IndonesiaLipReading_GNN

# Extract all splits with MediaPipe default FACEMESH_TESSELATION
bash scripts/extract_full_face.sh

# This will:
# - Backup existing extraction (if present)
# - Extract train/val/test with 468 nodes + ~900 edges
# - Verify node and edge counts
# - Takes ~30-60 minutes for full dataset
```

### Step 2: Start Pipeline

```bash
# Run full pipeline in background
nohup bash scripts/run_full_face_pipeline.sh > logs/fullface_main.log 2>&1 &

# Monitor progress
tail -f logs/fullface_pipeline_*.log
```

### Stop Pipeline

```bash
bash scripts/stop_full_face_pipeline.sh
```

### Check Progress

```bash
# See completed runs
find results/full -name "best.pth" | wc -l
# Should show: X / 25 when complete

# View live results table
column -t -s, results/full/results_table.csv | less -S

# Quick summary
tail -50 logs/fullface_pipeline_*.log
```

### View Results

```bash
# After completion, view summary
cat results/full/summary/summary.txt

# View visualization
open results/full/summary/results_summary.png
# or
eog results/full/summary/results_summary.png

# View results table
column -t -s, results/full/results_table.csv
```

## Results Structure

```
results/full/
├── B0/
│   ├── gcn/seed_0/
│   │   ├── best.pth                    # Best model checkpoint ✓
│   │   ├── last.pth                    # Last epoch checkpoint ✓
│   │   ├── history.pt                  # Training history
│   │   ├── loss_history.png            # Loss/accuracy curves ✓
│   │   ├── run_meta.pt                 # Metrics & metadata
│   │   ├── inference_test.pt           # Sample inference test ✓
│   │   ├── config.yaml                 # Config used
│   │   ├── confusion_matrix.png        # Confusion matrix
│   │   └── gradcam/                    # GradCAM visualizations
│   │       ├── samples/                # Per-sample heatmaps
│   │       ├── aggregated/             # Per-class aggregated
│   │       └── gradcam_summary.pt      # Summary statistics
│   ├── gat/seed_0/
│   ├── graphsage/seed_0/
│   ├── stgcn/seed_0/
│   └── gnn_lstm/seed_0/
├── B1/
│   └── (same 5 models)
├── B2/
│   └── (same 5 models)
├── B3/
│   └── (same 5 models)
├── B4/
│   └── (same 5 models)
├── results_table.csv                   # All results in table format
└── summary/
    ├── summary.txt                      # Text summary report
    └── results_summary.png              # Visualization (4 plots)
```

**Storage Efficiency:** Only best.pth and last.pth are kept; intermediate epoch checkpoints are automatically deleted.

## Evaluation Outputs

### Per-Model Results
Each trained model produces:
- **best.pth**: Best model checkpoint (validation)
- **last.pth**: Last epoch checkpoint
- **run_meta.pt**: Test accuracy, F1, validation scores
- **history.pt**: Training history (losses, accuracies)
- **loss_history.png**: Loss/accuracy curves visualization
- **confusion_matrix.png**: Classification confusion matrix
- **inference_test.pt**: Sample inference test result
- **gradcam/**: Node importance visualizations

**Note:** Only best.pth and last.pth are kept; intermediate checkpoints are automatically deleted to save storage.

### Aggregate Results
After all models complete:
- **results_table.csv**: CSV with all metrics
- **summary.txt**: Text report with best models per feature set
- **results_summary.png**: 4-panel visualization showing:
  1. Accuracy heatmap (models × features)
  2. F1 score heatmap (models × features)
  3. Model comparison (averaged across features)
  4. Feature set comparison (averaged across models)

## Feature Set Details (per .cursorrules)

### B0 - Baseline
- Raw normalized landmark coordinates (X, Y per node)
- Tests: Can geometrical raw coordinates separate visemes?

### B1 - Velocity
- B0 + first derivative (velocity)
- Tests: Does motion improve separability?

### B2 - Acceleration
- B1 + second derivative + speed magnitude
- Tests: Do acceleration and speed add discriminative power?

### B3 - Geometry
- B2 + pairwise distances, angles, ratios
- Tests: Do explicit geometric relations reduce confusion?

### B4 - Full Features
- B3 + AU-like features + PCA + motion energy
- Tests: Do higher-level articulatory patterns improve robustness?

## Configuration

**Moderate VRAM settings:**
- Batch size: 16
- Hidden dim: 128
- Epochs: 50 (with early stopping patience=10)
- Learning rate: 0.001
- Workers: 2

**Automatic Evaluations per Model:**
1. **Loss Visualization**: Training/validation loss and accuracy curves
2. **Inference Test**: Sample inference with timing (ms) and confidence
3. **Metrics Table**: CSV row with test accuracy, F1 score
4. **GradCAM**: Node importance heatmaps per class
5. **Checkpoint Cleanup**: Keep only best.pth and last.pth

## Expected Timeline

Per feature set: ~2-4 hours (5 models)
Total: ~10-20 hours for all 25 runs

## Why Full Face?

1. **Most comprehensive** - 468 nodes capture full facial articulation
2. **Default MediaPipe** - Standard, validated landmark detection
3. **Research completeness** - Covers full facial dynamics for lip reading
4. **Generalizable** - Can subset to lips/mouth later if needed

## Next Steps

After completion:
1. Analyze results across B0-B4 feature sets
2. Compare model performance (GCN, GAT, etc.)
3. Identify best feature set + model combination
4. Generate visualizations and reports

