# Project Reorganization Plan
Based on `.cursorrules` requirements

## Current vs Required Structure

### Current Structure (WRONG):
```
/project
  /data
    /IDLRW-DATASET/
    /processed_v3/          # ❌ Should be in extracted/
  /outputs/                 # ❌ Should be split into results/, extracted/, features/, graphs/
    /v1, v2, v3, v4, v5/   # ❌ Old results
    /landmark_previews/     # ❌ Should be in extracted/*/example_checks/
  /configs/                 # ❌ Should be in models/configs/
  /scripts/
  /src/
```

### Required Structure (from .cursorrules):
```
/project
  /data
    /raw/                   # original videos (read-only) - link to IDLRW-DATASET
    /IDLRW-DATASET/         # Keep as is
    /extracted/             # NEW: outputs from facemesh_extractor
      /lips/
      /mouth_area/
      /full_face/
    /features/              # NEW: B0, B1, B2, B3, B4 feature sets
      /B0/
      /B1/
      /B2/
      /B3/
      /B4/
    /graphs/                # NEW: adjacency matrices
      /lips/
      /mouth_area/
      /full_face/
  /models
    /checkpoints/           # NEW: model checkpoints
    /configs/               # MOVE from root configs/
  /results/                 # NEW: experiment results
    /<partition>/<FE>/<model>/<seed>/
  /logs/                    # NEW: experiment registry
  /scripts/                 # Keep as is
  /src/                     # Keep as is
  /archive/                 # NEW: old outputs moved here
```

## Reorganization Steps

### Step 1: Create New Directory Structure
```bash
mkdir -p data/extracted/{lips,mouth_area,full_face}/example_checks
mkdir -p data/features/{B0,B1,B2,B3,B4}
mkdir -p data/graphs/{lips,mouth_area,full_face}
mkdir -p models/{checkpoints,configs}
mkdir -p results
mkdir -p logs
```

### Step 2: Move/Archive Old Outputs
```bash
# Archive old outputs
mv outputs/v1 outputs/v2 outputs/v3 outputs/v4 archive/old_outputs/
# Keep v5 temporarily for reference, move later

# Move landmark_previews to extracted/full_face/example_checks/
mv outputs/landmark_previews/* data/extracted/full_face/example_checks/ 2>/dev/null || true
```

### Step 3: Move Configs
```bash
mv configs/* models/configs/
rmdir configs
```

### Step 4: Move processed_v3 to extracted
```bash
# processed_v3 contains train.pt, val.pt, test.pt
# These should be moved to extracted/full_face/ or kept as consolidated dataset
# Decision: Keep as consolidated dataset files, but also extract per-video
```

### Step 5: Update Code Paths
- Update `facemesh_extractor.py` output paths
- Update `train.py` data loading paths
- Update `dataset.py` paths
- Update all scripts

### Step 6: Update .gitignore
Add:
```
# Results and outputs
results/
data/extracted/
data/features/
data/graphs/
models/checkpoints/
logs/
outputs/
*.pt  # except in data/processed_v3/ (temporary)
```

## Files to Keep/Delete

### KEEP:
- `data/IDLRW-DATASET/` - original dataset
- `data/processed_v3/` - consolidated dataset (temporary, will be replaced)
- `src/` - all source code
- `scripts/` - all scripts
- `.cursorrules` - project rules
- `README.md` - documentation

### ARCHIVE (move to archive/):
- `outputs/v1/`, `outputs/v2/`, `outputs/v3/`, `outputs/v4/` - old results
- `outputs/landmark_previews/` - move to extracted/full_face/example_checks/
- `outputs/extraction/` - old extraction logs

### DELETE (after verification):
- `__MACOSX/` - macOS metadata
- `archive/old_outputs/` - after confirming backup
- Temporary files: `nohup.out`, `log_server.out`

## Git Safety

1. **Create backup branch first:**
   ```bash
   git checkout -b backup-before-reorganization
   git add -A
   git commit -m "Backup before reorganization"
   git checkout main
   ```

2. **Stage changes incrementally:**
   - Create directories first
   - Move files
   - Update code
   - Test
   - Commit

3. **Don't commit large files:**
   - Add `results/`, `data/extracted/`, etc. to .gitignore
   - Only commit structure and code changes

## Testing After Reorganization

1. Verify extraction still works
2. Verify training can load data
3. Verify paths in all scripts
4. Run a small test training run

