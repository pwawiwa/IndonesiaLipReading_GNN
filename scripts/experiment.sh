#!/bin/bash
# ============================================
# BEST CONFIGURATION: Optimal hyperparameters for LSTM+Mamba models
# - Batch size: 32, Hidden dim: 256, Epochs: 100
# - 3 GNN layers, 3 LSTM layers, bidirectional, dropout 0.7
# - Learning rate: 0.0005, Weight decay: 0.0001
# - Scheduler: ReduceLROnPlateau (factor 0.7, patience 5)
# ============================================

set -e

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
EXTRACTED_DIR="${PROJECT_ROOT}/data/extracted"
FEATURES_DIR="${PROJECT_ROOT}/data/features"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOGS_DIR="${PROJECT_ROOT}/logs"
DATASET_ROOT="${PROJECT_ROOT}/data/IDLRW-DATASET"

PARTITION="mouth"  # Changed to mouth partition: inner+outer lips + jaw + cheeks
SEED=0
FPS=25

# Target feature levels - will run for each level
# CUMULATIVE FEATURE COUNTS (each file contains all previous levels):
#   B0: 2 features (x, y)
#   B1: 5 features (B0: 2 + B1: 3 = vx, vy, speed)
#   B2: 7 features (B0: 2 + B1: 3 + B2: 2 = distance, angle)
#   B3: 11 features (B0: 2 + B1: 3 + B2: 2 + B3: 4 = AU features)
# Best model used B2 (geometric features), so we'll try both B1 and B2
TARGET_FEATURE_LEVELS=("B1")  # Run for B1 and B2 feature levels (B2 was best for full partition)

# Models for mouth partition with B1 features:
# - All LSTM models: gin_lstm, gnn_lstm, graphsage_lstm, adaptive_gcn_lstm
# - LSTM+Mamba hybrid models: gin_lstm_mamba, gnn_lstm_mamba, graphsage_lstm_mamba, adaptive_gcn_lstm_mamba
# - Non-LSTM models: gin, gcn, graphsage, graphwavenet (can be added if needed)
MODELS=("gin_lstm_mamba" "gnn_lstm_mamba" "graphsage_lstm_mamba" "adaptive_gcn_lstm_mamba")

# Training hyperparameters
# BEST CONFIG: Optimal hyperparameters for best performance
BATCH_SIZE=32  # Best config: optimal batch size
HIDDEN_DIM=256  # Best config: optimal hidden dimension
NUM_EPOCHS=${EPOCHS:-100}  # Best config: 100 epochs (can be overridden by EPOCHS env var)
NUM_WORKERS=0  # Set to 0 to avoid memory duplication across workers
LEARNING_RATE=0.0001  # Best config: matches best model (gin_lstm_mamba used 0.0001)
WEIGHT_DECAY=0.0001   # Best config: matches best model (gin_lstm_mamba used 0.0001)
GRADIENT_CLIP=1.0  # Gradient clipping to prevent exploding gradients

# Speech mask scaling factor - test multiple values to find optimal
# Controls how much speech_mask influences attention (higher = more influence)
# Values to test: 0.5, 1.0, 2.0, 5.0, 10.0 (default: 5.0)
# Set via env var: SPEECH_MASK_SCALE=10.0 ./scripts/experiment.sh
SPEECH_MASK_SCALE=${SPEECH_MASK_SCALE:-10.0}  # Default: 5.0 (can override via env var)

# Speech mask context - number of adjacent frames to include around speech_mask=1
# If > 0, applies weight to +/- N frames around speech regions
# Helps account for co-articulation and imperfect mask accuracy
# Recommended: 1-3 frames (at 25 FPS: 1 frame = 40ms, 2 frames = 80ms, 3 frames = 120ms)
SPEECH_MASK_CONTEXT=${SPEECH_MASK_CONTEXT:-1}  # Default: 1 frame (can override via env var)

# Model-specific batch sizes (all models use batch size 8 for full partition)
declare -A MODEL_BATCH_SIZES
# All models use the default BATCH_SIZE=8

# Log file
mkdir -p "${LOGS_DIR}"
PIPELINE_LOG="${LOGS_DIR}/all_scenarios_full_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOGS_DIR}/all_scenarios_full.pid"

echo $$ > "${PID_FILE}"

# ===================
# FUNCTIONS
# ===================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${PIPELINE_LOG}"
}

run_extraction() {
    log "Running extraction for ${PARTITION} partition..."
    
    local extracted_train="${EXTRACTED_DIR}/${PARTITION}/${PARTITION}_train.pt"
    local extracted_val="${EXTRACTED_DIR}/${PARTITION}/${PARTITION}_val.pt"
    local extracted_test="${EXTRACTED_DIR}/${PARTITION}/${PARTITION}_test.pt"
    
    # Check if all splits already exist
    if [ -f "${extracted_train}" ] && \
       [ -f "${extracted_val}" ] && \
       [ -f "${extracted_test}" ]; then
        log "  ✓ Extraction already complete for all splits, skipping..."
        return 0
    fi
    
    log "  Extracting landmarks for ${PARTITION} partition..."
    
    # Extract each split that doesn't exist
    for split in train val test; do
        local extracted_file="${EXTRACTED_DIR}/${PARTITION}/${PARTITION}_${split}.pt"
        if [ -f "${extracted_file}" ]; then
            log "  ✓ Split ${split} already extracted, skipping..."
            continue
        fi
        
        log "  Extracting ${split} split..."
        python3 preprocessing/facemesh_extractor.py \
            --partition "${PARTITION}" \
            --split "${split}" \
            --dataset-root "${DATASET_ROOT}" \
            --out-dir "${EXTRACTED_DIR}" \
            --fps ${FPS} \
            >> "${LOGS_DIR}/extraction_${PARTITION}_${split}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            log "  ✓ Split ${split} extracted successfully"
        else
            log "  ✗ Failed to extract ${split} split"
            log "  Check logs: ${LOGS_DIR}/extraction_${PARTITION}_${split}.log"
            return 1
        fi
    done
    
    log "  ✓ Extraction complete for ${PARTITION} partition"
    return 0
}

check_requirements() {
    log "Checking requirements..."
    
    local extracted_file="${EXTRACTED_DIR}/${PARTITION}/${PARTITION}_train.pt"
    if [ ! -f "${extracted_file}" ]; then
        log "ERROR: ${PARTITION} partition extraction not found: ${extracted_file}"
        exit 1
    fi
    
    log "Requirements check passed."
}

check_process_running() {
    local pid=$1
    if ps -p "${pid}" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

ensure_prerequisites() {
    local target_level=$1
    
    log "Ensuring prerequisites for ${target_level}..."
    
    # Determine which levels are needed for computation (cumulative approach)
    # B1 needs B0 (to compute B0+B1), B2 needs B1 (to compute B0+B1+B2), etc.
    # Each level file is cumulative, but computation requires previous levels
    case "${target_level}" in
        "B0")
            levels_needed=("B0")
            ;;
        "B1")
            levels_needed=("B0" "B1")  # B1 file contains B0+B1, but needs B0 to compute
            ;;
        "B2")
            levels_needed=("B0" "B1" "B2")  # B2 file contains B0+B1+B2, but needs B1 to compute
            ;;
        "B3")
            levels_needed=("B0" "B1" "B2" "B3")  # B3 file contains B0+B1+B2+B3, but needs B2 to compute
            ;;
        *)
            log "ERROR: Unknown feature level: ${target_level}"
            return 1
            ;;
    esac
    
    log "  Prerequisites needed for computation: ${levels_needed[@]}"
    
    # Compute each level in order if it doesn't exist
    for level in "${levels_needed[@]}"; do
        local feature_dir="${FEATURES_DIR}/${level}"
        
        # Check if already computed
        if [ -f "${feature_dir}/${PARTITION}_train.pt" ] && \
           [ -f "${feature_dir}/${PARTITION}_val.pt" ] && \
           [ -f "${feature_dir}/${PARTITION}_test.pt" ]; then
            log "  ✓ Features ${level} already exist, skipping..."
        else
            log "  Computing features ${level} (prerequisite for ${target_level})..."
            compute_features "${level}"
            if [ $? -ne 0 ]; then
                log "ERROR: Failed to compute prerequisite ${level}"
                return 1
            fi
        fi
    done
    
    log "  ✓ All prerequisites for ${target_level} are ready"
    return 0
}

compute_features() {
    local feature_set=$1
    
    log "Computing features ${feature_set} (sequential processing, no multiprocessing)..."
    
    local feature_dir="${FEATURES_DIR}/${feature_set}"
    mkdir -p "${feature_dir}"
    
    # Lock file for feature computation
    local lock_file="${feature_dir}/.computing.lock"
    
    # Check if already computed (all splits must exist)
    if [ -f "${feature_dir}/${PARTITION}_train.pt" ] && \
       [ -f "${feature_dir}/${PARTITION}_val.pt" ] && \
       [ -f "${feature_dir}/${PARTITION}_test.pt" ]; then
        log "  Features ${feature_set} already exist for all splits, skipping..."
        return 0
    fi
    
    # Check which splits need to be computed
    splits_to_compute=()
    if [ ! -f "${feature_dir}/${PARTITION}_train.pt" ]; then
        splits_to_compute+=("train")
    fi
    if [ ! -f "${feature_dir}/${PARTITION}_val.pt" ]; then
        splits_to_compute+=("val")
    fi
    if [ ! -f "${feature_dir}/${PARTITION}_test.pt" ]; then
        splits_to_compute+=("test")
    fi
    
    if [ ${#splits_to_compute[@]} -eq 0 ]; then
        log "  All splits already exist, skipping..."
        return 0
    fi
    
    log "  Will compute splits: ${splits_to_compute[@]}"
    
    # Check if another process is computing (lock file exists and process is running)
    if [ -f "${lock_file}" ]; then
        local lock_pid=$(cat "${lock_file}" 2>/dev/null)
        if [ -n "${lock_pid}" ] && ps -p "${lock_pid}" > /dev/null 2>&1; then
            log "  Features ${feature_set} are being computed by another process (PID: ${lock_pid}), waiting..."
            # Wait for lock to be released (check every 10 seconds, max 5 minutes)
            local wait_count=0
            while [ -f "${lock_file}" ] && [ ${wait_count} -lt 30 ]; do
                sleep 10
                wait_count=$((wait_count + 1))
            done
            # Check again if features were computed
            if [ -f "${feature_dir}/${PARTITION}_train.pt" ] && \
               [ -f "${feature_dir}/${PARTITION}_val.pt" ] && \
               [ -f "${feature_dir}/${PARTITION}_test.pt" ]; then
                log "  Features ${feature_set} computed by other process, skipping..."
                return 0
            fi
        else
            # Stale lock file, remove it
            log "  Removing stale lock file..."
            rm -f "${lock_file}"
        fi
    fi
    
    # Create lock file
    echo $$ > "${lock_file}"
    
    python3 << EOF
import sys
import os
# Disable multiprocessing - force sequential processing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.append('${PROJECT_ROOT}')
from pathlib import Path
from preprocessing.feature_engineering import process_split_features

extracted_base = Path('${EXTRACTED_DIR}') / '${PARTITION}'
output_base = Path('${feature_dir}')

# Process splits sequentially (no multiprocessing)
# Start from val, then test, then train
splits_order = ['val', 'test', 'train']
for split in splits_order:
    extracted_path = extracted_base / f'${PARTITION}_{split}.pt'
    output_path = output_base / f'${PARTITION}_{split}.pt'
    
    if not extracted_path.exists():
        print(f"ERROR: Extracted file not found: {extracted_path}")
        sys.exit(1)
    
    # Skip if already computed
    if output_path.exists():
        print(f"Skipping {split} split (already exists)...")
        continue
    
    print(f"Processing {split} split sequentially...")
    # OPTIMIZATION: Pass previous level feature paths for faster computation
    # Each level now stores cumulative features (B1 has B0+B1, B2 has B0+B1+B2, etc.)
    b0_features_path = None
    b1_features_path = None
    b2_features_path = None
    
    if '${feature_set}' == 'B1':
        b0_features_path = output_base.parent / 'B0' / f'${PARTITION}_{split}.pt'
        if not b0_features_path.exists():
            b0_features_path = None
    elif '${feature_set}' == 'B2':
        b1_features_path = output_base.parent / 'B1' / f'${PARTITION}_{split}.pt'
        if not b1_features_path.exists():
            b1_features_path = None
            # Fallback to B0 if B1 not available
            b0_features_path = output_base.parent / 'B0' / f'${PARTITION}_{split}.pt'
            if not b0_features_path.exists():
                b0_features_path = None
    elif '${feature_set}' == 'B3':
        b2_features_path = output_base.parent / 'B2' / f'${PARTITION}_{split}.pt'
        if not b2_features_path.exists():
            b2_features_path = None
            # Fallback to B0 if B2 not available
            b0_features_path = output_base.parent / 'B0' / f'${PARTITION}_{split}.pt'
            if not b0_features_path.exists():
                b0_features_path = None
    
    process_split_features(
        str(extracted_path),
        '${feature_set}',
        str(output_path),
        b0_features_path=str(b0_features_path) if b0_features_path else None,
        b1_features_path=str(b1_features_path) if b1_features_path else None,
        b2_features_path=str(b2_features_path) if b2_features_path else None
    )
    print(f"✓ {split} complete")
EOF
    
    local compute_exit_code=$?
    
    # Always remove lock file
    rm -f "${lock_file}"
    
    if [ $compute_exit_code -eq 0 ]; then
        log "  ✓ Features ${feature_set} computed"
        return 0
    else
        log "  ✗ Features ${feature_set} failed"
        return 1
    fi
}

train_model() {
    local feature_set=$1
    local model_name=$2
    
    log "Training ${model_name} with ${feature_set}..."
    
    local result_dir="${RESULTS_DIR}/${PARTITION}/${feature_set}/${model_name}/seed_${SEED}"
    mkdir -p "${result_dir}"
    
    # Check if already trained
    if [ -f "${result_dir}/best.pth" ]; then
        log "  Model ${model_name}/${feature_set} already trained, skipping..."
        return 0
    fi
    
    # Get model-specific batch size
    local batch_size="${MODEL_BATCH_SIZES[$model_name]:-$BATCH_SIZE}"
    log "  Using batch size: ${batch_size}"
    
    # Create temporary config file
    local config_file="${result_dir}/config.yaml"
    
    # Generate model-specific parameters
    local model_params=""
    case "${model_name}" in
        "gcn"|"gat"|"graphsage"|"gin")
            model_params="    num_layers: 3
    dropout: 0.5
    temporal_pool: max"
            if [ "${model_name}" = "gat" ]; then
                model_params="${model_params}
    num_heads: 6"
            fi
            if [ "${model_name}" = "gin" ]; then
                model_params="${model_params}
    eps: 0.0
    train_eps: false"
            fi
            ;;
        "stgcn")
            model_params="    num_blocks: 3
    kernel_size: 9
    dropout: 0.5
    temporal_pool: max"
            ;;
        "gconvlstm")
            model_params="    num_layers: 1
    dropout: 0.5
    temporal_pool: last"
            ;;
        "gnn_lstm"|"gnn_gru")
            model_params="    num_gnn_layers: 3
    dropout: 0.5"
            if [ "${model_name}" = "gnn_lstm" ]; then
                # Use bidirectional LSTM for better temporal context
                # Increased to 3 LSTM layers for deeper temporal modeling
                model_params="${model_params}
    num_lstm_layers: 3
    dropout: 0.7
    bidirectional: true"
            else
                model_params="${model_params}
    num_gru_layers: 2
    bidirectional: false"
            fi
            ;;
        "gin_gru")
            model_params="    num_gin_layers: 3
    num_gru_layers: 2
    dropout: 0.5
    bidirectional: false
    eps: 0.0
    train_eps: false"
            ;;
        "gin_lstm")
            # Use bidirectional LSTM for better temporal context
            # Bidirectional processes frames in both directions (forward + backward)
            # This improves lip reading by capturing co-articulation patterns
            # Increased to 3 LSTM layers for deeper temporal modeling
            model_params="    num_gin_layers: 3
    num_lstm_layers: 3
    dropout: 0.7
    bidirectional: true
    eps: 0.0
    train_eps: false"
            ;;
        "graphsage_gru")
            model_params="    num_sage_layers: 3
    num_gru_layers: 2
    dropout: 0.5
    bidirectional: false
    aggregator: mean"
            ;;
        "graphsage_lstm")
            # Use bidirectional LSTM for better temporal context
            # Increased to 3 LSTM layers for deeper temporal modeling
            model_params="    num_sage_layers: 3
    num_lstm_layers: 3
    dropout: 0.7
    bidirectional: true
    aggregator: mean"
            ;;
        "gnn_temporal_conv")
            model_params="    num_gnn_layers: 3
    num_temporal_layers: 2
    temporal_kernel_size: 3
    dropout: 0.5
    temporal_pool: max"
            ;;
        "graphwavenet")
            model_params="    num_blocks: 3
    kernel_size: 3
    dropout: 0.5
    temporal_pool: max"
            ;;
        "adaptive_gcn_lstm")
            # Use bidirectional LSTM for better temporal context
            # Combines adaptive graph learning with bidirectional temporal modeling
            # Increased to 3 LSTM layers for deeper temporal modeling
            model_params="    num_gcn_layers: 3
    num_lstm_layers: 3
    dropout: 0.7
    bidirectional: true
    alpha: 0.5"
            ;;
        "gin_lstm_mamba"|"gnn_lstm_mamba"|"graphsage_lstm_mamba"|"adaptive_gcn_lstm_mamba")
            # LSTM+Mamba hybrid: Sequential architecture (LSTM → Mamba)
            # ALL MODELS NOW USE IDENTICAL CONFIG (matching GIN)
            # - Temporal attention + speech mask (learnable attention with prior guidance)
            # - Layer normalization + residual connections
            # - Same dropout, layers, and hyperparameters
            # - Only model-specific params differ (eps for GIN, aggregator for GraphSAGE, alpha for Adaptive GCN)
            if [ "${model_name}" = "gin_lstm_mamba" ]; then
                model_params="    num_gin_layers: 2
    num_lstm_layers: 1
    dropout: 0.5  # Best config: matches best model (gin_lstm_mamba used 0.5)
    bidirectional: true
    eps: 0.0
    train_eps: false
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE}  # Speech mask scaling factor (test: 0.5, 1.0, 2.0, 5.0, 10.0)
    speech_mask_context: ${SPEECH_MASK_CONTEXT}  # Adjacent frames to include (0=exact, 1-3=context)"
            elif [ "${model_name}" = "gnn_lstm_mamba" ]; then
                model_params="    num_gnn_layers: 2
    num_lstm_layers: 1
    dropout: 0.5  # Best config: matches best model (same as gin_lstm_mamba)
    bidirectional: true
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE}  # Speech mask scaling factor (test: 0.5, 1.0, 2.0, 5.0, 10.0)
    speech_mask_context: ${SPEECH_MASK_CONTEXT}  # Adjacent frames to include (0=exact, 1-3=context)"
            elif [ "${model_name}" = "graphsage_lstm_mamba" ]; then
                model_params="    num_sage_layers: 2
    num_lstm_layers: 1
    dropout: 0.5  # Best config: matches best model (same as gin_lstm_mamba)
    bidirectional: true
    aggregator: mean
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE}  # Speech mask scaling factor (test: 0.5, 1.0, 2.0, 5.0, 10.0)
    speech_mask_context: ${SPEECH_MASK_CONTEXT}  # Adjacent frames to include (0=exact, 1-3=context)"
            elif [ "${model_name}" = "adaptive_gcn_lstm_mamba" ]; then
                model_params="    num_gcn_layers: 2
    num_lstm_layers: 1
    dropout: 0.5  # Best config: matches best model (same as gin_lstm_mamba)
    bidirectional: true
    alpha: 0.5
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE}  # Speech mask scaling factor (test: 0.5, 1.0, 2.0, 5.0, 10.0)
    speech_mask_context: ${SPEECH_MASK_CONTEXT}  # Adjacent frames to include (0=exact, 1-3=context)"
            fi
            ;;
        *)
            # Default parameters
            model_params="    num_layers: 3
    dropout: 0.5
    temporal_pool: max"
            ;;
    esac
    
    cat > "${config_file}" << EOF
data:
  partition: ${PARTITION}
  feature_level: ${feature_set}
  feature_dir: ${FEATURES_DIR}

model:
  name: ${model_name}
  params:
    hidden_dim: ${HIDDEN_DIM}
${model_params}

training:
  output_dir: ${RESULTS_DIR}
  batch_size: ${batch_size}
  epochs: ${NUM_EPOCHS}
  learning_rate: ${LEARNING_RATE}
  weight_decay: ${WEIGHT_DECAY}
  optimizer: adam
  scheduler:
    name: reduceonplateau  # Best config: matches best model (gin_lstm_mamba used reduceonplateau)
    mode: min
    factor: 0.7
    patience: 5
    min_lr: 1e-6
  early_stopping_patience: 25  # Best config: matches best model (gin_lstm_mamba used 10)
  gradient_clip: ${GRADIENT_CLIP}
  label_smoothing: 0.0  # Best config: matches best model (gin_lstm_mamba used 0.0)
  num_workers: ${NUM_WORKERS}
EOF
    
    # Run training
    # Note: loss_history.png will be automatically generated after each epoch during training
    python3 train.py \
        --config "${config_file}" \
        --seed ${SEED} \
        >> "${result_dir}/train.log" 2>&1
    
    local train_exit_code=$?
    
    if [ $train_exit_code -eq 0 ]; then
        # Verify model was actually saved
        if [ -f "${result_dir}/best.pth" ]; then
            log "  ✓ Training ${model_name}/${feature_set} complete"
            
            # Generate final training visualizations (loss_history.png is already generated each epoch)
            if [ -f "${result_dir}/history.pt" ]; then
                log "  Generating final training visualizations..."
                python3 utils/generate_visualizations.py \
                    --result-dir "${result_dir}" \
                    >> "${result_dir}/train.log" 2>&1
                if [ $? -eq 0 ]; then
                    log "  ✓ Final visualizations generated"
                else
                    log "  ⚠ Visualization generation failed (non-critical, loss_history.png already generated each epoch)"
                fi
            fi
            
            return 0
        else
            log "  ✗ Training ${model_name}/${feature_set} completed but no checkpoint saved"
            return 1
        fi
    else
        log "  ✗ Training ${model_name}/${feature_set} failed (exit code: ${train_exit_code})"
        log "  Check logs: ${result_dir}/train.log"
        # Don't return 1 - continue with other models
        return 0
    fi
}

cleanup_features() {
    local feature_set=$1
    
    log "Cleaning up features ${feature_set}..."
    
    local feature_dir="${FEATURES_DIR}/${feature_set}"
    if [ -d "${feature_dir}" ]; then
        rm -rf "${feature_dir}"
        log "  ✓ Features ${feature_set} deleted"
    fi
}

# ===================
# MAIN PIPELINE
# ===================

log "="*80
log "ALL SCENARIOS PIPELINE - MOUTH PARTITION (BIGGER MODELS)"
log "="*80
log "Partition: ${PARTITION} (inner+outer lips + jaw + cheeks)"
log "Target feature levels: ${TARGET_FEATURE_LEVELS[@]}"
log "Models: ${MODELS[@]}"
log "Total scenarios: $((${#MODELS[@]} * ${#TARGET_FEATURE_LEVELS[@]}))"
log "Model capacity: HIDDEN_DIM=${HIDDEN_DIM} (doubled from best config), BATCH_SIZE=${BATCH_SIZE}, increased layers (3 GIN/GCN, 2 LSTM/GRU)"
log "Processing: Will preprocess prerequisites for each level if needed"
log "Training: Each B level file contains cumulative features (no concatenation needed)"
log "Cumulative feature counts: B0=2, B1=5 (B0+B1), B2=7 (B0+B1+B2), B3=11 (B0+B1+B2+B3)"
log "Memory optimization: Acceleration removed from B1, 1 anchor+no ratio in B2, PCA/motion removed from B3"
log "="*80

# Run extraction first if needed
log ""
log "="*80
log "RUNNING EXTRACTION FOR ${PARTITION} PARTITION"
log "="*80
run_extraction

if [ $? -ne 0 ]; then
    log "ERROR: Failed to run extraction for ${PARTITION} partition"
    exit 1
fi

check_requirements

# Process each feature level from B3 to B0
for TARGET_FEATURE_LEVEL in "${TARGET_FEATURE_LEVELS[@]}"; do
    log ""
    log "="*80
    log "PROCESSING FEATURE LEVEL: ${TARGET_FEATURE_LEVEL}"
    log "="*80
    
    # Ensure all prerequisites are computed before training
    log ""
    log "Ensuring prerequisites for ${TARGET_FEATURE_LEVEL}..."
    ensure_prerequisites "${TARGET_FEATURE_LEVEL}"
    
    if [ $? -ne 0 ]; then
        log "ERROR: Failed to compute prerequisites for ${TARGET_FEATURE_LEVEL}"
        log "Continuing with next feature level..."
        continue
    fi
    
    # Process each model with the target feature level
    # Each B level file contains cumulative features (B1 has B0+B1, B2 has B0+B1+B2, etc.)
    for model_name in "${MODELS[@]}"; do
        log ""
        log "="*80
        log "SCENARIO: ${model_name} / ${TARGET_FEATURE_LEVEL}"
        log "="*80
        
        # Check if already trained
        result_dir="${RESULTS_DIR}/${PARTITION}/${TARGET_FEATURE_LEVEL}/${model_name}/seed_${SEED}"
        if [ -f "${result_dir}/best.pth" ]; then
            log "  Scenario ${model_name}/${TARGET_FEATURE_LEVEL} already trained, skipping..."
            continue
        fi
        
        # Train model with target feature level (cumulative file contains all features up to that level)
        log "  Training ${model_name} with ${TARGET_FEATURE_LEVEL}..."
        log "  Note: ${TARGET_FEATURE_LEVEL} file contains cumulative features (all previous levels included)"
        
        # Calculate total features for this level (cumulative)
        case "${TARGET_FEATURE_LEVEL}" in
            "B0") total_features=2 ;;
            "B1") total_features=5 ;;  # B0(2) + B1(3)
            "B2") total_features=7 ;;  # B0(2) + B1(3) + B2(2)
            "B3") total_features=11 ;; # B0(2) + B1(3) + B2(2) + B3(4)
            *) total_features="unknown" ;;
        esac
        log "  Cumulative feature count: ${total_features} features per node (${TARGET_FEATURE_LEVEL} file is self-contained)"
        
        train_model "${TARGET_FEATURE_LEVEL}" "${model_name}"
        
        if [ $? -ne 0 ]; then
            log "  WARNING: Training failed for ${model_name}/${TARGET_FEATURE_LEVEL}, continuing..."
        fi
    done
    
    # Cleanup features for this level after training (optional - comment out if you want to keep features)
    # log ""
    # log "Cleaning up features for ${TARGET_FEATURE_LEVEL}..."
    # cleanup_features "${TARGET_FEATURE_LEVEL}"
done

# Final cleanup - remove all feature sets after all scenarios are complete
log ""
log "="*80
log "FINAL CLEANUP - REMOVING ALL FEATURE SETS"
log "="*80
for feature_set in "${TARGET_FEATURE_LEVELS[@]}"; do
    cleanup_features "${feature_set}"
done

# Aggregate results
log ""
log "="*80
log "AGGREGATING RESULTS"
log "="*80

python3 utils/aggregate_results_table.py \
    --results-dir "${RESULTS_DIR}" \
    --partition "${PARTITION}" \
    --output "${RESULTS_DIR}/${PARTITION}/results_table.pt" \
    --output-csv "${RESULTS_DIR}/${PARTITION}/results_table.csv" \
    >> "${PIPELINE_LOG}" 2>&1

if [ $? -eq 0 ]; then
    log "  ✓ Results table generated"
else
    log "  ⚠ Results table generation failed (non-critical)"
fi

# Generate Excel summary with all configs and descriptions
log ""
log "="*80
log "GENERATING EXCEL SUMMARY"
log "="*80

EXCEL_OUTPUT="${RESULTS_DIR}/training_summary.xlsx"
python3 utils/generate_excel_summary.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${EXCEL_OUTPUT}" \
    >> "${PIPELINE_LOG}" 2>&1

if [ $? -eq 0 ]; then
    log "  ✓ Excel summary generated: ${EXCEL_OUTPUT}"
    log "  Contains 9 sheets with all results, configs, and statistics"
else
    log "  ⚠ Excel summary generation failed (non-critical)"
fi

log ""
log "="*80
log "PIPELINE COMPLETE"
log "="*80
log "Results table (CSV): ${RESULTS_DIR}/${PARTITION}/results_table.csv"
log "Excel summary: ${EXCEL_OUTPUT}"
log "All logs: ${PIPELINE_LOG}"

rm -f "${PID_FILE}"

