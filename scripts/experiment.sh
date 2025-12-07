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
# CUMULATIVE FEATURE COUNTS (each file contains all previous levels, 3D coordinates):
#   B0: 3 features (X, Y, Z - 3D normalized coordinates)
#   B1: 10 features (B0: 3 + B1: 7 = vx, vy, vz, speed, ax, ay, az - all 3D)
#   B2: 18 features (B0: 3 + B1: 7 + B2: 8 = MAR, lip_width, lip_height, jaw_height, cheek_puff, lip_curvature, lip_corner_angle, jaw_opening) - computed directly from extracted data using 3D Euclidean distances
#   B3: 22 features (B0: 3 + B1: 7 + B2: 8 + B3: 4 = AU25, AU26, AU12, AU27 - using 3D displacement magnitudes)
# Best model used B2 (geometric features), so we'll try both B1 and B2
TARGET_FEATURE_LEVELS=("B1" "B0")  # Run for B1 and B2 feature levels (B2 was best for full partition)

# Models for mouth partition with B1 features:
# - All LSTM models: gin_lstm, gnn_lstm, graphsage_lstm, adaptive_gcn_lstm
# - LSTM+Mamba hybrid models: gin_lstm_mamba, gnn_lstm_mamba, graphsage_lstm_mamba, adaptive_gcn_lstm_mamba
# - Non-LSTM models: gin, gcn, graphsage, graphwavenet (can be added if needed)
MODELS=("gin_lstm_mamba" "gnn_lstm_mamba" "graphsage_lstm_mamba" "adaptive_gcn_lstm_mamba")

# Training hyperparameters
# BEST CONFIG: Optimal hyperparameters for best performance
BATCH_SIZE=${BATCH_SIZE:-32}  # Matches best_config_34,2.yaml (default: 32)
HIDDEN_DIM=256  # Best config: optimal hidden dimension
NUM_EPOCHS=${EPOCHS:-100}  # 150 epochs (no early stopping) - matches best_config_34,2.yaml
NUM_WORKERS=0  # Set to 0 to avoid memory duplication across workers
LEARNING_RATE=${LEARNING_RATE:-0.0001}  # Matches best_config_34,2.yaml (was: 0.00005)
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}   # Matches best_config_34,2.yaml (fixed, no adaptive scheduler)
GRADIENT_CLIP=1.0  # Gradient clipping to prevent exploding gradients

# Weight Decay Strategy - DISABLED by default (use fixed weight_decay from best_config_34,2.yaml)
# Set WEIGHT_DECAY_SCHEDULER_TYPE to enable adaptive weight decay if needed
# Examples:
#   ./scripts/experiment.sh  # Uses fixed weight_decay=0.0001 (matches best_config_34,2.yaml)
#   WEIGHT_DECAY_SCHEDULER_TYPE=adaptive_gap ./scripts/experiment.sh  # Enable adaptive scheduler
WEIGHT_DECAY_SCHEDULER_TYPE=${WEIGHT_DECAY_SCHEDULER_TYPE:-}  # Default: disabled (empty = fixed weight_decay), options: adaptive_gap, linear_warmup, cosine, step, plateau, exponential
WEIGHT_DECAY_SCHEDULER_MIN_WD=${WEIGHT_DECAY_SCHEDULER_MIN_WD:-0.0005}  # For adaptive_gap: minimum WD (increased from 1e-6 to 0.0005 to start with stronger regularization)
WEIGHT_DECAY_SCHEDULER_MAX_WD=${WEIGHT_DECAY_SCHEDULER_MAX_WD:-0.01}  # For adaptive_gap: maximum WD
WEIGHT_DECAY_SCHEDULER_GAP_THRESHOLD=${WEIGHT_DECAY_SCHEDULER_GAP_THRESHOLD:-0.02}  # For adaptive_gap: gap threshold (reduced from 0.05 to 0.02 = 2% accuracy gap to detect overfitting earlier)
WEIGHT_DECAY_SCHEDULER_FAST_DROP_THRESHOLD=${WEIGHT_DECAY_SCHEDULER_FAST_DROP_THRESHOLD:-0.15}  # For adaptive_gap: fast drop threshold (increased from 0.1 to 0.15 = 15% per epoch to be less sensitive)
WEIGHT_DECAY_SCHEDULER_INCREASE_FACTOR=${WEIGHT_DECAY_SCHEDULER_INCREASE_FACTOR:-1.5}  # For adaptive_gap: multiply by this when gap increases (increased from 1.2 to 1.5 for more aggressive regularization)
WEIGHT_DECAY_SCHEDULER_DECREASE_FACTOR=${WEIGHT_DECAY_SCHEDULER_DECREASE_FACTOR:-0.95}  # For adaptive_gap: multiply by this when training drops too fast (reduced from 0.9 to 0.95 to decrease less aggressively)
WEIGHT_DECAY_SCHEDULER_LOOKBACK_WINDOW=${WEIGHT_DECAY_SCHEDULER_LOOKBACK_WINDOW:-3}  # For adaptive_gap: epochs to look back (reduced from 5 to 3 for faster response)
# Legacy linear_warmup parameters (used if type=linear_warmup)
WEIGHT_DECAY_SCHEDULER_START_WD=${WEIGHT_DECAY_SCHEDULER_START_WD:-0.00001}  # For linear_warmup: start at 1e-5
WEIGHT_DECAY_SCHEDULER_TARGET_WD=${WEIGHT_DECAY_SCHEDULER_TARGET_WD:-0.0001}  # For linear_warmup: target 1e-4
WEIGHT_DECAY_SCHEDULER_WARMUP_EPOCHS=${WEIGHT_DECAY_SCHEDULER_WARMUP_EPOCHS:-10}  # For linear_warmup: warmup epochs
WEIGHT_DECAY_SCHEDULER_T_MAX=${WEIGHT_DECAY_SCHEDULER_T_MAX:-${NUM_EPOCHS}}  # For cosine (defaults to NUM_EPOCHS)
WEIGHT_DECAY_SCHEDULER_MIN_WD=${WEIGHT_DECAY_SCHEDULER_MIN_WD:-0.0}  # For cosine, step, exponential
WEIGHT_DECAY_SCHEDULER_STEP_SIZE=${WEIGHT_DECAY_SCHEDULER_STEP_SIZE:-10}  # For step
WEIGHT_DECAY_SCHEDULER_GAMMA=${WEIGHT_DECAY_SCHEDULER_GAMMA:-0.1}  # For step, exponential
WEIGHT_DECAY_SCHEDULER_PATIENCE=${WEIGHT_DECAY_SCHEDULER_PATIENCE:-5}  # For plateau
WEIGHT_DECAY_SCHEDULER_FACTOR=${WEIGHT_DECAY_SCHEDULER_FACTOR:-5}  # For plateau

# Optimizer selection - matches best_config_34,2.yaml
OPTIMIZER=${OPTIMIZER:-adam}  # Options: adam, adamw, sgd (default: adam - matches best_config_34,2.yaml)

# Class weights for handling imbalanced classes - DISABLED by default (causes overfitting)
# Class weights were found to cause severe overfitting (train: 60% vs val: 15.6%)
# Use label smoothing instead for better generalization
USE_CLASS_WEIGHTS=${USE_CLASS_WEIGHTS:-false}  # Disabled by default (was causing overfitting)
CLASS_WEIGHT_METHOD=${CLASS_WEIGHT_METHOD:-moderate}  # moderate, balanced, sqrt, inverse, log (not used if disabled)

# Label smoothing - matches best_config_34,2.yaml
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}  # Best config: 0.0 (no label smoothing)

# Dropout - matches best_config_34,2.yaml
DROPOUT=${DROPOUT:-0.5}  # Best config: 0.5 (matches best_config_34,2.yaml)

# Early stopping - matches best_config_34,2.yaml (disabled)
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-999999}  # Best config: disabled (999999 = no early stopping)

# Speech mask scaling factor - matches best_config_34,2.yaml
# Controls how much speech_mask influences attention (higher = more influence)
# Best config: 10.0
SPEECH_MASK_SCALE=${SPEECH_MASK_SCALE:-10.0}  # Best config: 10.0 (matches best_config_34,2.yaml)

# Speech mask context - matches best_config_34,2.yaml
# Number of adjacent frames to include around speech_mask=1
# Best config: 1 frame
SPEECH_MASK_CONTEXT=${SPEECH_MASK_CONTEXT:-1}  # Best config: 1 (matches best_config_34,2.yaml)

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
            levels_needed=("B2")  # B2 file contains B0+B1+B2, computed directly from extracted data (no B0/B1 files needed)
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
# Allow threading for parallel feature computation (ThreadPoolExecutor uses threads)
# Keep BLAS single-threaded to avoid conflicts with threading
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
    # B2 can be computed directly from extracted data without needing B0/B1 files
    b0_features_path = None
    b1_features_path = None
    b2_features_path = None
    
    if '${feature_set}' == 'B1':
        # B1 can optionally use B0 for speedup, but can compute directly from landmarks
        b0_features_path = output_base.parent / 'B0' / f'${PARTITION}_{split}.pt'
        if not b0_features_path.exists():
            b0_features_path = None
    elif '${feature_set}' == 'B2':
        # B2 computes B0+B1+B2 directly from extracted data (landmarks) in one pass
        # No need to load B0/B1 files - everything computed on-the-fly
        # Optional: can use B1/B0 files for speedup if available, but not required
        b1_features_path = None  # Don't require B1 file - compute directly
        b0_features_path = None  # Don't require B0 file - compute directly
    elif '${feature_set}' == 'B3':
        # B3 can optionally use B2 for speedup, but can compute directly from landmarks
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
        
        # Generate feature samples for verification
        log "  Generating feature samples for ${feature_set}..."
        generate_feature_samples "${feature_set}"
        
        return 0
    else
        log "  ✗ Features ${feature_set} failed"
        return 1
    fi
}

generate_feature_samples() {
    local feature_set=$1
    
    # Only generate samples for train split (to save time)
    local extracted_file="${EXTRACTED_DIR}/${PARTITION}/${PARTITION}_train.pt"
    local samples_dir="${FEATURES_DIR}/B0_B1_B2_B3_samples/${PARTITION}"
    
    # Check if samples already exist
    if [ -d "${samples_dir}" ] && [ "$(ls -A ${samples_dir} 2>/dev/null)" ]; then
        log "    Feature samples already exist, skipping..."
        return 0
    fi
    
    # Generate samples using the Python script
    log "    Running feature sample generation..."
    python3 scripts/generate_feature_samples.py \
        --extracted-file "${extracted_file}" \
        --partition "${PARTITION}" \
        --output-dir "${FEATURES_DIR}" \
        --num-samples 3 \
        >> "${LOGS_DIR}/feature_samples_${feature_set}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        log "    ✓ Feature samples generated"
    else
        log "    ⚠ Feature sample generation failed (non-critical)"
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
    
    # Use best_config_34_val_acc.yaml as base template
    local best_config="${RESULTS_DIR}/configs/best_config_34_val_acc.yaml"
    if [ ! -f "${best_config}" ]; then
        log "  ⚠ Warning: best_config_34_val_acc.yaml not found, using default config generation"
        best_config=""
    else
        log "  Using best_config_34_val_acc.yaml as base template"
    fi
    
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
    dropout: ${DROPOUT:-0.5}  # Best config: 0.5 (matches best_config_34,2.yaml)
    bidirectional: true
    eps: 0.0
    train_eps: false
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE:-10.0}  # Best config: 10.0 (matches best_config_34,2.yaml)
    speech_mask_context: ${SPEECH_MASK_CONTEXT:-1}  # Best config: 1 (matches best_config_34,2.yaml)"
            elif [ "${model_name}" = "gnn_lstm_mamba" ]; then
                model_params="    num_gnn_layers: 2
    num_lstm_layers: 1
    dropout: ${DROPOUT:-0.5}  # Best config: 0.5 (matches best_config_34,2.yaml)
    bidirectional: true
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE:-10.0}  # Best config: 10.0 (matches best_config_34,2.yaml)
    speech_mask_context: ${SPEECH_MASK_CONTEXT:-1}  # Best config: 1 (matches best_config_34,2.yaml)"
            elif [ "${model_name}" = "graphsage_lstm_mamba" ]; then
                model_params="    num_sage_layers: 2
    num_lstm_layers: 1
    dropout: ${DROPOUT:-0.5}  # Best config: 0.5 (matches best_config_34,2.yaml)
    bidirectional: true
    aggregator: mean
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE:-10.0}  # Best config: 10.0 (matches best_config_34,2.yaml)
    speech_mask_context: ${SPEECH_MASK_CONTEXT:-1}  # Best config: 1 (matches best_config_34,2.yaml)"
            elif [ "${model_name}" = "adaptive_gcn_lstm_mamba" ]; then
                model_params="    num_gcn_layers: 2
    num_lstm_layers: 1
    dropout: ${DROPOUT:-0.5}  # Best config: 0.5 (matches best_config_34,2.yaml)
    bidirectional: true
    alpha: 0.5
    mamba_d_state: 16
    mamba_d_conv: 4
    mamba_expand: 2
    speech_mask_scale: ${SPEECH_MASK_SCALE:-10.0}  # Best config: 10.0 (matches best_config_34,2.yaml)
    speech_mask_context: ${SPEECH_MASK_CONTEXT:-1}  # Best config: 1 (matches best_config_34,2.yaml)"
            fi
            ;;
        *)
            # Default parameters
            model_params="    num_layers: 3
    dropout: 0.5
    temporal_pool: max"
            ;;
    esac
    
    # Build weight decay scheduler config (if enabled)
    weight_decay_scheduler_config=""
    if [ -n "${WEIGHT_DECAY_SCHEDULER_TYPE}" ]; then
        weight_decay_scheduler_config="  # Adaptive Weight Decay Scheduler
  weight_decay_scheduler:
    type: ${WEIGHT_DECAY_SCHEDULER_TYPE}"
        case "${WEIGHT_DECAY_SCHEDULER_TYPE}" in
            linear_warmup)
                weight_decay_scheduler_config="${weight_decay_scheduler_config}
    start_wd: ${WEIGHT_DECAY_SCHEDULER_START_WD}
    target_wd: ${WEIGHT_DECAY_SCHEDULER_TARGET_WD}
    warmup_epochs: ${WEIGHT_DECAY_SCHEDULER_WARMUP_EPOCHS}"
                ;;
            cosine)
                weight_decay_scheduler_config="${weight_decay_scheduler_config}
    T_max: ${WEIGHT_DECAY_SCHEDULER_T_MAX}
    min_wd: ${WEIGHT_DECAY_SCHEDULER_MIN_WD}"
                ;;
            step)
                weight_decay_scheduler_config="${weight_decay_scheduler_config}
    step_size: ${WEIGHT_DECAY_SCHEDULER_STEP_SIZE}
    gamma: ${WEIGHT_DECAY_SCHEDULER_GAMMA}
    min_wd: ${WEIGHT_DECAY_SCHEDULER_MIN_WD}"
                ;;
            plateau)
                weight_decay_scheduler_config="${weight_decay_scheduler_config}
    mode: min
    factor: ${WEIGHT_DECAY_SCHEDULER_FACTOR}
    patience: ${WEIGHT_DECAY_SCHEDULER_PATIENCE}
    min_wd: ${WEIGHT_DECAY_SCHEDULER_MIN_WD}"
                ;;
            exponential)
                weight_decay_scheduler_config="${weight_decay_scheduler_config}
    gamma: ${WEIGHT_DECAY_SCHEDULER_GAMMA}
    min_wd: ${WEIGHT_DECAY_SCHEDULER_MIN_WD}"
                ;;
            adaptive_gap|adaptive)
                weight_decay_scheduler_config="${weight_decay_scheduler_config}
    min_wd: ${WEIGHT_DECAY_SCHEDULER_MIN_WD}
    max_wd: ${WEIGHT_DECAY_SCHEDULER_MAX_WD}
    gap_threshold: ${WEIGHT_DECAY_SCHEDULER_GAP_THRESHOLD}
    fast_drop_threshold: ${WEIGHT_DECAY_SCHEDULER_FAST_DROP_THRESHOLD}
    increase_factor: ${WEIGHT_DECAY_SCHEDULER_INCREASE_FACTOR}
    decrease_factor: ${WEIGHT_DECAY_SCHEDULER_DECREASE_FACTOR}
    lookback_window: ${WEIGHT_DECAY_SCHEDULER_LOOKBACK_WINDOW}"
                ;;
        esac
    fi
    
    # Expand bash variables with defaults
    local early_stopping_patience="${EARLY_STOPPING_PATIENCE:-999999}"
    local label_smoothing="${LABEL_SMOOTHING:-0.0}"
    local use_class_weights="${USE_CLASS_WEIGHTS:-false}"
    local class_weight_method="${CLASS_WEIGHT_METHOD:-moderate}"
    
    # Generate config file using best_config_34_val_acc.yaml as base
    if [ -n "${best_config}" ]; then
        # Use Python to load best config and update necessary fields
        python3 << PYTHON_SCRIPT
import yaml
import sys
import os
from pathlib import Path
import io

# Helper function to convert string booleans to Python booleans
def str_to_bool(s):
    if isinstance(s, bool):
        return s
    return s.lower() in ('true', '1', 'yes', 'on')

# Get variables from environment (set by bash)
best_config_path = "${best_config}"
config_file_path = "${config_file}"
partition = "${PARTITION}"
feature_set = "${feature_set}"
features_dir = "${FEATURES_DIR}"
model_name = "${model_name}"
hidden_dim = ${HIDDEN_DIM}
batch_size = ${batch_size}
num_epochs = ${NUM_EPOCHS}
learning_rate = ${LEARNING_RATE}
weight_decay = ${WEIGHT_DECAY}
optimizer = "${OPTIMIZER}"
early_stopping_patience = ${early_stopping_patience}
gradient_clip = ${GRADIENT_CLIP}
label_smoothing = ${label_smoothing}
num_workers = ${NUM_WORKERS}
use_class_weights = str_to_bool("${use_class_weights}")
class_weight_method = "${class_weight_method}"
model_params_yaml = """${model_params}"""
weight_decay_scheduler_yaml = """${weight_decay_scheduler_config}"""

# Load best config
with open(best_config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update data section
config['data']['partition'] = partition
config['data']['feature_level'] = feature_set
config['data']['feature_dir'] = features_dir

# Update model section
config['model']['name'] = model_name
config['model']['params']['hidden_dim'] = hidden_dim

# Parse and update model-specific params
if model_params_yaml.strip():
    model_params_dict = yaml.safe_load(io.StringIO(model_params_yaml))
    if model_params_dict:
        config['model']['params'].update(model_params_dict)

# Update training section
config['training']['output_dir'] = "${RESULTS_DIR}"
config['training']['batch_size'] = batch_size
config['training']['epochs'] = num_epochs
config['training']['learning_rate'] = learning_rate
config['training']['weight_decay'] = weight_decay
config['training']['optimizer'] = optimizer
if 'scheduler' not in config['training']:
    config['training']['scheduler'] = {}
config['training']['scheduler']['name'] = "reduceonplateau"
config['training']['scheduler']['mode'] = "min"
config['training']['scheduler']['factor'] = 0.7
config['training']['scheduler']['patience'] = 25
config['training']['scheduler']['min_lr'] = 1e-6
config['training']['early_stopping_patience'] = early_stopping_patience
config['training']['gradient_clip'] = gradient_clip
config['training']['label_smoothing'] = label_smoothing
config['training']['num_workers'] = num_workers
config['training']['balance_classes'] = False
config['training']['balance_factor'] = 1.0
config['training']['use_class_weights'] = use_class_weights
config['training']['class_weight_method'] = class_weight_method

# Update augmentation (disabled)
if 'augmentation' not in config:
    config['augmentation'] = {}
config['augmentation']['enabled'] = False

# Add weight decay scheduler if configured
if weight_decay_scheduler_yaml.strip():
    wd_scheduler_dict = yaml.safe_load(io.StringIO(weight_decay_scheduler_yaml))
    if wd_scheduler_dict:
        config['training'].update(wd_scheduler_dict)

# Save config
with open(config_file_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
PYTHON_SCRIPT
    else
        # Fallback: generate config from scratch (original method)
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
  optimizer: ${OPTIMIZER}
  scheduler:
    name: reduceonplateau
    mode: min
    factor: 0.7
    patience: 25
    min_lr: 1e-6
${weight_decay_scheduler_config}
  early_stopping_patience: ${EARLY_STOPPING_PATIENCE:-999999}
  gradient_clip: ${GRADIENT_CLIP}
  label_smoothing: ${LABEL_SMOOTHING:-0.0}
  num_workers: ${NUM_WORKERS}
  balance_classes: false
  balance_factor: 1.0
  use_class_weights: ${USE_CLASS_WEIGHTS:-false}
  class_weight_method: ${CLASS_WEIGHT_METHOD:-moderate}

augmentation:
  enabled: false
EOF
    fi
    
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
            
            # Generate classification report and confusion matrix
            log "  Generating classification report..."
            python3 evaluate_best_model.py \
                --result-dir "${result_dir}" \
                --device cuda \
                >> "${result_dir}/train.log" 2>&1
            if [ $? -eq 0 ]; then
                log "  ✓ Classification report generated"
            else
                log "  ⚠ Classification report generation failed (non-critical)"
            fi
            
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
log "Cumulative feature counts (3D): B0=3, B1=10 (B0+B1), B2=18 (B0+B1+B2), B3=22 (B0+B1+B2+B3)"
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
    
    # Generate feature samples after prerequisites are ready (if not already generated)
    log ""
    log "Generating feature samples for verification..."
    generate_feature_samples "${TARGET_FEATURE_LEVEL}"
    
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
            log "  Scenario ${model_name}/${TARGET_FEATURE_LEVEL} already trained"
            
            # Check if classification report exists, if not generate it
            if [ ! -f "${result_dir}/classification_report.txt" ] || [ ! -f "${result_dir}/confusion_matrix.png" ]; then
                log "  Generating classification report for existing model..."
                python3 evaluate_best_model.py \
                    --result-dir "${result_dir}" \
                    --device cuda \
                    >> "${result_dir}/train.log" 2>&1
                if [ $? -eq 0 ]; then
                    log "  ✓ Classification report generated"
                else
                    log "  ⚠ Classification report generation failed (non-critical)"
                fi
            else
                log "  ✓ Classification report already exists, skipping..."
            fi
            continue
        fi
        
        # Train model with target feature level (cumulative file contains all features up to that level)
        log "  Training ${model_name} with ${TARGET_FEATURE_LEVEL}..."
        log "  Note: ${TARGET_FEATURE_LEVEL} file contains cumulative features (all previous levels included)"
        
        # Calculate total features for this level (cumulative, 3D coordinates)
        case "${TARGET_FEATURE_LEVEL}" in
            "B0") total_features=3 ;;  # X, Y, Z - 3D normalized coordinates
            "B1") total_features=10 ;;  # B0(3) + B1(7) = vx, vy, vz, speed, ax, ay, az - all 3D
            "B2") total_features=18 ;;  # B0(3) + B1(7) + B2(8) - computed directly from extracted data using 3D Euclidean distances
            "B3") total_features=22 ;; # B0(3) + B1(7) + B2(8) + B3(4) - AU features using 3D displacement magnitudes
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

