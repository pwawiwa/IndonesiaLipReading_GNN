#!/bin/bash
# ============================================
# Run extraction for partitions and splits
# Usage: bash extraction.sh [partition]
#   If partition is specified, only extract that partition
#   If not specified, extract all partitions (lips, mouth, full)
# ============================================

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
DATASET_ROOT="${PROJECT_ROOT}/data/IDLRW-DATASET"
OUT_DIR="${PROJECT_ROOT}/data/extracted"
LOG_DIR="${PROJECT_ROOT}/logs"

# Check if partition argument is provided
if [ -n "$1" ]; then
    # Extract only specified partition
    PARTITIONS=("$1")
    PID_FILE="${LOG_DIR}/extraction_${1}.pid"
    echo "Extracting only partition: $1"
else
    # Extract all partitions
    PARTITIONS=("lips" "mouth" "full")
    PID_FILE="${LOG_DIR}/extraction_all.pid"
    echo "Extracting all partitions: ${PARTITIONS[*]}"
fi

SPLITS=("train" "val" "test")
FPS=25

# Create log directory
mkdir -p "${LOG_DIR}"

# Check if already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        echo "Extraction already running with PID: ${OLD_PID}"
        echo "To stop: kill ${OLD_PID} or bash scripts/kill_extraction.sh"
        exit 1
    else
        echo "Removing stale PID file"
        rm -f "${PID_FILE}"
    fi
fi

echo "Starting extraction..."
echo "Logs will be saved to: ${LOG_DIR}/extraction_*.log"
echo "PID will be saved to: ${PID_FILE}"

# Save PID
echo $$ > "${PID_FILE}"

# Run extraction for all partitions and splits
for partition in "${PARTITIONS[@]}"; do
    for split in "${SPLITS[@]}"; do
        extracted_file="${OUT_DIR}/${partition}/${partition}_${split}.pt"
        if [ -f "${extracted_file}" ]; then
            echo "✓ ${partition}/${split} already extracted, skipping..."
            continue
        fi
        
        echo "Extracting ${partition}/${split}..."
        python3 "${PROJECT_ROOT}/preprocessing/facemesh_extractor.py" \
            --partition "${partition}" \
            --split "${split}" \
            --dataset-root "${DATASET_ROOT}" \
            --out-dir "${OUT_DIR}" \
            --fps ${FPS} \
            --max-workers 1 \
            >> "${LOG_DIR}/extraction_${partition}_${split}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ ${partition}/${split} extracted successfully"
        else
            echo "✗ Failed to extract ${partition}/${split}"
            echo "  Check logs: ${LOG_DIR}/extraction_${partition}_${split}.log"
        fi
    done
done

echo ""
echo "✓ Extraction complete!"
rm -f "${PID_FILE}"
