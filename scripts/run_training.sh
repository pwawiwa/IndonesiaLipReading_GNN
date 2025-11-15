#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/member2/tomoooo/IndonesiaLipReading_GNN"
cd "$ROOT"

LOG_DIR="outputs/combined"
mkdir -p "$LOG_DIR"

STDOUT_LOG="$LOG_DIR/train_stdout.log"

echo "Starting training. Stdout -> $STDOUT_LOG"
nohup /usr/bin/python3 -u src/train.py >> "$STDOUT_LOG" 2>&1 &
echo $! > "$LOG_DIR/train.pid"
echo "Training PID: $(cat "$LOG_DIR/train.pid")"

