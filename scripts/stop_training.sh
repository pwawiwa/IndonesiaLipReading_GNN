#!/usr/bin/env bash
# Stop training and log server

set -euo pipefail

ROOT="/home/member2/tomoooo/IndonesiaLipReading_GNN"
cd "$ROOT"

# Check multiple output directories
LOG_DIRS=("outputs/v5" "outputs/v4" "outputs/v3" "outputs/v2" "outputs/v1" "outputs/combined")

echo "Stopping training and log server..."

# Also check for screen sessions
SCREEN_NAME="lipreading_v5"
if screen -list 2>/dev/null | grep -q "$SCREEN_NAME"; then
    echo "Stopping screen session: $SCREEN_NAME"
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
    echo "✓ Screen session stopped"
fi

# Check all output directories
for LOG_DIR in "${LOG_DIRS[@]}"; do
    if [ ! -d "$LOG_DIR" ]; then
        continue
    fi
    
    # Stop log server
    if [ -f "$LOG_DIR/log_server.pid" ]; then
        PID=$(cat "$LOG_DIR/log_server.pid")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping log server in $LOG_DIR (PID: $PID)"
            kill "$PID" 2>/dev/null || true
            sleep 1
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        rm -f "$LOG_DIR/log_server.pid"
        echo "✓ Log server stopped in $LOG_DIR"
    fi
    
    # Stop training
    if [ -f "$LOG_DIR/train.pid" ]; then
        PID=$(cat "$LOG_DIR/train.pid")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping training in $LOG_DIR (PID: $PID)"
            kill "$PID" 2>/dev/null || true
            sleep 1
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Force killing training..."
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        rm -f "$LOG_DIR/train.pid"
        echo "✓ Training stopped in $LOG_DIR"
    fi
done

echo "Done!"



