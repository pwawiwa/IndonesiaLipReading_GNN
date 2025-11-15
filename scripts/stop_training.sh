#!/usr/bin/env bash
# Stop training and log server

set -euo pipefail

ROOT="/home/member2/tomoooo/IndonesiaLipReading_GNN"
cd "$ROOT"

LOG_DIR="outputs/combined_simplified"

echo "Stopping training and log server..."

# Stop log server
if [ -f "$LOG_DIR/log_server.pid" ]; then
    PID=$(cat "$LOG_DIR/log_server.pid")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping log server (PID: $PID)"
        kill "$PID" 2>/dev/null || true
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Force killing log server..."
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
    rm -f "$LOG_DIR/log_server.pid"
    echo "✓ Log server stopped"
else
    echo "No log server PID file found"
fi

# Stop training
if [ -f "$LOG_DIR/train.pid" ]; then
    PID=$(cat "$LOG_DIR/train.pid")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping training (PID: $PID)"
        kill "$PID" 2>/dev/null || true
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Force killing training..."
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
    rm -f "$LOG_DIR/train.pid"
    echo "✓ Training stopped"
else
    echo "No training PID file found"
fi

echo "Done!"



