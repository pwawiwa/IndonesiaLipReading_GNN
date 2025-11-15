#!/usr/bin/env bash
# Combined script to start log server and training in background
# Uses screen to ensure processes continue after SSH disconnection

set -euo pipefail

ROOT="/home/member2/tomoooo/IndonesiaLipReading_GNN"
cd "$ROOT"

LOG_DIR="outputs/combined_simplified"
mkdir -p "$LOG_DIR"

# Log server configuration
LOG_FILE="$LOG_DIR/training.log"
LOG_SERVER_HOST="0.0.0.0"
LOG_SERVER_PORT="8080"

# Training stdout log
STDOUT_LOG="$LOG_DIR/train_stdout.log"
STDERR_LOG="$LOG_DIR/train_stderr.log"

SCREEN_NAME="lipreading_training"

echo "=========================================="
echo "Starting Training with Log Server"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo "Training log: $LOG_FILE"
echo "Stdout log: $STDOUT_LOG"
echo "Stderr log: $STDERR_LOG"
echo "Log server: http://$LOG_SERVER_HOST:$LOG_SERVER_PORT/"
echo "Screen session: $SCREEN_NAME"
echo "=========================================="

# Check if screen is available
if ! command -v screen &> /dev/null; then
    echo "Warning: screen not found, using nohup instead"
    USE_SCREEN=false
else
    USE_SCREEN=true
fi

# Kill existing processes if they exist
if [ -f "$LOG_DIR/log_server.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/log_server.pid")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Stopping existing log server (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_DIR/log_server.pid"
fi

if [ -f "$LOG_DIR/train.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/train.pid")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Stopping existing training (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_DIR/train.pid"
fi

# Kill existing screen session if exists
if [ "$USE_SCREEN" = true ]; then
    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "Stopping existing screen session: $SCREEN_NAME"
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
        sleep 1
    fi
fi

if [ "$USE_SCREEN" = true ]; then
    # Use screen for persistent session
    echo ""
    echo "Starting in screen session: $SCREEN_NAME"
    echo "To attach: screen -r $SCREEN_NAME"
    echo "To detach: Ctrl+A then D"
    
    # Create a startup script that runs both processes
    STARTUP_SCRIPT="$LOG_DIR/startup.sh"
    cat > "$STARTUP_SCRIPT" << EOF
#!/bin/bash
cd "$ROOT"

# Start log server
echo "Starting log server..."
python3 -u -c "
from src.utils.log_dashboard import run_log_server_blocking
from pathlib import Path
run_log_server_blocking(
    Path('$LOG_FILE'),
    host='$LOG_SERVER_HOST',
    port=$LOG_SERVER_PORT,
    entries=10,
    refresh_minutes=5
)
" > "$LOG_DIR/log_server.out" 2>&1 &
LOG_SERVER_PID=\$!
echo \$LOG_SERVER_PID > "$LOG_DIR/log_server.pid"
echo "Log server PID: \$LOG_SERVER_PID"

# Wait a moment
sleep 2

# Start training
echo "Starting training..."
python3 -u src/train.py > "$STDOUT_LOG" 2> "$STDERR_LOG"
EOF
    chmod +x "$STARTUP_SCRIPT"
    
    # Start screen session with the startup script
    screen -dmS "$SCREEN_NAME" bash -c "$STARTUP_SCRIPT; exec bash"
    
    echo "✓ Screen session started: $SCREEN_NAME"
    echo ""
    echo "To attach to session:"
    echo "  screen -r $SCREEN_NAME"
    echo ""
    echo "To view logs without attaching:"
    echo "  tail -f $STDOUT_LOG"
    echo "  tail -f $LOG_FILE"
    
else
    # Fallback to nohup
    echo ""
    echo "Starting log server with nohup..."
    nohup python3 -u -c "
from src.utils.log_dashboard import run_log_server_blocking
from pathlib import Path
run_log_server_blocking(
    Path('$LOG_FILE'),
    host='$LOG_SERVER_HOST',
    port=$LOG_SERVER_PORT,
    entries=10,
    refresh_minutes=5
)
" > "$LOG_DIR/log_server.out" 2>&1 &
LOG_SERVER_PID=$!
echo $LOG_SERVER_PID > "$LOG_DIR/log_server.pid"
echo "✓ Log server started (PID: $LOG_SERVER_PID)"
echo "  Access at: http://$LOG_SERVER_HOST:$LOG_SERVER_PORT/"

# Wait a moment for log server to start
sleep 2

# Start training in background (detached)
echo ""
echo "Starting training with nohup..."
nohup python3 -u src/train.py \
  > "$STDOUT_LOG" 2> "$STDERR_LOG" &
TRAIN_PID=$!
echo $TRAIN_PID > "$LOG_DIR/train.pid"
echo "✓ Training started (PID: $TRAIN_PID)"
fi

echo ""
echo "=========================================="
echo "Processes started!"
echo "=========================================="
echo "To check status:"
echo "  ps aux | grep -E '(log_dashboard|train.py)'"
if [ "$USE_SCREEN" = true ]; then
    echo "  screen -list"
fi
echo ""
echo "To view logs:"
echo "  tail -f $STDOUT_LOG"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop:"
echo "  bash scripts/stop_training.sh"
echo ""
echo "Processes will continue even if you disconnect!"
echo "=========================================="

