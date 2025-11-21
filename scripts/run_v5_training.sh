#!/bin/bash
# Run V5 AST-GCN training in background (survives disconnection)
# Uses full face (468 landmarks) with MediaPipe FACEMESH_TESSELATION connections
set -euo pipefail

ROOT="/home/member2/tomoooo/IndonesiaLipReading_GNN"
cd "$ROOT"

LOG_DIR="outputs/v5"
mkdir -p "$LOG_DIR"

STDOUT_LOG="$LOG_DIR/train_stdout.log"
STDERR_LOG="$LOG_DIR/train_stderr.log"

echo "=========================================="
echo "V5 AST-GCN Training"
echo "Full Face (468 landmarks) + MediaPipe Connections"
echo "=========================================="
echo "Config: configs/v5.py (processed_v3)"
echo "Log directory: $LOG_DIR"
echo "=========================================="

# Kill existing training if it exists
if [ -f "$LOG_DIR/train.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/train.pid")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Stopping existing training (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$LOG_DIR/train.pid"
fi

# Check if screen is available (preferred method)
if command -v screen &> /dev/null; then
    SCREEN_NAME="lipreading_v5"
    
    # Kill existing screen session if exists
    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "Stopping existing screen session: $SCREEN_NAME"
        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
        sleep 1
    fi
    
    echo ""
    echo "Starting training in screen session: $SCREEN_NAME"
    echo "To attach: screen -r $SCREEN_NAME"
    echo "To detach: Ctrl+A then D"
    
    # Start training directly
    screen -dmS "$SCREEN_NAME" bash -c "cd '$ROOT' && python3 -u src/train.py v5.py > '$STDOUT_LOG' 2> '$STDERR_LOG'; exec bash"
    
    sleep 2
    PYTHON_PID=$(ps aux | grep "[p]ython3.*src/train.py" | awk '{print $2}' | head -1)
    if [ -n "$PYTHON_PID" ]; then
        echo "$PYTHON_PID" > "$LOG_DIR/train.pid"
        echo "✓ Training started in screen (PID: $PYTHON_PID)"
    else
        echo "✓ Training started in screen (check with: screen -r $SCREEN_NAME)"
    fi
    
    echo ""
    echo "To attach to session:"
    echo "  screen -r $SCREEN_NAME"
    echo ""
    echo "To view logs without attaching:"
    echo "  tail -f $STDOUT_LOG"
    
else
    # Fallback to nohup with setsid for better persistence
    echo ""
    echo "Starting training with nohup + setsid (screen not available)..."
    # Use setsid to create a new session, and redirect all output properly
    nohup setsid bash -c "cd '$ROOT' && exec python3 -u src/train.py v5.py" \
        > "$STDOUT_LOG" 2> "$STDERR_LOG" &
    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$LOG_DIR/train.pid"
    echo "✓ Training started (PID: $TRAIN_PID)"
    echo ""
    echo "To view logs:"
    echo "  tail -f $STDOUT_LOG"
fi

echo ""
echo "=========================================="
echo "Training will continue even if you disconnect!"
echo "=========================================="
echo "To check status:"
echo "  ps aux | grep train.py"
echo "  tail -f $STDOUT_LOG"
echo ""
echo "To stop:"
echo "  bash scripts/stop_training.sh"
echo "  or: kill \$(cat $LOG_DIR/train.pid)"
echo "=========================================="


