#!/bin/bash
# ============================================
# Kill running extraction process
# ============================================

cd "$(dirname "$0")/.."

PID_FILE="logs/extraction_all.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "No extraction PID file found. Extraction may not be running."
    exit 1
fi

PID=$(cat "${PID_FILE}")

if ps -p "${PID}" > /dev/null 2>&1; then
    echo "Stopping extraction (PID: ${PID})..."
    kill "${PID}"
    
    # Wait a bit, then force kill if still running
    sleep 2
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 "${PID}"
    fi
    
    rm -f "${PID_FILE}"
    echo "✓ Extraction stopped"
else
    echo "Process ${PID} not found. Removing stale PID file."
    rm -f "${PID_FILE}"
fi

