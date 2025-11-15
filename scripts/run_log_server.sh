#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/member2/tomoooo/IndonesiaLipReading_GNN"
cd "$ROOT"

LOG_FILE="outputs/combined/training.log"
mkdir -p "$(dirname "$LOG_FILE")"

HOST="127.0.0.1"
PORT="8081"

echo "Starting log server on http://$HOST:$PORT/ using $LOG_FILE"
nohup /usr/bin/python3 scripts/log_server.py \
  --log-file "$LOG_FILE" \
  --host "$HOST" \
  --port "$PORT" \
  > log_server.out 2>&1 &
echo $! > outputs/combined/log_server.pid
echo "Log server PID: $(cat outputs/combined/log_server.pid)"

