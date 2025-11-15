#!/usr/bin/env python3
"""
Wrapper script to expose the log dashboard on demand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.log_dashboard import run_log_server_blocking


def parse_args():
    parser = argparse.ArgumentParser(description="Serve recent training logs over HTTP.")
    parser.add_argument("--log-file", type=Path, default=Path("outputs/combined/training.log"),
                        help="Path to the training log file.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to listen on (default: 8080).")
    parser.add_argument("--entries", type=int, default=10,
                        help="Number of recent log lines to display.")
    parser.add_argument("--refresh-minutes", type=int, default=5,
                        help="Auto-refresh cadence for the dashboard (minutes).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_log_server_blocking(
        log_path=args.log_file,
        host=args.host,
        port=args.port,
        entries=args.entries,
        refresh_minutes=args.refresh_minutes
    )


if __name__ == "__main__":
    main()

