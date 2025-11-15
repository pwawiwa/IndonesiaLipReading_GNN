"""
Utilities for exposing recent training logs over HTTP.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Deque, List, Tuple


def _tail_lines(log_path: Path, count: int) -> List[str]:
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    
    buffer: Deque[str] = deque(maxlen=count)
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.rstrip()
            if stripped:
                buffer.append(stripped)
    return list(buffer)


def _build_handler(log_path: Path, entries: int, refresh_seconds: int):
    class LogRequestHandler(BaseHTTPRequestHandler):
        _log_path = Path(log_path)
        _entries = max(1, entries)
        _refresh_seconds = max(5, refresh_seconds)
        
        def _set_headers(self, status: int = 200, content_type: str = "text/html"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
        
        def _handle_root(self):
            self._set_headers()
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Training Logs Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #0d1117;
      color: #e6edf3;
      margin: 0;
      padding: 2rem;
    }}
    h1 {{
      margin-bottom: 0.5rem;
    }}
    .meta {{
      color: #7d8590;
      margin-bottom: 1.5rem;
    }}
    pre {{
      background: #161b22;
      padding: 1rem;
      border-radius: 8px;
      overflow-x: auto;
      border: 1px solid #30363d;
      line-height: 1.4;
    }}
    .timestamp {{
      font-size: 0.9rem;
      color: #a1b0c3;
    }}
  </style>
  <script>
    const refreshMs = {self._refresh_seconds * 1000};
    async function fetchLogs() {{
      try {{
        const response = await fetch('/logs');
        const payload = await response.json();
        const container = document.getElementById('log-container');
        const ts = document.getElementById('timestamp');
        container.textContent = payload.logs.join('\\n') || 'No log entries yet.';
        ts.textContent = `Updated: ${{
          new Date(payload.generated_at).toLocaleString()
        }}`;
      }} catch (err) {{
        console.error(err);
      }}
    }}
    window.addEventListener('load', () => {{
      fetchLogs();
      setInterval(fetchLogs, refreshMs);
    }});
  </script>
</head>
<body>
  <h1>Latest Training Logs</h1>
  <div class="meta">
    Showing the {self._entries} most recent events. Auto-refresh every {self._refresh_seconds // 60 or 1} minutes.
  </div>
  <div class="timestamp" id="timestamp">Updated: --</div>
  <pre id="log-container">Loading…</pre>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        
        def _handle_logs(self):
            entries = _tail_lines(self._log_path, self._entries)
            payload = {
                "logs": entries,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
            self._set_headers(content_type="application/json")
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        
        def log_message(self, format, *args):
            # Silence default request logging
            return
        
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._handle_root()
            elif self.path.startswith("/logs"):
                self._handle_logs()
            else:
                self._set_headers(status=404)
                self.wfile.write(b"Not Found")
    
    return LogRequestHandler


def create_log_server(log_path: Path, host: str = "127.0.0.1", port: int = 8080,
                      entries: int = 10, refresh_minutes: int = 5) -> HTTPServer:
    handler = _build_handler(
        log_path=log_path,
        entries=entries,
        refresh_seconds=max(5, refresh_minutes * 60)
    )
    return HTTPServer((host, port), handler)


def start_log_server(log_path: Path, host: str = "127.0.0.1", port: int = 8080,
                     entries: int = 10, refresh_minutes: int = 5) -> Tuple[HTTPServer, threading.Thread]:
    server = create_log_server(log_path, host, port, entries, refresh_minutes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def run_log_server_blocking(log_path: Path, host: str = "127.0.0.1", port: int = 8080,
                            entries: int = 10, refresh_minutes: int = 5):
    server = create_log_server(log_path, host, port, entries, refresh_minutes)
    print(f"Serving log dashboard at http://{host}:{port}/")
    print(f"Reading from {log_path} (showing {entries} entries)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down log server...")
    finally:
        server.server_close()

