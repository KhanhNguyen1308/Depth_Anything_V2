#!/bin/bash
# Run the depth processing server

export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

PORT="${1:-9000}"
WEB_PORT="${2:-5000}"

echo "[Server] Starting depth processing server..."
echo "[Server] Camera stream port: ${PORT}"
echo "[Server] Web UI port: ${WEB_PORT}"

python3 server.py --port "$PORT" --web-port "$WEB_PORT"
