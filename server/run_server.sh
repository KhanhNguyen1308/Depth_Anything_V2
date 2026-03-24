#!/bin/bash
# Run the depth processing server
# Usage:
#   ./run_server.sh                     # TCP mode, port 9000, web on 6000
#   ./run_server.sh tunnel              # Tunnel mode, web on 6000
#   ./run_server.sh tcp 9000 6000       # TCP mode with custom ports

export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

SOURCE="${1:-tcp}"
PORT="${2:-9000}"
WEB_PORT="${3:-6000}"

echo "[Server] Starting depth processing server..."
echo "[Server] Source mode: ${SOURCE}"

if [ "$SOURCE" = "tunnel" ]; then
    echo "[Server] Reading from cloudflared tunnel"
    echo "[Server] Web UI port: ${WEB_PORT}"
    python3 server.py --source tunnel --web-port "$WEB_PORT"
else
    echo "[Server] Camera stream port: ${PORT}"
    echo "[Server] Web UI port: ${WEB_PORT}"
    python3 server.py --source tcp --port "$PORT" --web-port "$WEB_PORT"
fi
