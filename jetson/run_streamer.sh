#!/bin/bash
# Run camera streamer on Jetson Nano
# Streams camera frames to the processing server

# Jetson performance mode
echo "[Setup] Setting Jetson performance mode..."
sudo jetson_clocks 2>/dev/null || echo "[WARN] jetson_clocks not available"

if [ -f /sys/devices/gpu.0/devfreq/57000000.gpu/governor ]; then
    echo "performance" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/governor > /dev/null 2>&1
fi

export MALLOC_TRIM_THRESHOLD_=65536
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

# Default server, can override via args
SERVER="${1:-192.168.2.10}"
PORT="${2:-9000}"

echo "[Setup] Starting camera streamer → ${SERVER}:${PORT}"
python3 camera_streamer.py --server "$SERVER" --port "$PORT"
