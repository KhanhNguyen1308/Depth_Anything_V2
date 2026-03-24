#!/bin/bash
# Run HTTP camera streamer on Jetson Nano with cloudflared tunnel
# Serves MJPEG stream on port 9000 for cloudflared to tunnel

# Jetson performance mode
echo "[Setup] Setting Jetson performance mode..."
sudo jetson_clocks 2>/dev/null || echo "[WARN] jetson_clocks not available"

if [ -f /sys/devices/gpu.0/devfreq/57000000.gpu/governor ]; then
    echo "performance" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/governor > /dev/null 2>&1
fi

export MALLOC_TRIM_THRESHOLD_=65536
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

PORT="${1:-9000}"

echo "[Setup] Starting HTTP camera streamer on port ${PORT}"
echo "[Setup] Use cloudflared to tunnel this port:"
echo "  cloudflared tunnel --url http://localhost:${PORT}"

python3 http_streamer.py --port "$PORT"
