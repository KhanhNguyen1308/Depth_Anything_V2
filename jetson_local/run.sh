#!/bin/bash
# Jetson Nano Local Mode - Run script
# Runs Depth Anything V2 + YOLOv8n + Web Dashboard locally

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Jetson Nano Local Mode ==="
echo "Setting performance mode..."

# Max performance
sudo jetson_clocks 2>/dev/null || true
sudo nvpmodel -m 0 2>/dev/null || true

# Camera exposure (optional)
CAMERA_DEV="/dev/video0"
if [ -e "$CAMERA_DEV" ]; then
    v4l2-ctl -d "$CAMERA_DEV" --set-ctrl=exposure_auto=3 2>/dev/null || true
    echo "Camera: $CAMERA_DEV configured"
else
    echo "WARNING: $CAMERA_DEV not found"
fi

echo "Starting local processor..."
echo "Web dashboard: http://$(hostname -I | awk '{print $1}'):8080"
echo ""

python3 main.py
