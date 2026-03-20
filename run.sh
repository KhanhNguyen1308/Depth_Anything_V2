#!/bin/bash
#
# Script chạy ứng dụng Depth Anything V2
# Tối ưu cho Jetson Nano
#

# Tối ưu hiệu năng Jetson
echo "[Setup] Setting Jetson to max performance mode..."
sudo jetson_clocks 2>/dev/null || echo "[WARN] jetson_clocks not available"

# Set max GPU/CPU frequency (nếu có quyền)
if [ -f /sys/devices/gpu.0/devfreq/57000000.gpu/governor ]; then
    echo "performance" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/governor > /dev/null 2>&1
fi

# Tối ưu memory
export MALLOC_TRIM_THRESHOLD_=65536

# PyTorch optimizations
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Disable Python buffering (để thấy log realtime)
export PYTHONUNBUFFERED=1

echo "[Setup] Starting Depth Anything V2..."
python3 main.py
