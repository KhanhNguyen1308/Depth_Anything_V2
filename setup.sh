#!/bin/bash
#
# Setup script cho Depth Anything V2 trên Jetson Nano
# Ubuntu 20.04, Python 3.8, JetPack 4.6+
#

set -e

echo "============================================"
echo " Depth Anything V2 - Jetson Nano Setup"
echo "============================================"

# Kiểm tra Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "[WARN] Không phát hiện Jetson platform. Script này dành cho Jetson Nano."
    read -p "Tiếp tục? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        exit 0
    fi
fi

# System dependencies
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libopenblas-base \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev

# Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# PyTorch for Jetson (nếu chưa cài)
echo ""
echo "[3/5] Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} OK, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "PyTorch chưa được cài. Cài đặt PyTorch cho Jetson..."
    echo ""
    echo "Tải PyTorch wheel từ NVIDIA:"
    echo "  wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.13.0-cp38-cp38-linux_aarch64.whl"
    echo "  pip3 install torch-1.13.0-cp38-cp38-linux_aarch64.whl"
    echo ""
    echo "Sau đó cài torchvision:"
    echo "  git clone --branch v0.14.0 https://github.com/pytorch/vision torchvision"
    echo "  cd torchvision && python3 setup.py install && cd .."
    echo ""
    echo "Sau khi cài PyTorch, chạy lại script này."
    exit 1
}

# Python dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
pip3 install -r requirements.txt

# TensorRT & ONNX Runtime check
echo ""
echo "[5/8] Checking inference backends..."
python3 -c "import tensorrt as trt; print(f'TensorRT {trt.__version__} OK')" 2>/dev/null || {
    echo "TensorRT chưa cài. Cài đặt:"
    echo "  sudo apt-get install tensorrt python3-libnvinfer-dev"
}

# PyCUDA check & install
echo ""
echo "[6/8] Installing PyCUDA..."
python3 -c "import pycuda; print('PyCUDA OK')" 2>/dev/null || {
    pip3 install pycuda
}

# Patch PyCUDA & TensorRT cho Python 3.8 + numpy >= 1.24
echo ""
echo "[7/8] Patching PyCUDA/TensorRT cho Python 3.8..."
PYCUDA_DIR=$(python3 -c "import pycuda; import os; print(os.path.dirname(pycuda.__file__))" 2>/dev/null)
if [ -n "$PYCUDA_DIR" ] && [ -f "$PYCUDA_DIR/compyte/dtypes.py" ]; then
    if ! grep -q "from __future__ import annotations" "$PYCUDA_DIR/compyte/dtypes.py" 2>/dev/null; then
        sed -i '1i from __future__ import annotations' "$PYCUDA_DIR/compyte/dtypes.py"
        echo "  Patched PyCUDA type hints OK"
    else
        echo "  PyCUDA already patched"
    fi
fi
TRT_INIT=$(python3 -c "import tensorrt; import os; print(os.path.join(os.path.dirname(tensorrt.__file__), '__init__.py'))" 2>/dev/null)
if [ -n "$TRT_INIT" ] && grep -q "np.bool," "$TRT_INIT" 2>/dev/null; then
    sed -i 's/np\.bool,/np.bool_,/' "$TRT_INIT"
    echo "  Patched TensorRT np.bool OK"
else
    echo "  TensorRT np.bool already patched or not needed"
fi

# Fix ONNX model & build TRT engine
echo ""
echo "[8/8] Preparing ONNX model & TensorRT engine..."
ONNX_FILE="depth_anything_v2_vits.onnx"
FIXED_ONNX="depth_anything_v2_vits_fixed.onnx"
ENGINE_FILE="checkpoints/depth_anything_v2_vits_fixed_fp16.engine"

if [ -f "$ENGINE_FILE" ]; then
    echo "TensorRT engine đã tồn tại: $ENGINE_FILE"
elif [ -f "$ONNX_FILE" ]; then
    pip3 install onnx --quiet 2>/dev/null
    if [ ! -f "$FIXED_ONNX" ]; then
        echo "Fixing ONNX model for Jetson Nano..."
        python3 fix_onnx_model.py --input "$ONNX_FILE" --output "$FIXED_ONNX"
    fi
    echo "Converting to TensorRT engine (mất vài phút)..."
    python3 convert_onnx_to_trt.py --onnx "$FIXED_ONNX" --fp16 --height 518 --width 518 --workspace 512
else
    echo "WARN: Không tìm thấy $ONNX_FILE"
    echo "Đặt file ONNX vào thư mục project rồi chạy lại setup."
fi

echo ""
echo "============================================"
echo " Setup hoàn tất!"
echo ""
echo " Test TensorRT inference:"
echo "   python3 test_trt.py"
echo ""
echo " Chạy ứng dụng:"
echo "   python3 main.py"
echo ""
echo " Truy cập web UI:"
echo "   http://$(hostname -I | awk '{print $1}'):5000"
echo "============================================"
