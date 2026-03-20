"""
Cấu hình hệ thống Depth Anything V2 - Jetson Nano
"""

# === Camera ===
CAMERA_0_INDEX = 0          # USB0 = physical LEFT camera
CAMERA_1_INDEX = 1          # USB1 = physical RIGHT camera
CAMERA_WIDTH = 640          # Độ phân giải ngang
CAMERA_HEIGHT = 480         # Độ phân giải dọc
CAMERA_FPS = 30             # FPS camera

# Camera exposure control (OV9732)
# Dùng v4l2-ctl để set sau khi mở camera
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = 64              # Gain limit (lower = less noise/brightness)

# Stereo calibration file (từ cv2.stereoCalibrate)
# Chứa: mtx_l, dist_l, mtx_r, dist_r, R, T
STEREO_CALIB_FILE = "calibration.npz"
STEREO_CALIB_WIDTH = 640    # Resolution khi calibrate
STEREO_CALIB_HEIGHT = 480

# GStreamer pipeline cho OV9732 trên Jetson Nano (tối ưu hơn V4L2)
# Đặt USE_GSTREAMER = True nếu muốn dùng GStreamer
USE_GSTREAMER = True

# === Depth Anything V2 Model ===
# "vits" (nhẹ, nhanh) hoặc "vitb" (chính xác hơn, chậm hơn)
MODEL_ENCODER = "vits"
MODEL_DIR = "checkpoints"
# Input size cho model (model gốc: 518, giảm = nhanh hơn nhưng cần re-export ONNX)
MODEL_INPUT_SIZE = 252

# === Inference Backend ===
# "tensorrt" (nhanh nhất, cần convert ONNX trước)
# "onnxrt" (ONNX Runtime + TRT/CUDA EP)
# "pytorch" (fallback)
INFERENCE_BACKEND = "tensorrt"

# ONNX model file (dùng để convert sang TensorRT)
# Dùng file _fixed.onnx đã chuẩn bị cho Jetson Nano (IR v8, opset 17, single file)
ONNX_MODEL = "depth_anything_v2_vits_fixed.onnx"

# TensorRT engine file (trong MODEL_DIR)
# Tạo bằng: python3 fix_onnx_model.py && python3 convert_onnx_to_trt.py --onnx depth_anything_v2_vits_fixed.onnx --fp16 --height 518 --width 518
TENSORRT_ENGINE = "depth_anything_v2_vits_fixed_fp16.engine"

# TensorRT workspace size (MB) - Jetson Nano 4GB nên dùng 512
TENSORRT_WORKSPACE_MB = 512

USE_FP16 = True             # Dùng FP16 (PyTorch backend)
TARGET_FPS = 15             # FPS mục tiêu cho depth inference

# === Web Server ===
WEB_HOST = "0.0.0.0"       # Lắng nghe tất cả interfaces
WEB_PORT = 5000
JPEG_QUALITY = 85           # Chất lượng JPEG streaming (1-100)
STREAM_FPS = 20             # FPS streaming lên web
