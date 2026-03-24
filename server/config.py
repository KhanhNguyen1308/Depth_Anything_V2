"""
Configuration for the processing server.
Receives camera streams from Jetson, runs depth estimation.
"""

# === Camera (must match Jetson config) ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# === Stereo Calibration ===
# Copy calibration.npz from the Jetson after calibrating
STEREO_CALIB_FILE = "calibration.npz"
STEREO_CALIB_WIDTH = 640
STEREO_CALIB_HEIGHT = 480

# === Depth Anything V2 Model ===
MODEL_ENCODER = "vitb"      # "vits", "vitb", or "vitl"
MODEL_DIR = "checkpoints"
MODEL_INPUT_SIZE = 518

# === Metric Depth ===
# Use metric depth model for absolute distance (meters) instead of relative depth.
# Requires metric depth checkpoint: depth_anything_v2_metric_{METRIC_DATASET}_{MODEL_ENCODER}.pth
METRIC_DEPTH = True
METRIC_DATASET = "hypersim"  # "hypersim" (indoor, max 20m) or "vkitti" (outdoor, max 80m)
MAX_DEPTH = 20               # 20 for hypersim, 80 for vkitti

# === Inference Backend ===
# "tensorrt" | "onnxrt" | "pytorch"
INFERENCE_BACKEND = "pytorch"

ONNX_MODEL = "depth_anything_v2_vits_fixed.onnx"
TENSORRT_ENGINE = "depth_anything_v2_vits_fixed_fp16.engine"
TENSORRT_WORKSPACE_MB = 512

USE_FP16 = True
TARGET_FPS = 30

# === Network ===
STREAM_PORT = 9000          # TCP port for receiving camera frames

# === Cloudflared Tunnel ===
# Set STREAM_SOURCE to "tunnel" to read from cloudflared tunnel instead of TCP
STREAM_SOURCE = "tcp"       # "tcp" (LAN) or "tunnel" (cloudflared)
TUNNEL_STREAM_URL = "https://stream.ndkforge.io.vn"

# === Web Server ===
WEB_HOST = "0.0.0.0"
WEB_PORT = 6000
JPEG_QUALITY = 85
STREAM_FPS = 30
