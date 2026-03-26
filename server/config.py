"""
Configuration for the processing server.
Receives camera stream from Jetson, runs depth estimation + object detection.
"""

# === Camera (must match Jetson STREAM_WIDTH/STREAM_HEIGHT) ===
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

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

# === Object Detection (YOLOv8) ===
YOLO_ENABLED = True            # Enable/disable object detection
YOLO_MODEL = "yolov8m.pt"      # Model: yolov8n/s/m/l/x (.pt auto-downloads)
YOLO_CONFIDENCE = 0.5          # Detection confidence threshold
YOLO_INPUT_SIZE = 320          # YOLOv8 input size (smaller = faster)
YOLO_DEVICE = "auto"           # "auto", "cuda", or "cpu"
YOLO_SKIP_FRAMES = 2           # Run detection every N frames (1=every frame)

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
