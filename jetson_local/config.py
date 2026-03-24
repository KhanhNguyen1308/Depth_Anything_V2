"""
Configuration for Jetson Nano local mode.
Runs depth estimation + object detection + web dashboard all on-device.
"""

# === Camera ===
CAMERA_INDEX = 0             # USB camera index
CAMERA_WIDTH = 640           # Lower res for Jetson Nano performance
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Camera exposure control
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = None            # None = auto

# GStreamer pipeline (True = GStreamer MJPEG decode, False = V4L2)
USE_GSTREAMER = True

# === Depth Anything V2 Model ===
MODEL_ENCODER = "vits"       # "vits" recommended for Jetson Nano (fastest)
MODEL_DIR = "checkpoints"
MODEL_INPUT_SIZE = 308       # Smaller input for Jetson Nano speed

# === Metric Depth ===
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

# === Object Detection (YOLOv8) ===
YOLO_ENABLED = True            # Enable/disable object detection
YOLO_MODEL = "yolov8n.pt"     # yolov8n = fastest, best for Jetson Nano
YOLO_CONFIDENCE = 0.5          # Detection confidence threshold
YOLO_INPUT_SIZE = 320          # YOLOv8 input size
YOLO_DEVICE = "auto"           # "auto", "cuda", or "cpu"
YOLO_SKIP_FRAMES = 3           # Run detection every N frames (saves GPU)

# === Web Server ===
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
JPEG_QUALITY = 80
STREAM_FPS = 15               # Lower FPS for web stream to save bandwidth
