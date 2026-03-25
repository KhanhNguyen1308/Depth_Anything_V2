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
USE_GSTREAMER = False

# === Depth Anything V2 Model ===
MODEL_ENCODER = "vits"       # "vits" recommended for Jetson Nano (fastest)
MODEL_DIR = "checkpoints"
MODEL_INPUT_SIZE = 308       # Smaller input for Jetson Nano speed

# === Metric Depth ===
METRIC_DEPTH = True
METRIC_DATASET = "hypersim"  # "hypersim" (indoor, max 20m) or "vkitti" (outdoor, max 80m)
MAX_DEPTH = 20               # 20 for hypersim, 80 for vkitti

# === Depth Calibration ===
# The metric model is calibrated for its training camera (Hypersim virtual cam).
# Real webcams with a different FOV will have a systematic offset.
# Corrected depth = (raw_depth - DEPTH_OFFSET) / DEPTH_SCALE
# Set both to defaults (0.0 / 1.0) to disable correction.
#
# How to calibrate (need ≥3 measurements):
#   1. Measure real distances (e.g. 0.5, 1.0, 1.5, 2.0 m) to a flat surface
#   2. Read the raw depth value at that pixel from the console "[depth]" log
#   3. Run: python3 calibrate_depth.py  (will print DEPTH_SCALE and DEPTH_OFFSET)
#
# Quick two-point estimate from user measurements (OLD squished 308×308 engine):
#   real=0.60m → raw≈1.63m    real=1.68m → raw≈2.43m
#   → DEPTH_SCALE ≈ 0.741,  DEPTH_OFFSET ≈ 1.185
# The offset (~1.19m) is model-intrinsic (Hypersim virtual camera vs real webcam FOV).
# The same values apply to the new 308×420 engine since the model weights are unchanged.
# To re-calibrate: measure at 2+ known distances, read raw depth from the console
# "[TRT] Output" line (or use center crosshair on the depth view), then:
#   DEPTH_SCALE  = (raw2 - raw1) / (real2 - real1)
#   DEPTH_OFFSET = raw1 - DEPTH_SCALE * real1
DEPTH_SCALE  = 0.741   # slope:     raw = DEPTH_SCALE * real + DEPTH_OFFSET
DEPTH_OFFSET = 1.186   # intercept: corrected = (raw - DEPTH_OFFSET) / DEPTH_SCALE

# === Inference Backend ===
# "tensorrt" | "onnxrt" | "pytorch"
INFERENCE_BACKEND = "tensorrt"

ONNX_MODEL = "hypersim_vits_308x420.onnx"
TENSORRT_ENGINE = "hypersim_vits_308x420.engine"
TENSORRT_WORKSPACE_MB = 512

USE_FP16 = True

# === Object Detection (YOLOv8) ===
YOLO_ENABLED = False           # Enable/disable object detection
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
