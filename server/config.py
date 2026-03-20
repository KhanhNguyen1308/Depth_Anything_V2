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
MODEL_ENCODER = "vits"      # "vits" or "vitb"
MODEL_DIR = "checkpoints"
MODEL_INPUT_SIZE = 252

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

# === Web Server ===
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
JPEG_QUALITY = 85
STREAM_FPS = 30
