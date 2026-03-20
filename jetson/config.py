"""
Configuration for Jetson Nano camera streamer.
Camera settings and server connection.
"""

# === Camera ===
CAMERA_0_INDEX = 0          # USB0 = physical LEFT camera
CAMERA_1_INDEX = 1          # USB1 = physical RIGHT camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Camera exposure control (OV9732)
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = 64

# GStreamer pipeline (True = GStreamer MJPEG decode, False = V4L2)
USE_GSTREAMER = True

# === Stereo Calibration ===
STEREO_CALIB_FILE = "calibration.npz"
STEREO_CALIB_WIDTH = 640
STEREO_CALIB_HEIGHT = 480

# === Server Connection ===
SERVER_HOST = "192.168.2.10"
SERVER_PORT = 9000
JPEG_QUALITY = 90           # JPEG quality for streaming to server

# === Calibration Web UI ===
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
STREAM_FPS = 15
