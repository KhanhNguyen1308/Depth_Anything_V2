"""
Configuration for Jetson Nano camera streamer.
Camera settings and server connection.
UGREEN 4K mono USB camera.
"""

# === Camera ===
CAMERA_INDEX = 0             # USB camera index (UGREEN 4K mono)
CAMERA_WIDTH = 3840          # Capture resolution — 4K
CAMERA_HEIGHT = 2160
CAMERA_FPS = 15              # 4K MJPEG @ 15fps (USB bandwidth limit)

# Stream resolution: resize before sending to reduce network bandwidth.
# Depth Anything V2 internally resizes to MODEL_INPUT_SIZE anyway.
# Set to None to stream at full capture resolution (very heavy at 4K).
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

# Camera exposure control
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = None            # None = auto

# GStreamer pipeline (True = GStreamer MJPEG decode, False = V4L2)
# V4L2 is more reliable for UGREEN UVC cameras; use False unless GStreamer is needed.
USE_GSTREAMER = False

# === Server Connection (TCP mode) ===
SERVER_HOST = "192.168.2.10"
SERVER_PORT = 9000
JPEG_QUALITY = 85            # JPEG quality for streaming to server

# === Cloudflared Tunnel Mode ===
TUNNEL_STREAM_PORT = 9000   # HTTP MJPEG port for cloudflared tunnel
TUNNEL_DOMAIN = "https://stream.ndkforge.io.vn"

# === Calibration Web UI ===
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
STREAM_FPS = 15
