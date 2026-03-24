"""
Configuration for Jetson Nano camera streamer.
Camera settings and server connection.
"""

# === Camera ===
CAMERA_INDEX = 0             # USB camera index
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# Camera exposure control
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = None            # None = auto

# GStreamer pipeline (True = GStreamer MJPEG decode, False = V4L2)
USE_GSTREAMER = True

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
