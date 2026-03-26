"""
Configuration for Jetson Nano camera streamer.
UGREEN 4K mono USB camera — capture at 1080p (most compatible native UVC mode).
Depth Anything V2 resizes input to 518px internally, so 4K capture gives no depth benefit.
"""

# === Camera ===
CAMERA_INDEX = 0             # USB camera index (UGREEN 4K mono)
CAMERA_WIDTH = 1920          # Capture at 1080p — reliable on all USB modes
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# Stream resolution override (resize before sending).
# Keep None to stream at capture resolution (recommended at 1080p).
STREAM_WIDTH = None
STREAM_HEIGHT = None

# Camera exposure control
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = None            # None = auto

# GStreamer pipeline (True = GStreamer MJPEG decode, False = V4L2)
# V4L2 is more reliable for UGREEN UVC cameras.
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
STREAM_FPS = 30
