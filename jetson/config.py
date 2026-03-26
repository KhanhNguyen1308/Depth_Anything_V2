"""
Configuration for Jetson Nano camera streamer.
UGREEN 4K mono USB camera.
Capture at 4K (full sensor FOV) then downscale to 1080p before streaming.
If you capture at 1080p directly, the camera crops the sensor center → zoomed-in look.
"""

# === Camera ===
CAMERA_INDEX = 0             # USB camera index (UGREEN 4K mono)
CAMERA_WIDTH = 3840          # Capture at full 4K sensor resolution
CAMERA_HEIGHT = 2160
CAMERA_FPS = 30              # 4K MJPEG @ 30fps (camera native, use GStreamer)

# GStreamer pipeline resizes before output — STREAM_WIDTH/HEIGHT used in pipeline.
# At full 4K raw decode would be 24MB/frame; GStreamer resizes inside the pipeline.
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

# Camera exposure control
CAMERA_EXPOSURE_AUTO = 3      # 1=manual, 3=aperture_priority (auto)
CAMERA_GAIN = None            # None = auto

# GStreamer pipeline — must be True for 4K to avoid out-of-memory crash.
# GStreamer decodes MJPEG and scales to STREAM_WIDTH×STREAM_HEIGHT inside the pipeline.
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
STREAM_FPS = 30
