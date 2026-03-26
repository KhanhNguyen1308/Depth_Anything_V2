"""
HTTP Camera Streamer - Runs on Jetson Nano.
Serves camera frames as MJPEG over HTTP on port 9000.
Designed to work behind a cloudflared tunnel for remote access.

Usage:
    python3 http_streamer.py
    python3 http_streamer.py --port 9000
"""

import argparse
import cv2
import numpy as np
import signal
import subprocess
import sys
import threading
import time

from flask import Flask, Response

import config

class FfmpegCamera:
    """
    Camera capture via ffmpeg subprocess pipe.
    Reads 4K MJPEG from v4l2, scales inside ffmpeg, outputs raw BGR frames.
    Avoids allocating huge raw frames in Python and doesn't need OpenCV GStreamer.
    """

    def __init__(self, device, in_w, in_h, fps, out_w, out_h):
        self._out_w = out_w
        self._out_h = out_h
        self._fps = fps
        self._frame_bytes = out_w * out_h * 3

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", f"{in_w}x{in_h}",
            "-framerate", str(fps),
            "-i", device,
            "-vf", f"scale={out_w}:{out_h}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 2,
        )

    def isOpened(self):
        return self._proc.poll() is None

    def read(self):
        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._out_h, self._out_w, 3)
        return True, frame.copy()

    def grab(self):
        """Read and discard one frame (used for buffer flush)."""
        raw = self._proc.stdout.read(self._frame_bytes)
        return len(raw) == self._frame_bytes

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._out_w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._out_h)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def release(self):
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


app = Flask(__name__)

_frame_lock = threading.Lock()
_current_frame = None


def _configure_exposure(camera_index, name):
    """Configure camera exposure via v4l2-ctl."""
    device = f"/dev/video{camera_index}"
    controls = [
        ("exposure_auto", config.CAMERA_EXPOSURE_AUTO),
        ("white_balance_temperature_auto", 1),
    ]
    if config.CAMERA_EXPOSURE_AUTO == 1:
        expo_val = getattr(config, "CAMERA_EXPOSURE_VALUE", 300)
        controls.append(("exposure_absolute", expo_val))
    gain = getattr(config, "CAMERA_GAIN", None)
    if gain is not None:
        controls.append(("gain", gain))

    for ctrl_name, ctrl_val in controls:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, "--set-ctrl", f"{ctrl_name}={ctrl_val}"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                print(f"  [{name}] v4l2: {ctrl_name}={ctrl_val}")
        except Exception:
            pass


def open_camera(index, name):
    """Open a USB camera."""
    sw = getattr(config, "STREAM_WIDTH", None)
    sh = getattr(config, "STREAM_HEIGHT", None)
    need_scale = sw and sh and (sw != config.CAMERA_WIDTH or sh != config.CAMERA_HEIGHT)

    if config.USE_GSTREAMER:
        if sw and sh:
            pipeline = (
                f"v4l2src device=/dev/video{index} ! "
                f"image/jpeg,width={config.CAMERA_WIDTH},height={config.CAMERA_HEIGHT},"
                f"framerate={config.CAMERA_FPS}/1 ! "
                f"jpegdec ! videoscale ! "
                f"video/x-raw,width={sw},height={sh} ! "
                f"videoconvert ! appsink drop=true sync=false max-buffers=1"
            )
        else:
            pipeline = (
                f"v4l2src device=/dev/video{index} ! "
                f"image/jpeg,width={config.CAMERA_WIDTH},height={config.CAMERA_HEIGHT},"
                f"framerate={config.CAMERA_FPS}/1 ! "
                f"jpegdec ! videoconvert ! "
                f"appsink drop=true sync=false max-buffers=1"
            )
        print(f"  [{name}] GStreamer pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif need_scale:
        # Use ffmpeg pipe: 4K decode + scale inside ffmpeg, Python gets 1080p frames.
        print(f"  [{name}] Using ffmpeg pipe: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} "
              f"\u2192 {sw}x{sh}")
        cap = FfmpegCamera(
            f"/dev/video{index}",
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS,
            sw, sh,
        )
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  [{name}] Opened: {w}x{h} @ {fps:.0f}fps")

    _configure_exposure(index, name)
    return cap


def _capture_loop(cap, quality, stream_override=None):
    """Continuously capture frames from camera."""
    global _current_frame
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    frame_count = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue

        # Resize only if an explicit stream override is configured
        if stream_override and (frame.shape[1] != stream_override[0] or frame.shape[0] != stream_override[1]):
            frame = cv2.resize(frame, stream_override, interpolation=cv2.INTER_AREA)

        _, jpeg = cv2.imencode(".jpg", frame, encode_param)
        if jpeg is not None:
            with _frame_lock:
                _current_frame = jpeg.tobytes()

        frame_count += 1
        now = time.time()
        if now - fps_time >= 5.0:
            fps = frame_count / (now - fps_time)
            print(f"[HTTP Streamer] Capturing at {fps:.1f} FPS")
            frame_count = 0
            fps_time = now


def _generate_mjpeg():
    """MJPEG stream generator."""
    interval = 1.0 / config.CAMERA_FPS
    while True:
        with _frame_lock:
            frame_data = _current_frame

        if frame_data is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_data
            + b"\r\n"
        )
        time.sleep(interval)


@app.route("/stream")
def video_stream():
    """MJPEG video stream endpoint."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    """Health check for cloudflared."""
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="HTTP Camera Streamer for Jetson Nano")
    parser.add_argument("--port", type=int, default=config.TUNNEL_STREAM_PORT,
                        help="HTTP server port")
    parser.add_argument("--quality", type=int, default=config.JPEG_QUALITY,
                        help="JPEG quality")
    args = parser.parse_args()

    print("=" * 50)
    print("  HTTP Camera Streamer - Jetson Nano")
    print("  (Cloudflared Tunnel Mode)")
    print("=" * 50)
    print(f"  HTTP port: {args.port}")
    print(f"  Stream URL: http://0.0.0.0:{args.port}/stream")
    print(f"  Camera: {config.CAMERA_INDEX}")
    print(f"  Capture: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ {config.CAMERA_FPS}fps")
    sw = getattr(config, "STREAM_WIDTH", None)
    sh = getattr(config, "STREAM_HEIGHT", None)
    if sw and sh and (sw != config.CAMERA_WIDTH or sh != config.CAMERA_HEIGHT):
        print(f"  Stream:  {sw}x{sh} (resized before send)")
    print(f"  JPEG quality: {args.quality}")
    print("=" * 50)

    # Open camera
    print("\n[HTTP Streamer] Opening camera...")
    cap = open_camera(config.CAMERA_INDEX, "Cam")

    # Flush buffers
    for _ in range(5):
        cap.grab()

    # Stream size override — only resize if explicitly configured
    _sw = getattr(config, "STREAM_WIDTH", None)
    _sh = getattr(config, "STREAM_HEIGHT", None)
    stream_override = (_sw, _sh) if (_sw and _sh) else None

    # Start capture thread
    capture_thread = threading.Thread(
        target=_capture_loop, args=(cap, args.quality, stream_override), daemon=True
    )
    capture_thread.start()

    print(f"[HTTP Streamer] MJPEG stream ready at http://0.0.0.0:{args.port}/stream")
    print("[HTTP Streamer] Use cloudflared to tunnel this port")

    # Run Flask server
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
