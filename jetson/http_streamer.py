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
    if config.USE_GSTREAMER:
        sw = getattr(config, "STREAM_WIDTH", None)
        sh = getattr(config, "STREAM_HEIGHT", None)
        if sw and sh:
            # Decode MJPEG and scale inside GStreamer — Python receives 1080p frames directly.
            # This avoids allocating a ~24MB raw 4K frame in Python memory.
            scale = f"videoscale ! video/x-raw, width={sw}, height={sh}, format=BGR ! "
        else:
            scale = "video/x-raw, format=BGR ! "
        pipeline = (
            f"v4l2src device=/dev/video{index} ! "
            f"image/jpeg, width={config.CAMERA_WIDTH}, height={config.CAMERA_HEIGHT}, "
            f"framerate={config.CAMERA_FPS}/1 ! "
            f"jpegdec ! videoconvert ! {scale}"
            f"appsink drop=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
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
    print(f"  [{name}] Opened: index={index}, {w}x{h} @ {fps:.0f}fps")

    if w != config.CAMERA_WIDTH or h != config.CAMERA_HEIGHT:
        print(f"  [{name}] WARNING: Requested {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} "
              f"but got {w}x{h}. Update config to match actual camera output.")

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
