"""
Camera Streamer - Runs on Jetson Nano.
Captures frames from a single USB camera and streams to a remote server.
All depth processing happens on the server side.

Usage:
    python3 camera_streamer.py
    python3 camera_streamer.py --server 192.168.2.10 --port 9000
"""

import argparse
import cv2
import numpy as np
import signal
import socket
import struct
import subprocess
import sys
import time

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
              f"→ {sw}x{sh}")
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


def send_frame(sock, frame, quality):
    """
    Send a single frame over TCP.
    Protocol: [4 bytes: size][jpeg_data]
    """
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, jpg = cv2.imencode(".jpg", frame, encode_param)
    data = jpg.tobytes()
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def main():
    parser = argparse.ArgumentParser(description="Camera Streamer for Jetson Nano")
    parser.add_argument("--server", default=config.SERVER_HOST, help="Server IP")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="Server port")
    parser.add_argument("--quality", type=int, default=config.JPEG_QUALITY, help="JPEG quality")
    args = parser.parse_args()

    print("=" * 50)
    print("  Camera Streamer - Jetson Nano")
    print("=" * 50)
    print(f"  Server: {args.server}:{args.port}")
    print(f"  Camera: {config.CAMERA_INDEX}")
    print(f"  Config: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ {config.CAMERA_FPS}fps")
    print(f"  JPEG quality: {args.quality}")
    print("=" * 50)

    # Open camera
    print("\n[Streamer] Opening camera...")
    cap = open_camera(config.CAMERA_INDEX, "Cam")

    # Flush buffers
    for _ in range(5):
        cap.grab()

    # Capture one real frame to determine actual frame dimensions from the driver
    ret_test, test_frame = cap.read()
    if not ret_test or test_frame is None:
        raise RuntimeError("[Streamer] Cannot read initial frame — check camera index/resolution")
    actual_h, actual_w = test_frame.shape[:2]
    print(f"  [Cam] Actual frame size from driver: {actual_w}x{actual_h}")

    # Stream size override (only if explicitly set in config)
    _sw = getattr(config, "STREAM_WIDTH", None)
    _sh = getattr(config, "STREAM_HEIGHT", None)
    stream_override = (_sw, _sh) if (_sw and _sh) else None
    if stream_override:
        print(f"  [Cam] Stream override: {stream_override[0]}x{stream_override[1]}")

    # Handshake dimensions = stream override (if set) else actual frame dims
    handshake_w = stream_override[0] if stream_override else actual_w
    handshake_h = stream_override[1] if stream_override else actual_h

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[Streamer] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Streamer] Camera ready. Connecting to server...")

    frame_count = 0
    fps_time = time.time()
    reconnect_delay = 3

    while running:
        # Connect to server
        sock = None
        while running and sock is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(5)
                sock.connect((args.server, args.port))
                info = struct.pack("!HH", handshake_w, handshake_h)
                sock.sendall(info)
                print(f"[Streamer] Connected to {args.server}:{args.port} ({handshake_w}x{handshake_h})")
            except (ConnectionRefusedError, OSError, socket.timeout) as e:
                sock = None
                print(f"[Streamer] Cannot connect: {e}. Retrying in {reconnect_delay}s...")
                time.sleep(reconnect_delay)

        # Stream loop
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # Resize only when an explicit stream override is configured
            if stream_override and (frame.shape[1] != stream_override[0] or frame.shape[0] != stream_override[1]):
                frame = cv2.resize(frame, stream_override, interpolation=cv2.INTER_AREA)

            try:
                send_frame(sock, frame, args.quality)
                frame_count += 1

                now = time.time()
                if now - fps_time >= 5.0:
                    fps = frame_count / (now - fps_time)
                    print(f"[Streamer] Streaming at {fps:.1f} FPS")
                    frame_count = 0
                    fps_time = now

            except (BrokenPipeError, ConnectionResetError, OSError):
                print("[Streamer] Connection lost. Reconnecting...")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                break

    # Cleanup
    cap.release()
    if sock:
        sock.close()
    print("[Streamer] Done.")


if __name__ == "__main__":
    main()
