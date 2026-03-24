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
        pipeline = (
            f"v4l2src device=/dev/video{index} ! "
            f"image/jpeg, width={config.CAMERA_WIDTH}, height={config.CAMERA_HEIGHT}, "
            f"framerate={config.CAMERA_FPS}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
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
    print(f"  Resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ {config.CAMERA_FPS}fps")
    print(f"  JPEG quality: {args.quality}")
    print("=" * 50)

    # Open camera
    print("\n[Streamer] Opening camera...")
    cap = open_camera(config.CAMERA_INDEX, "Cam")

    # Flush buffers
    for _ in range(5):
        cap.grab()

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
                info = struct.pack("!HH", config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                sock.sendall(info)
                print(f"[Streamer] Connected to {args.server}:{args.port}")
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
