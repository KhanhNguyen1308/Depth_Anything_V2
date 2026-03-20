"""
Server - Receives camera streams from Jetson and runs all processing locally.
Receives synchronized frame pairs via TCP, applies rectification, runs depth
estimation + stereo SGBM, and serves the web UI.

Usage:
    python3 server.py                           # listen on 0.0.0.0:9000
    python3 server.py --port 9000 --web-port 5000

Requirements on server:
    - calibration.npz (copy from Jetson)
    - Depth model files (checkpoints/ or ONNX)
    - pip install flask flask-cors opencv-python numpy torch
"""

import argparse
import cv2
import numpy as np
import os
import signal
import socket
import struct
import sys
import threading
import time

import config
from depth_estimator import DualCameraDepthEstimator
from web_server import run_server, set_depth_processor


class RemoteCameraReceiver:
    """
    Receives synchronized frame pairs from camera_streamer.py over TCP.
    Provides the same read_both() interface as DualCameraManager.
    """

    def __init__(self, host="0.0.0.0", port=9000):
        self._host = host
        self._port = port
        self._frame0 = None
        self._frame1 = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False
        self._server_sock = None
        # Stereo rectification maps
        self._map_l1 = None
        self._map_l2 = None
        self._map_r1 = None
        self._map_r2 = None
        self._rectified = False
        self._init_rectification()

    def _init_rectification(self):
        """Compute stereo rectification maps from calibration file."""
        calib_file = getattr(config, "STEREO_CALIB_FILE", "calibration.npz")
        if not os.path.exists(calib_file):
            print(f"[Receiver] Calibration file not found: {calib_file}")
            print("[Receiver] Running without rectification")
            return

        data = np.load(calib_file)
        mtx_l = data["mtx_l"].copy()
        dist_l = data["dist_l"]
        mtx_r = data["mtx_r"].copy()
        dist_r = data["dist_r"]
        R = data["R"]
        T = data["T"]

        img_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        calib_w = getattr(config, "STEREO_CALIB_WIDTH", 640)
        calib_h = getattr(config, "STEREO_CALIB_HEIGHT", 480)
        sx = config.CAMERA_WIDTH / calib_w
        sy = config.CAMERA_HEIGHT / calib_h
        if sx != 1.0 or sy != 1.0:
            mtx_l[0, :] *= sx
            mtx_l[1, :] *= sy
            mtx_r[0, :] *= sx
            mtx_r[1, :] *= sy

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0,
        )

        self._map_l1, self._map_l2 = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2,
        )
        self._map_r1, self._map_r2 = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2,
        )

        self._rectified = True
        baseline = np.linalg.norm(T)
        print(f"[Receiver] Stereo rectification initialized")
        print(f"  Baseline: {baseline*100:.1f}cm | Image: {img_size[0]}x{img_size[1]}")

    def _rectify(self, frame0, frame1):
        """Apply rectification maps: map_l→frame0, map_r→frame1."""
        if not self._rectified:
            return frame0, frame1
        rect0 = cv2.remap(frame0, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
        rect1 = cv2.remap(frame1, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        return rect0, rect1

    def _recv_exact(self, sock, n):
        """Receive exactly n bytes from socket."""
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def start(self):
        """Start listening for camera connections."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.settimeout(1.0)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(1)

        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

        print(f"[Receiver] Listening on {self._host}:{self._port}")

    def _receive_loop(self):
        """Accept connections and receive frame pairs."""
        while self._running:
            # Wait for a connection
            client_sock = None
            while self._running and client_sock is None:
                try:
                    client_sock, addr = self._server_sock.accept()
                    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Read resolution info (first 4 bytes: width, height as uint16)
                    info = self._recv_exact(client_sock, 4)
                    cam_w, cam_h = struct.unpack("!HH", info)
                    print(f"[Receiver] Camera connected from {addr[0]}:{addr[1]} ({cam_w}x{cam_h})")
                    self._connected = True
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        time.sleep(0.1)
                    continue

            if not self._running:
                break

            # Receive frame pairs
            frame_count = 0
            fps_time = time.time()

            while self._running:
                try:
                    # Read header: two uint32 sizes
                    header = self._recv_exact(client_sock, 8)
                    size0, size1 = struct.unpack("!II", header)

                    # Sanity check sizes (max ~2MB per frame)
                    if size0 > 2_000_000 or size1 > 2_000_000:
                        print(f"[Receiver] Invalid frame sizes: {size0}, {size1}")
                        break

                    # Read JPEG data
                    jpg0 = self._recv_exact(client_sock, size0)
                    jpg1 = self._recv_exact(client_sock, size1)

                    # Decode
                    frame0 = cv2.imdecode(
                        np.frombuffer(jpg0, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    frame1 = cv2.imdecode(
                        np.frombuffer(jpg1, dtype=np.uint8), cv2.IMREAD_COLOR
                    )

                    if frame0 is None or frame1 is None:
                        continue

                    # Apply rectification
                    frame0, frame1 = self._rectify(frame0, frame1)

                    with self._lock:
                        self._frame0 = frame0
                        self._frame1 = frame1

                    frame_count += 1
                    now = time.time()
                    if now - fps_time >= 5.0:
                        fps = frame_count / (now - fps_time)
                        print(f"[Receiver] Receiving at {fps:.1f} FPS")
                        frame_count = 0
                        fps_time = now

                except (ConnectionError, OSError) as e:
                    print(f"[Receiver] Connection lost: {e}")
                    break

            # Connection ended, cleanup
            self._connected = False
            if client_sock:
                try:
                    client_sock.close()
                except Exception:
                    pass
            print("[Receiver] Waiting for reconnection...")

    def read_both(self):
        """Read synchronized frame pair (same interface as DualCameraManager)."""
        with self._lock:
            if self._frame0 is not None and self._frame1 is not None:
                return True, self._frame0, self._frame1
            return False, None, None

    def is_connected(self):
        return self._connected

    def stop(self):
        """Stop receiver."""
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)
        print("[Receiver] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Depth Server - receives camera streams and processes")
    parser.add_argument("--port", type=int, default=9000, help="TCP port for camera stream")
    parser.add_argument("--web-port", type=int, default=5000, help="Web UI port")
    args = parser.parse_args()

    print("=" * 60)
    print("  Depth Anything V2 - Remote Processing Server")
    print("=" * 60)
    print(f"  Camera stream port: {args.port}")
    print(f"  Web UI: http://0.0.0.0:{args.web_port}")
    print(f"  Model: {config.MODEL_ENCODER} | Input: {config.MODEL_INPUT_SIZE}px")
    print("=" * 60)

    # Initialize depth processor
    print("\n[Server] Initializing depth estimator...")
    depth_processor = DualCameraDepthEstimator()

    # Initialize remote camera receiver
    print("[Server] Starting camera receiver...")
    receiver = RemoteCameraReceiver(host="0.0.0.0", port=args.port)
    receiver.start()

    # Register processor with web server
    set_depth_processor(depth_processor)

    # Override web port if specified
    config.WEB_PORT = args.web_port

    # Start depth processing with remote camera source
    depth_processor.start(receiver)

    # Start web server in background thread
    web_thread = threading.Thread(target=run_server, daemon=True)
    web_thread.start()

    # Graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[Server] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Server] Ready. Waiting for camera connection from Jetson...")
    print(f"  On Jetson run: python3 camera_streamer.py --server <this-server-ip> --port {args.port}")

    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Server] Cleaning up...")
        depth_processor.stop()
        receiver.stop()
        print("[Server] Done.")


if __name__ == "__main__":
    main()
