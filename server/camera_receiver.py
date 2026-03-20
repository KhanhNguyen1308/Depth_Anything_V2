"""
Remote Camera Receiver - Receives synchronized frame pairs from Jetson over TCP.
Provides the same read_both() interface as DualCameraManager.
"""

import cv2
import numpy as np
import os
import socket
import struct
import threading
import time

import config


class RemoteCameraReceiver:
    """
    Receives synchronized frame pairs from camera_streamer.py over TCP.
    Provides read_both() compatible with DualCameraDepthEstimator.
    """

    def __init__(self, host="0.0.0.0", port=None):
        self._host = host
        self._port = port or config.STREAM_PORT
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
            print(f"[Receiver] No calibration file: {calib_file}")
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

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
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
        if not self._rectified:
            return frame0, frame1
        rect0 = cv2.remap(frame0, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
        rect1 = cv2.remap(frame1, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        return rect0, rect1

    def _recv_exact(self, sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf.extend(chunk)
        return bytes(buf)

    def start(self):
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
        while self._running:
            client_sock = None
            while self._running and client_sock is None:
                try:
                    client_sock, addr = self._server_sock.accept()
                    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

            frame_count = 0
            fps_time = time.time()

            while self._running:
                try:
                    header = self._recv_exact(client_sock, 8)
                    size0, size1 = struct.unpack("!II", header)

                    if size0 > 2_000_000 or size1 > 2_000_000:
                        print(f"[Receiver] Invalid frame sizes: {size0}, {size1}")
                        break

                    jpg0 = self._recv_exact(client_sock, size0)
                    jpg1 = self._recv_exact(client_sock, size1)

                    frame0 = cv2.imdecode(
                        np.frombuffer(jpg0, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    frame1 = cv2.imdecode(
                        np.frombuffer(jpg1, dtype=np.uint8), cv2.IMREAD_COLOR
                    )

                    if frame0 is None or frame1 is None:
                        continue

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

            self._connected = False
            if client_sock:
                try:
                    client_sock.close()
                except Exception:
                    pass
            print("[Receiver] Waiting for reconnection...")

    def read_both(self):
        with self._lock:
            if self._frame0 is not None and self._frame1 is not None:
                return True, self._frame0, self._frame1
            return False, None, None

    def is_connected(self):
        return self._connected

    def stop(self):
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)
        print("[Receiver] Stopped")
