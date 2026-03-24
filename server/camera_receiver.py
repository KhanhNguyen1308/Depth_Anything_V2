"""
Remote Camera Receiver - Receives frames from Jetson over TCP.
Provides read() interface for single camera mode.
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
    Receives frames from camera_streamer.py over TCP.
    Single camera mode — provides read() for depth estimation.
    """

    def __init__(self, host="0.0.0.0", port=None):
        self._host = host
        self._port = port or config.STREAM_PORT
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False
        self._server_sock = None

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
                    header = self._recv_exact(client_sock, 4)
                    size, = struct.unpack("!I", header)

                    if size > 5_000_000:
                        print(f"[Receiver] Invalid frame size: {size}")
                        break

                    jpg = self._recv_exact(client_sock, size)
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        continue

                    with self._lock:
                        self._frame = frame

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

    def read(self):
        """Read the latest frame."""
        with self._lock:
            if self._frame is not None:
                return True, self._frame
            return False, None

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
