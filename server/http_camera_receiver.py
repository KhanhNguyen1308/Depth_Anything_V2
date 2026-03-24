"""
HTTP Camera Receiver - Reads MJPEG stream from cloudflared tunnel.
Receives combined stereo frames and splits them back into left/right.
Provides the same read_both() interface as RemoteCameraReceiver.
"""

import cv2
import numpy as np
import os
import threading
import time

import config


class HttpCameraReceiver:
    """
    Receives stereo frame pairs via MJPEG HTTP stream from a cloudflared tunnel.
    The Jetson sends combined side-by-side frames [cam0 | cam1].
    This receiver splits them back into individual frames.
    """

    def __init__(self, stream_url=None):
        self._stream_url = stream_url or (config.TUNNEL_STREAM_URL.rstrip("/") + "/stream")
        self._frame0 = None
        self._frame1 = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False
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
            print(f"[HTTP Receiver] No calibration file: {calib_file}")
            print("[HTTP Receiver] Running without rectification")
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
        print(f"[HTTP Receiver] Stereo rectification initialized")
        print(f"  Baseline: {baseline*100:.1f}cm | Image: {img_size[0]}x{img_size[1]}")

    def _rectify(self, frame0, frame1):
        if not self._rectified:
            return frame0, frame1
        rect0 = cv2.remap(frame0, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
        rect1 = cv2.remap(frame1, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        return rect0, rect1

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        print(f"[HTTP Receiver] Reading stream from {self._stream_url}")

    def _receive_loop(self):
        reconnect_delay = 3

        while self._running:
            cap = None
            try:
                print(f"[HTTP Receiver] Connecting to {self._stream_url}...")
                cap = cv2.VideoCapture(self._stream_url)

                if not cap.isOpened():
                    print(f"[HTTP Receiver] Cannot open stream. Retrying in {reconnect_delay}s...")
                    time.sleep(reconnect_delay)
                    continue

                self._connected = True
                print("[HTTP Receiver] Stream connected")
                frame_count = 0
                fps_time = time.time()

                while self._running:
                    ret, combined = cap.read()
                    if not ret or combined is None:
                        print("[HTTP Receiver] Stream read failed. Reconnecting...")
                        break

                    # Split combined side-by-side frame into left and right
                    h, w = combined.shape[:2]
                    mid = w // 2
                    frame0 = combined[:, :mid]
                    frame1 = combined[:, mid:]

                    frame0, frame1 = self._rectify(frame0, frame1)

                    with self._lock:
                        self._frame0 = frame0
                        self._frame1 = frame1

                    frame_count += 1
                    now = time.time()
                    if now - fps_time >= 5.0:
                        fps = frame_count / (now - fps_time)
                        print(f"[HTTP Receiver] Receiving at {fps:.1f} FPS")
                        frame_count = 0
                        fps_time = now

            except Exception as e:
                print(f"[HTTP Receiver] Error: {e}")
            finally:
                self._connected = False
                if cap is not None:
                    cap.release()

            if self._running:
                print(f"[HTTP Receiver] Reconnecting in {reconnect_delay}s...")
                time.sleep(reconnect_delay)

    def read_both(self):
        with self._lock:
            if self._frame0 is not None and self._frame1 is not None:
                return True, self._frame0, self._frame1
            return False, None, None

    def is_connected(self):
        return self._connected

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[HTTP Receiver] Stopped")
