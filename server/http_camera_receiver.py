"""
HTTP Camera Receiver - Reads MJPEG stream from cloudflared tunnel.
Provides the same read() interface as RemoteCameraReceiver.
"""

import cv2
import threading
import time

import config


class HttpCameraReceiver:
    """
    Receives frames via MJPEG HTTP stream from a cloudflared tunnel.
    Single camera mode — provides read() for depth estimation.
    """

    def __init__(self, stream_url=None):
        self._stream_url = stream_url or (config.TUNNEL_STREAM_URL.rstrip("/") + "/stream")
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False

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
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[HTTP Receiver] Stream read failed. Reconnecting...")
                        break

                    with self._lock:
                        self._frame = frame

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

    def read(self):
        with self._lock:
            if self._frame is not None:
                return True, self._frame
            return False, None

    def is_connected(self):
        return self._connected

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[HTTP Receiver] Stopped")
