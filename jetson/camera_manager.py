"""
Camera Manager - Manages 2 USB cameras with grab/retrieve sync.
Used for calibration on the Jetson side.
"""

import cv2
import numpy as np
import os
import subprocess
import threading
import time

import config


def _gstreamer_pipeline(device_index, width, height, fps):
    return (
        f"v4l2src device=/dev/video{device_index} ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false"
    )


def _configure_exposure(camera_index, name):
    device = f'/dev/video{camera_index}'
    controls = [
        ('exposure_auto', getattr(config, 'CAMERA_EXPOSURE_AUTO', 3)),
        ('white_balance_temperature_auto', 1),
    ]
    if getattr(config, 'CAMERA_EXPOSURE_AUTO', 3) == 1:
        controls.append(('exposure_absolute', getattr(config, 'CAMERA_EXPOSURE_VALUE', 300)))
    gain = getattr(config, 'CAMERA_GAIN', None)
    if gain is not None:
        controls.append(('gain', gain))

    for ctrl_name, ctrl_val in controls:
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', device, '--set-ctrl', f'{ctrl_name}={ctrl_val}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                print(f"[{name}] v4l2: {ctrl_name}={ctrl_val}")
        except Exception:
            pass


def _open_camera(camera_index, name):
    if config.USE_GSTREAMER:
        pipeline = _gstreamer_pipeline(
            camera_index, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"[{name}] Cannot open camera index {camera_index}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{name}] Opened: index={camera_index}, {w}x{h} @ {fps:.0f}fps")
    _configure_exposure(camera_index, name)
    return cap


class DualCameraManager:
    def __init__(self):
        self.cap0 = None
        self.cap1 = None
        self.frame0 = None
        self.frame1 = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        print("[CameraManager] Starting dual cameras...")
        self.cap0 = _open_camera(config.CAMERA_0_INDEX, "Camera-0")
        self.cap1 = _open_camera(config.CAMERA_1_INDEX, "Camera-1")

        for _ in range(5):
            self.cap0.grab()
            self.cap1.grab()

        self.running = True
        self.thread = threading.Thread(target=self._sync_capture_loop, daemon=True)
        self.thread.start()
        time.sleep(0.5)
        print("[CameraManager] Both cameras ready")

    def _sync_capture_loop(self):
        while self.running:
            ok0 = self.cap0.grab()
            ok1 = self.cap1.grab()
            if not (ok0 and ok1):
                time.sleep(0.001)
                continue
            ret0, frame0 = self.cap0.retrieve()
            ret1, frame1 = self.cap1.retrieve()
            if ret0 and ret1:
                with self.lock:
                    self.frame0 = frame0
                    self.frame1 = frame1

    def read_both(self):
        with self.lock:
            if self.frame0 is not None and self.frame1 is not None:
                return True, self.frame0, self.frame1
            return False, None, None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.cap0:
            self.cap0.release()
        if self.cap1:
            self.cap1.release()
        print("[CameraManager] Stopped")
