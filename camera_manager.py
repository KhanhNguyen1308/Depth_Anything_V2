"""
Camera Manager - Quản lý 2 camera USB OV9732 trên Jetson Nano
Sử dụng grab()/retrieve() để đồng bộ frame giữa 2 camera.
Hỗ trợ V4L2 và GStreamer pipeline.
"""

import os
import cv2
import numpy as np
import subprocess
import threading
import time

import config


def _gstreamer_pipeline(device_index, width, height, fps):
    """Tạo GStreamer pipeline cho camera USB trên Jetson Nano."""
    return (
        f"v4l2src device=/dev/video{device_index} ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false"
    )


def _configure_exposure(camera_index, name):
    """Configure OV9732 exposure via v4l2-ctl after camera is opened."""
    device = f'/dev/video{camera_index}'

    expo_auto = getattr(config, 'CAMERA_EXPOSURE_AUTO', 3)
    controls = [
        ('exposure_auto', expo_auto),
        ('white_balance_temperature_auto', 1),
    ]

    # Only set exposure_absolute if manual mode
    if expo_auto == 1:
        expo_val = getattr(config, 'CAMERA_EXPOSURE_VALUE', 300)
        controls.append(('exposure_absolute', expo_val))

    gain_val = getattr(config, 'CAMERA_GAIN', None)
    if gain_val is not None:
        controls.append(('gain', gain_val))

    for ctrl_name, ctrl_val in controls:
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', device, '--set-ctrl', f'{ctrl_name}={ctrl_val}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                print(f"[{name}] v4l2: {ctrl_name}={ctrl_val}")
            else:
                print(f"[{name}] v4l2: {ctrl_name} skipped: {result.stderr.strip()}")
        except FileNotFoundError:
            print(f"[{name}] v4l2-ctl not found, install: sudo apt install v4l-utils")
            break
        except Exception as e:
            print(f"[{name}] v4l2-ctl error for {ctrl_name}: {e}")


def _open_camera(camera_index, name):
    """Mở camera và cấu hình."""
    if config.USE_GSTREAMER:
        pipeline = _gstreamer_pipeline(
            camera_index,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.CAMERA_FPS,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        # MJPEG format cho tốc độ tốt hơn với USB camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Buffer size = 1 để luôn lấy frame mới nhất
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            f"[{name}] Không thể mở camera index {camera_index}. "
            f"Kiểm tra kết nối USB và chạy 'ls /dev/video*'"
        )

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[{name}] Opened: index={camera_index}, "
          f"{actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    # Configure exposure AFTER camera is opened (device must be active)
    _configure_exposure(camera_index, name)

    return cap


class DualCameraManager:
    """
    Quản lý 2 camera USB OV9732 với đồng bộ grab()/retrieve().

    grab() chỉ lấy frame từ buffer (rất nhanh, ~0.1ms),
    retrieve() decode frame (chậm hơn, ~2-5ms).

    Gọi grab() trên cả 2 camera trước, rồi retrieve() sau,
    giúp giảm độ lệch thời gian giữa 2 frame xuống < 1ms
    (so với ~5-10ms nếu dùng read() tuần tự).
    """

    def __init__(self):
        self.cap0 = None
        self.cap1 = None
        self.frame0 = None
        self.frame1 = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        # Stereo rectification maps (tính 1 lần khi start)
        self._map_l1 = None
        self._map_l2 = None
        self._map_r1 = None
        self._map_r2 = None
        self._rectified = False

    def _init_rectification(self):
        """Tính stereo rectification maps từ calibration file.
        
        map_l applies to cam0 (frame0), map_r applies to cam1 (frame1).
        """
        calib_file = getattr(config, "STEREO_CALIB_FILE", "calibration.npz")
        if not os.path.exists(calib_file):
            print(f"[CameraManager] Calibration file not found: {calib_file}")
            print("[CameraManager] Running without rectification")
            return

        data = np.load(calib_file)
        mtx_l = data["mtx_l"].copy()
        dist_l = data["dist_l"]
        mtx_r = data["mtx_r"].copy()
        dist_r = data["dist_r"]
        R = data["R"]
        T = data["T"]

        img_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        # Calibration thường ở 640×480 — scale intrinsics nếu resolution khác
        calib_w = getattr(config, "STEREO_CALIB_WIDTH", 640)
        calib_h = getattr(config, "STEREO_CALIB_HEIGHT", 480)
        sx = config.CAMERA_WIDTH / calib_w
        sy = config.CAMERA_HEIGHT / calib_h
        if sx != 1.0 or sy != 1.0:
            mtx_l[0, :] *= sx  # fx, cx
            mtx_l[1, :] *= sy  # fy, cy
            mtx_r[0, :] *= sx
            mtx_r[1, :] *= sy
            print(f"[CameraManager] Scaled intrinsics: {calib_w}x{calib_h} → {img_size[0]}x{img_size[1]}")

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
            alpha=0,  # Crop to valid pixels only
        )

        self._map_l1, self._map_l2 = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2,
        )
        self._map_r1, self._map_r2 = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2,
        )

        self._rectified = True
        baseline = np.linalg.norm(T)
        print(f"[CameraManager] Stereo rectification initialized")
        print(f"  Baseline: {baseline*100:.1f}cm | Image: {img_size[0]}x{img_size[1]}")

    def _rectify(self, frame0, frame1):
        """Apply rectification maps: map_l→frame0, map_r→frame1."""
        if not self._rectified:
            return frame0, frame1
        rect0 = cv2.remap(frame0, self._map_l1, self._map_l2, cv2.INTER_LINEAR)
        rect1 = cv2.remap(frame1, self._map_r1, self._map_r2, cv2.INTER_LINEAR)
        return rect0, rect1

    def start(self):
        """Khởi động cả 2 camera."""
        print("[CameraManager] Starting dual cameras (grab/retrieve sync)...")
        self.cap0 = _open_camera(config.CAMERA_0_INDEX, "Camera-0")
        self.cap1 = _open_camera(config.CAMERA_1_INDEX, "Camera-1")

        # Tính rectification maps 1 lần
        self._init_rectification()

        # Flush buffer cũ - grab vài frame để clear
        for _ in range(5):
            self.cap0.grab()
            self.cap1.grab()

        self.running = True
        self.thread = threading.Thread(target=self._sync_capture_loop, daemon=True)
        self.thread.start()

        time.sleep(0.5)
        print("[CameraManager] Both cameras ready (synced)")

    def _sync_capture_loop(self):
        """
        Loop capture đồng bộ:
          1. grab() cả 2 camera (nhanh, chỉ lấy từ buffer)
          2. retrieve() cả 2 camera (decode MJPEG -> BGR)
        Đảm bảo 2 frame gần nhau nhất về mặt thời gian.
        """
        while self.running:
            # --- GRAB cả 2 (đồng bộ) ---
            ok0 = self.cap0.grab()
            ok1 = self.cap1.grab()

            if not (ok0 and ok1):
                time.sleep(0.001)
                continue

            # --- RETRIEVE cả 2 (decode) ---
            ret0, frame0 = self.cap0.retrieve()
            ret1, frame1 = self.cap1.retrieve()

            if ret0 and ret1:
                # Apply stereo rectification
                frame0, frame1 = self._rectify(frame0, frame1)
                with self.lock:
                    self.frame0 = frame0
                    self.frame1 = frame1

    def read_both(self):
        """
        Đọc cặp frame đồng bộ từ 2 camera (thread-safe).

        Returns:
            (success, frame0, frame1)
        """
        with self.lock:
            if self.frame0 is not None and self.frame1 is not None:
                return True, self.frame0, self.frame1
            return False, None, None

    def read_cam(self, cam_id):
        """
        Đọc frame từ 1 camera (thread-safe).

        Args:
            cam_id: 0 hoặc 1

        Returns:
            (success, frame)
        """
        with self.lock:
            frame = self.frame0 if cam_id == 0 else self.frame1
            if frame is not None:
                return True, frame
            return False, None

    def stop(self):
        """Dừng cả 2 camera."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.cap0 is not None:
            self.cap0.release()
        if self.cap1 is not None:
            self.cap1.release()
        print("[CameraManager] All cameras stopped")
