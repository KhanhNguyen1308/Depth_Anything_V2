"""
Depth Anything V2 - Depth Estimator cho Jetson Nano
Sử dụng model DepthAnythingV2 với encoder DINOv2 (ViT-S hoặc ViT-B)
"""

import os
import cv2
import numpy as np
import torch
import time
import threading

import config


class DepthAnythingV2Estimator:
    """Depth estimation sử dụng Depth Anything V2."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        """Tải model Depth Anything V2."""
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        }

        encoder = config.MODEL_ENCODER
        if encoder not in model_configs:
            raise ValueError(f"Encoder '{encoder}' không hỗ trợ. Chọn 'vits' hoặc 'vitb'.")

        model_cfg = model_configs[encoder]
        self.model = DepthAnythingV2(**model_cfg)

        # Tải checkpoint
        ckpt_path = os.path.join(config.MODEL_DIR, f"depth_anything_v2_{encoder}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {ckpt_path}\n"
                f"Chạy 'python3 download_model.py' để tải model."
            )

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

        # FP16 để tăng tốc trên Jetson
        if config.USE_FP16 and self.device.type == "cuda":
            self.model = self.model.half()

        print(f"[DepthEstimator] Model loaded: {encoder} on {self.device}")
        if config.USE_FP16:
            print("[DepthEstimator] FP16 enabled")

    @torch.no_grad()
    def estimate(self, frame):
        """
        Tạo depth map từ một frame BGR.
        
        Args:
            frame: numpy array BGR (H, W, 3)
            
        Returns:
            depth_colormap: numpy array BGR depth colormap (H, W, 3)
            depth_raw: numpy array float32 normalized [0, 1] (H, W)
        """
        with self.lock:
            h, w = frame.shape[:2]

            # Preprocess: BGR -> RGB, resize, normalize
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE),
                             interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            if config.USE_FP16 and self.device.type == "cuda":
                img = img.half()

            # Inference
            depth = self.model(img)

            # Post-process
            depth = depth.squeeze().cpu().float().numpy()

            # Resize về kích thước gốc
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

            # Normalize về [0, 255]
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max - depth_min > 1e-6:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth)

            depth_colormap = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO
            )

            return depth_colormap, depth_normalized


def create_estimator():
    """Factory: tạo estimator theo config backend."""
    backend = getattr(config, "INFERENCE_BACKEND", "pytorch").lower()

    if backend == "onnxrt":
        try:
            from onnxrt_depth_estimator import OnnxRTDepthEstimator
            print("[Backend] Using ONNX Runtime (TRT/CUDA EP)")
            return OnnxRTDepthEstimator()
        except Exception as e:
            print(f"[Backend] ONNX Runtime failed: {e}")
            print("[Backend] Trying raw TensorRT...")
            backend = "tensorrt"

    if backend == "tensorrt":
        try:
            from trt_depth_estimator import TensorRTDepthEstimator
            print("[Backend] Using TensorRT")
            return TensorRTDepthEstimator()
        except Exception as e:
            print(f"[Backend] TensorRT failed: {e}")
            print("[Backend] Falling back to PyTorch...")

    print("[Backend] Using PyTorch")
    return DepthAnythingV2Estimator()


class DualCameraDepthEstimator:
    """
    Xử lý depth estimation cho 2 camera xen kẽ.

    1 inference thread luân phiên: cam0 → cam1 → cam0 → cam1 → ...
    Kết hợp stereo SGBM + mono depth để tính khoảng cách tuyệt đối (mét).
    """

    def __init__(self):
        self.estimator = create_estimator()
        self._cam_rgb = [None, None]
        self._cam_depth = [None, None]
        self._combined = None
        self._fps = 0.0
        self._depth_info = {"min_dist": 0, "max_dist": 0, "center_dist": 0}
        self._lock = threading.Lock()
        self._camera_mgr = None
        self._running = False
        self._thread = None
        # Stereo SGBM
        self._sgbm = None
        self._focal_length = 0.0
        self._baseline = 0.0
        self._cam0_is_right = False  # Auto-detected from T[0] sign
        self._depth_history = []  # Rolling window cho stable measurement
        self._stereo_frame_count = 0
        self._stereo_skip = 3  # Only compute stereo every N frames
        self._cached_depth_m = None  # Reuse last stereo result between skips
        self._init_stereo()

    def _init_stereo(self):
        """Khởi tạo stereo SGBM và đọc calibration params."""
        calib_file = getattr(config, "STEREO_CALIB_FILE", "calibration.npz")
        if not os.path.exists(calib_file):
            print("[DualDepth] No calibration file — stereo depth disabled")
            return

        data = np.load(calib_file)
        mtx_l = data["mtx_l"].copy()
        dist_l = data["dist_l"]
        mtx_r = data["mtx_r"].copy()
        dist_r = data["dist_r"]
        R = data["R"]
        T = data["T"]
        self._baseline = float(np.linalg.norm(T))

        # Auto-detect physical L/R from T[0]
        # In stereoCalibrate: P_cam1 = R * P_cam0 + T
        # T[0] < 0 → cam1 is to the RIGHT of cam0 → cam0=LEFT
        # T[0] > 0 → cam1 is to the LEFT of cam0 → cam0=RIGHT
        self._cam0_is_right = (T[0, 0] > 0)
        side_info = ("USB0=RIGHT, USB1=LEFT" if self._cam0_is_right
                     else "USB0=LEFT, USB1=RIGHT")
        print(f"[DualDepth] T[0]={T[0,0]:.4f} → {side_info}")

        # Scale intrinsics theo resolution hiện tại
        calib_w = getattr(config, "STEREO_CALIB_WIDTH", 640)
        calib_h = getattr(config, "STEREO_CALIB_HEIGHT", 480)
        sx = config.CAMERA_WIDTH / calib_w
        sy = config.CAMERA_HEIGHT / calib_h
        if sx != 1.0 or sy != 1.0:
            mtx_l[0, :] *= sx
            mtx_l[1, :] *= sy
            mtx_r[0, :] *= sx
            mtx_r[1, :] *= sy

        img_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        # Dùng stereoRectify để lấy focal length ĐÚNG sau rectification
        _, _, P1, _, _, _, _ = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0,
        )
        self._focal_length = float(P1[0, 0])

        # numDisparities phải đủ lớn cho vật thể gần
        # min_depth = focal * baseline / numDisp
        # Với fx=720, B=0.145: numDisp=192 → min ~0.54m
        num_disp = 192
        block_size = 7
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        min_depth = self._focal_length * self._baseline / num_disp
        print(f"[DualDepth] Stereo depth enabled: "
              f"focal={self._focal_length:.1f}px (rectified), "
              f"baseline={self._baseline*100:.1f}cm, "
              f"numDisp={num_disp}, min_depth={min_depth:.2f}m")

    def _compute_stereo_depth(self, frame_l, frame_r):
        """Tính absolute depth (mét) từ stereo disparity.

        frame_l = rectified cam0, frame_r = rectified cam1.
        SGBM needs compute(physical_LEFT, physical_RIGHT).
        Auto-detected from T[0] sign in _init_stereo.
        """
        if self._sgbm is None:
            return None

        gray0 = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # SGBM: compute(physical_LEFT, physical_RIGHT)
        if self._cam0_is_right:
            # USB0=RIGHT, USB1=LEFT → compute(gray1, gray0)
            disp = self._sgbm.compute(gray1, gray0).astype(np.float32) / 16.0
        else:
            # USB0=LEFT, USB1=RIGHT → compute(gray0, gray1)
            disp = self._sgbm.compute(gray0, gray1).astype(np.float32) / 16.0

        # Z = f * B / d  (chỉ ở vùng disparity > 0)
        valid = disp > 1.0
        depth_m = np.zeros_like(disp)
        depth_m[valid] = (self._focal_length * self._baseline) / disp[valid]

        # Clamp range hợp lý (0.1m – 10m)
        depth_m = np.clip(depth_m, 0, 10.0)
        depth_m[~valid] = 0

        return depth_m

    def _overlay_distance(self, depth_color, depth_m):
        """Vẽ thông tin khoảng cách lên depth colormap."""
        if depth_m is None:
            return depth_color

        h, w = depth_m.shape
        result = depth_color

        # Khoảng cách vùng trung tâm (20% giữa)
        ch, cw = h // 2, w // 2
        rh, rw = h // 5, w // 5
        center_region = depth_m[ch - rh:ch + rh, cw - rw:cw + rw]
        valid_center = center_region[center_region > 0.1]

        if len(valid_center) > 10:
            center_dist = float(np.median(valid_center))
            min_dist = float(np.min(valid_center))
            max_dist = float(np.max(valid_center))

            # Rolling median: giữ 15 mẫu gần nhất, lọc outliers
            self._depth_history.append(center_dist)
            if len(self._depth_history) > 15:
                self._depth_history.pop(0)

            # Dùng median của history thay vì single-frame value
            stable_dist = float(np.median(self._depth_history))

            # Cập nhật depth info cho API
            self._depth_info = {
                "min_dist": round(min_dist, 2),
                "max_dist": round(max_dist, 2),
                "center_dist": round(stable_dist, 2),
            }

            # Vẽ crosshair + distance text (dùng stable value)
            cv2.rectangle(result, (cw - rw, ch - rh), (cw + rw, ch + rh),
                          (0, 255, 0), 1)
            cv2.drawMarker(result, (cw, ch), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 1)

            text = f"{stable_dist:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx, ty = cw - tw // 2, ch - rh - 10
            cv2.rectangle(result, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                          (0, 0, 0), -1)
            cv2.putText(result, text, (tx, ty), font, font_scale,
                        (0, 255, 0), thickness)

            # Min/max nhỏ ở góc
            info = f"Min:{min_dist:.2f}m Max:{max_dist:.2f}m"
            cv2.putText(result, info, (8, h - 10), font, 0.45,
                        (200, 200, 200), 1)

        return result

    def start(self, camera_mgr):
        """Khởi động inference thread xen kẽ."""
        self._camera_mgr = camera_mgr
        self._running = True

        self._thread = threading.Thread(
            target=self._alternating_loop, daemon=True,
            name="inference-alternating",
        )
        self._thread.start()
        print("[DualDepth] Alternating pipeline started (cam0→cam1→cam0→...)")

    def _alternating_loop(self):
        """Luân phiên inference giữa 2 camera."""
        cam_id = 0
        # Cache cặp frame đồng bộ cho stereo
        self._stereo_pair = (None, None)

        while self._running:
            # Lấy cặp frame đồng bộ (cùng lock → cùng thời điểm grab)
            ok_both, frame_l, frame_r = self._camera_mgr.read_both()
            if not ok_both:
                time.sleep(0.001)
                continue

            frame = frame_l if cam_id == 0 else frame_r

            t0 = time.time()
            depth_color, depth_raw = self.estimator.estimate(frame)

            # Stereo depth: tính khi xử lý cam0, skip frames để tăng FPS
            if cam_id == 0 and self._sgbm is not None:
                self._stereo_frame_count += 1
                if self._stereo_frame_count >= self._stereo_skip:
                    self._stereo_frame_count = 0
                    self._cached_depth_m = self._compute_stereo_depth(frame_l, frame_r)
                depth_color = self._overlay_distance(depth_color, self._cached_depth_m)

            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 1e-6)

            with self._lock:
                self._cam_rgb[cam_id] = frame
                self._cam_depth[cam_id] = depth_color
                self._fps = fps
                self._update_combined()

            # Chuyển camera
            cam_id = 1 - cam_id

    def _update_combined(self):
        """Build combined view: blended RGB (both cams) + depth with distance."""
        if self._cam_rgb[0] is None or self._cam_rgb[1] is None:
            return
        if self._cam_depth[0] is None:
            return

        h = min(self._cam_rgb[0].shape[0], self._cam_rgb[1].shape[0])
        w = min(self._cam_rgb[0].shape[1], self._cam_rgb[1].shape[1])

        f0 = cv2.resize(self._cam_rgb[0], (w, h))
        f1 = cv2.resize(self._cam_rgb[1], (w, h))
        # Blend both cameras 50/50 so center represents midpoint
        blended = cv2.addWeighted(f0, 0.5, f1, 0.5, 0)

        d0 = cv2.resize(self._cam_depth[0], (w, h))

        self._combined = np.hstack([blended, d0])

    def process_frames(self, frame0, frame1):
        """Chế độ đồng bộ (backward compat cho test_trt.py)."""
        t0 = time.time()

        depth_color0, depth_raw0 = self.estimator.estimate(frame0)
        depth_color1, depth_raw1 = self.estimator.estimate(frame1)

        h = min(frame0.shape[0], frame1.shape[0])
        w = min(frame0.shape[1], frame1.shape[1])

        f0 = cv2.resize(frame0, (w, h))
        d0 = cv2.resize(depth_color0, (w, h))
        f1 = cv2.resize(frame1, (w, h))
        d1 = cv2.resize(depth_color1, (w, h))

        top_row = np.hstack([f0, d0])
        bottom_row = np.hstack([f1, d1])
        combined = np.vstack([top_row, bottom_row])

        elapsed = time.time() - t0
        fps = 1.0 / max(elapsed, 1e-6)

        with self._lock:
            self._cam_rgb[0] = frame0
            self._cam_depth[0] = depth_color0
            self._cam_rgb[1] = frame1
            self._cam_depth[1] = depth_color1
            self._combined = combined
            self._fps = fps

    def get_results(self):
        """Lấy kết quả mới nhất (thread-safe)."""
        with self._lock:
            return {
                "cam0_rgb": self._cam_rgb[0],
                "cam0_depth": self._cam_depth[0],
                "cam1_rgb": self._cam_rgb[1],
                "cam1_depth": self._cam_depth[1],
                "combined": self._combined,
                "fps": self._fps,
                "depth_info": dict(self._depth_info),
            }

    def stop(self):
        """Dừng inference thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        print("[DualDepth] Pipeline stopped")
