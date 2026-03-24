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

# Lazy import — only loaded if YOLO_ENABLED is True
_object_detector = None


def _get_object_detector():
    """Lazy-load ObjectDetector if YOLO is enabled."""
    global _object_detector
    if _object_detector is None and getattr(config, "YOLO_ENABLED", False):
        from object_detector import ObjectDetector
        _object_detector = ObjectDetector()
    return _object_detector


class DepthAnythingV2Estimator:
    """Depth estimation sử dụng Depth Anything V2."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        """Tải model Depth Anything V2."""
        from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        encoder = config.MODEL_ENCODER
        if encoder not in model_configs:
            raise ValueError(f"Encoder '{encoder}' không hỗ trợ. Chọn 'vits', 'vitb', hoặc 'vitl'.")

        model_cfg = model_configs[encoder]

        # Metric depth support
        self.metric_depth = getattr(config, "METRIC_DEPTH", False)
        if self.metric_depth:
            max_depth = getattr(config, "MAX_DEPTH", 20)
            model_cfg["max_depth"] = max_depth
            dataset = getattr(config, "METRIC_DATASET", "hypersim")
            ckpt_name = f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
        else:
            ckpt_name = f"depth_anything_v2_{encoder}.pth"

        self.model = DepthAnythingV2(**model_cfg)

        # Tải checkpoint
        ckpt_path = os.path.join(config.MODEL_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {ckpt_path}\n"
                f"Download: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/tree/main"
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
        if self.metric_depth:
            print(f"[DepthEstimator] Metric depth enabled (max_depth={model_cfg['max_depth']}m)")

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

            if self.metric_depth:
                # Metric depth: model output is already in meters
                depth_meters = depth
                # Normalize for colormap visualization only
                depth_max = depth.max()
                if depth_max > 1e-6:
                    depth_vis = depth / depth_max
                else:
                    depth_vis = np.zeros_like(depth)
                depth_colormap = cv2.applyColorMap(
                    (depth_vis * 255).astype(np.uint8),
                    cv2.COLORMAP_INFERNO
                )
                return depth_colormap, depth_meters
            else:
                # Relative depth: normalize to [0, 1]
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


class SingleCameraDepthEstimator:
    """
    Depth estimation cho single camera.
    Sử dụng mono metric depth (Depth Anything V2) + YOLO object detection.
    """

    def __init__(self):
        self.estimator = create_estimator()
        self._cam_rgb = None
        self._cam_depth = None
        self._combined = None
        self._fps = 0.0
        self._depth_info = {"min_dist": 0, "max_dist": 0, "center_dist": 0}
        self._lock = threading.Lock()
        self._camera_mgr = None
        self._running = False
        self._thread = None
        # Object detection
        self._detector = _get_object_detector()
        self._detections = []
        self._cam_detection = None
        self._detect_skip = getattr(config, "YOLO_SKIP_FRAMES", 2)
        self._detect_frame_count = 0
        self._cached_detections = []
        # Depth history for stable measurement
        self._depth_history = []

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

    def _overlay_relative_depth(self, depth_color, depth_normalized):
        """Overlay relative depth info when stereo calibration is unavailable."""
        h, w = depth_normalized.shape
        result = depth_color

        ch, cw = h // 2, w // 2
        rh, rw = h // 5, w // 5
        center_region = depth_normalized[ch - rh:ch + rh, cw - rw:cw + rw]

        if center_region.size > 0:
            # Depth Anything: higher value = closer (inverted depth)
            center_val = float(np.median(center_region))
            min_val = float(np.min(depth_normalized[depth_normalized > 0.01])) if np.any(depth_normalized > 0.01) else 0
            max_val = float(np.max(depth_normalized))

            self._depth_info = {
                "center_dist": round(center_val, 3),
                "min_dist": round(min_val, 3),
                "max_dist": round(max_val, 3),
            }

            # Draw crosshair + relative depth value
            cv2.rectangle(result, (cw - rw, ch - rh), (cw + rw, ch + rh),
                          (0, 255, 0), 1)
            cv2.drawMarker(result, (cw, ch), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 1)

            text = f"{center_val:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th_t), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx, ty = cw - tw // 2, ch - rh - 10
            cv2.rectangle(result, (tx - 4, ty - th_t - 4), (tx + tw + 4, ty + 4),
                          (0, 0, 0), -1)
            cv2.putText(result, text, (tx, ty), font, font_scale,
                        (0, 255, 0), thickness)

            info = f"Near:{max_val:.2f} Far:{min_val:.2f} (relative)"
            cv2.putText(result, info, (8, h - 10), font, 0.45,
                        (200, 200, 200), 1)

        return result

    def start(self, camera_mgr):
        """Khởi động inference thread."""
        self._camera_mgr = camera_mgr
        self._running = True

        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True,
            name="inference-loop",
        )
        self._thread.start()
        print("[DepthProcessor] Single camera pipeline started")

    def _inference_loop(self):
        """Inference loop cho single camera."""
        while self._running:
            ok, frame = self._camera_mgr.read()
            if not ok:
                time.sleep(0.001)
                continue

            t0 = time.time()
            depth_color, depth_raw = self.estimator.estimate(frame)

            depth_m_for_detection = None

            if getattr(self.estimator, 'metric_depth', False):
                # Metric depth: depth_raw is in meters
                depth_color = self._overlay_distance(depth_color, depth_raw)
                depth_m_for_detection = depth_raw
            else:
                # Relative depth
                depth_color = self._overlay_relative_depth(depth_color, depth_raw)

            # Object detection
            det_frame = None
            if self._detector is not None:
                self._detect_frame_count += 1
                if self._detect_frame_count >= self._detect_skip:
                    self._detect_frame_count = 0
                    dets = self._detector.detect(frame)
                    has_metric = getattr(self.estimator, 'metric_depth', False)
                    depth_for_dist = depth_m_for_detection if depth_m_for_detection is not None else depth_raw
                    dets = self._detector.measure_distances(
                        dets, depth_for_dist, has_metric_depth=has_metric
                    )
                    self._cached_detections = dets

                det_frame = self._detector.draw_detections(frame, self._cached_detections)

            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 1e-6)

            with self._lock:
                self._cam_rgb = frame
                self._cam_depth = depth_color
                self._fps = fps
                if self._detector is not None:
                    self._cam_detection = det_frame
                    self._detections = list(self._cached_detections)
                self._update_combined()

    def _update_combined(self):
        """Build combined view: RGB + Depth side by side."""
        if self._cam_rgb is None or self._cam_depth is None:
            return

        h, w = self._cam_rgb.shape[:2]
        depth_resized = cv2.resize(self._cam_depth, (w, h))
        self._combined = np.hstack([self._cam_rgb, depth_resized])

    def get_results(self):
        """Lấy kết quả mới nhất (thread-safe)."""
        with self._lock:
            return {
                "rgb": self._cam_rgb,
                "depth": self._cam_depth,
                "cam_detection": self._cam_detection,
                "combined": self._combined,
                "fps": self._fps,
                "depth_info": dict(self._depth_info),
                "detections": list(self._detections),
            }

    def stop(self):
        """Dừng inference thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        print("[DepthProcessor] Pipeline stopped")
