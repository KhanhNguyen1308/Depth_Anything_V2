"""
Depth Anything V2 - Local Depth Estimator for Jetson Nano.
Runs depth estimation + object detection + web dashboard on-device.
"""

import os
import cv2
import numpy as np
import torch
import time
import threading
import subprocess
import sys

import config


# ============================================================
# Object Detector (inline to keep jetson_local self-contained)
# ============================================================

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

_DET_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (64, 255, 64), (255, 64, 64), (64, 64, 255),
]


class ObjectDetector:
    def __init__(self):
        self._model = None
        self._device = "cpu"
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        model_name = config.YOLO_MODEL
        model_path = os.path.join(config.MODEL_DIR, model_name)

        if os.path.exists(model_path):
            self._model = YOLO(model_path)
            print(f"[YOLO] Loaded from {model_path}")
        else:
            self._model = YOLO(model_name)
            print(f"[YOLO] Loaded: {model_name} (auto-download)")

        device = config.YOLO_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        print(f"[YOLO] Device: {self._device}")

    def detect(self, frame):
        results = self._model.predict(
            frame, conf=config.YOLO_CONFIDENCE, imgsz=config.YOLO_INPUT_SIZE,
            device=self._device, verbose=False,
        )
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}",
                    "confidence": float(box.conf[0]),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
        return detections

    def measure_distances(self, detections, depth_map, has_metric_depth=False):
        if depth_map is None or len(detections) == 0:
            return detections
        h, w = depth_map.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
            x2, y2 = max(0, min(x2, w-1)), max(0, min(y2, h-1))
            if x2 <= x1 or y2 <= y1:
                det["distance"] = 0
                det["distance_unit"] = "m" if has_metric_depth else ""
                continue
            cx, cy = (x1+x2)//2, (y1+y2)//2
            rx, ry = int((x2-x1)*0.3), int((y2-y1)*0.3)
            roi = depth_map[max(0,cy-ry):min(h,cy+ry), max(0,cx-rx):min(w,cx+rx)]
            if roi.size == 0:
                det["distance"] = 0
                det["distance_unit"] = "m" if has_metric_depth else ""
                continue
            if has_metric_depth:
                valid = roi[roi > 0.1]
                det["distance"] = round(float(np.median(valid)), 2) if len(valid) > 0 else 0
                det["distance_unit"] = "m"
            else:
                det["distance"] = round(float(np.median(roi)), 3)
                det["distance_unit"] = ""
        return detections

    def draw_detections(self, frame, detections):
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = _DET_COLORS[det["class_id"] % len(_DET_COLORS)]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            dist = det.get("distance", 0)
            unit = det.get("distance_unit", "")
            if dist > 0 and unit:
                label = f"{det['class_name']} {det['confidence']:.0%} | {dist:.2f}{unit}"
            elif dist > 0:
                label = f"{det['class_name']} {det['confidence']:.0%} | {dist:.3f}"
            else:
                label = f"{det['class_name']} {det['confidence']:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
            ly = max(y1 - 6, th + 6)
            cv2.rectangle(result, (x1, ly-th-6), (x1+tw+8, ly+2), color, -1)
            cv2.putText(result, label, (x1+4, ly-2), font, 0.55, (0,0,0), 1, cv2.LINE_AA)
        return result


# ============================================================
# Depth Estimator
# ============================================================

class DepthAnythingV2Estimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.metric_depth = False
        self.lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        # Add parent dir to path for Depth_Anything_V2 package
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        encoder = config.MODEL_ENCODER
        model_cfg = model_configs[encoder]

        self.metric_depth = config.METRIC_DEPTH
        if self.metric_depth:
            model_cfg["max_depth"] = config.MAX_DEPTH
            ckpt_name = f"depth_anything_v2_metric_{config.METRIC_DATASET}_{encoder}.pth"
        else:
            ckpt_name = f"depth_anything_v2_{encoder}.pth"

        self.model = DepthAnythingV2(**model_cfg)

        ckpt_path = os.path.join(config.MODEL_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Download from HuggingFace and place in {config.MODEL_DIR}/"
            )

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

        if config.USE_FP16 and self.device.type == "cuda":
            self.model = self.model.half()

        print(f"[Depth] Model: {encoder} on {self.device}" +
              (" (FP16)" if config.USE_FP16 else "") +
              (f" | Metric depth max={config.MAX_DEPTH}m" if self.metric_depth else ""))

    @torch.no_grad()
    def estimate(self, frame):
        with self.lock:
            h, w = frame.shape[:2]
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE),
                             interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            if config.USE_FP16 and self.device.type == "cuda":
                img = img.half()

            depth = self.model(img).squeeze().cpu().float().numpy()
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

            if self.metric_depth:
                depth_max = depth.max()
                depth_vis = depth / depth_max if depth_max > 1e-6 else np.zeros_like(depth)
                colormap = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                return colormap, depth  # depth is in meters
            else:
                d_min, d_max = depth.min(), depth.max()
                if d_max - d_min > 1e-6:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth)
                colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                return colormap, depth_norm


# ============================================================
# Camera
# ============================================================

def open_camera():
    idx = config.CAMERA_INDEX
    if config.USE_GSTREAMER:
        pipeline = (
            f"v4l2src device=/dev/video{idx} ! "
            f"image/jpeg, width={config.CAMERA_WIDTH}, height={config.CAMERA_HEIGHT}, "
            f"framerate={config.CAMERA_FPS}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {idx}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] Opened: /dev/video{idx} {w}x{h} @ {fps:.0f}fps")

    # Configure exposure
    device = f"/dev/video{idx}"
    controls = [("exposure_auto", config.CAMERA_EXPOSURE_AUTO), ("white_balance_temperature_auto", 1)]
    if config.CAMERA_GAIN is not None:
        controls.append(("gain", config.CAMERA_GAIN))
    for ctrl, val in controls:
        try:
            subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"{ctrl}={val}"],
                           capture_output=True, text=True, timeout=2)
        except Exception:
            pass

    # Flush buffers
    for _ in range(5):
        cap.grab()

    return cap


# ============================================================
# Processing Pipeline
# ============================================================

class LocalProcessor:
    """All-in-one: camera capture + depth + YOLO + web results."""

    def __init__(self):
        self._estimator = DepthAnythingV2Estimator()
        self._detector = ObjectDetector() if config.YOLO_ENABLED else None
        self._cap = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Results
        self._rgb = None
        self._depth = None
        self._detection = None
        self._combined = None
        self._fps = 0.0
        self._depth_info = {"min_dist": 0, "max_dist": 0, "center_dist": 0}
        self._detections = []
        self._depth_history = []

        # Detection frame skipping
        self._detect_count = 0
        self._cached_dets = []

    def start(self):
        self._cap = open_camera()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="processor")
        self._thread.start()
        print("[Processor] Pipeline started")

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            t0 = time.time()

            # Depth estimation
            depth_color, depth_raw = self._estimator.estimate(frame)

            # Overlay distance info
            if self._estimator.metric_depth:
                depth_color = self._overlay_metric(depth_color, depth_raw)
                depth_for_det = depth_raw
            else:
                depth_color = self._overlay_relative(depth_color, depth_raw)
                depth_for_det = depth_raw

            # Object detection
            det_frame = None
            if self._detector is not None:
                self._detect_count += 1
                if self._detect_count >= config.YOLO_SKIP_FRAMES:
                    self._detect_count = 0
                    dets = self._detector.detect(frame)
                    dets = self._detector.measure_distances(
                        dets, depth_for_det, has_metric_depth=self._estimator.metric_depth
                    )
                    self._cached_dets = dets
                det_frame = self._detector.draw_detections(frame, self._cached_dets)

            # Combined view
            h, w = frame.shape[:2]
            depth_resized = cv2.resize(depth_color, (w, h))
            combined = np.hstack([frame, depth_resized])

            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 1e-6)

            with self._lock:
                self._rgb = frame
                self._depth = depth_color
                self._detection = det_frame
                self._combined = combined
                self._fps = fps
                if self._detector is not None:
                    self._detections = list(self._cached_dets)

    def _overlay_metric(self, depth_color, depth_m):
        h, w = depth_m.shape
        ch, cw = h // 2, w // 2
        rh, rw = h // 5, w // 5
        center = depth_m[ch-rh:ch+rh, cw-rw:cw+rw]
        valid = center[center > 0.1]

        if len(valid) > 10:
            center_dist = float(np.median(valid))
            min_dist = float(np.min(valid))
            max_dist = float(np.max(valid))

            self._depth_history.append(center_dist)
            if len(self._depth_history) > 15:
                self._depth_history.pop(0)
            stable = float(np.median(self._depth_history))

            self._depth_info = {
                "min_dist": round(min_dist, 2),
                "max_dist": round(max_dist, 2),
                "center_dist": round(stable, 2),
            }

            result = depth_color
            cv2.rectangle(result, (cw-rw, ch-rh), (cw+rw, ch+rh), (0,255,0), 1)
            cv2.drawMarker(result, (cw, ch), (0,255,0), cv2.MARKER_CROSS, 20, 1)

            text = f"{stable:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
            tx, ty = cw - tw//2, ch - rh - 10
            cv2.rectangle(result, (tx-4, ty-th-4), (tx+tw+4, ty+4), (0,0,0), -1)
            cv2.putText(result, text, (tx, ty), font, 0.8, (0,255,0), 2)
            cv2.putText(result, f"Min:{min_dist:.2f}m Max:{max_dist:.2f}m",
                        (8, h-10), font, 0.45, (200,200,200), 1)
            return result
        return depth_color

    def _overlay_relative(self, depth_color, depth_norm):
        h, w = depth_norm.shape
        ch, cw = h // 2, w // 2
        rh, rw = h // 5, w // 5
        center = depth_norm[ch-rh:ch+rh, cw-rw:cw+rw]

        if center.size > 0:
            center_val = float(np.median(center))
            min_val = float(np.min(depth_norm[depth_norm > 0.01])) if np.any(depth_norm > 0.01) else 0
            max_val = float(np.max(depth_norm))

            self._depth_info = {
                "center_dist": round(center_val, 3),
                "min_dist": round(min_val, 3),
                "max_dist": round(max_val, 3),
            }

            result = depth_color
            cv2.rectangle(result, (cw-rw, ch-rh), (cw+rw, ch+rh), (0,255,0), 1)
            cv2.drawMarker(result, (cw, ch), (0,255,0), cv2.MARKER_CROSS, 20, 1)

            text = f"{center_val:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
            tx, ty = cw - tw//2, ch - rh - 10
            cv2.rectangle(result, (tx-4, ty-th-4), (tx+tw+4, ty+4), (0,0,0), -1)
            cv2.putText(result, text, (tx, ty), font, 0.8, (0,255,0), 2)
            cv2.putText(result, f"Near:{max_val:.2f} Far:{min_val:.2f} (relative)",
                        (8, h-10), font, 0.45, (200,200,200), 1)
            return result
        return depth_color

    def get_results(self):
        with self._lock:
            return {
                "rgb": self._rgb,
                "depth": self._depth,
                "cam_detection": self._detection,
                "combined": self._combined,
                "fps": self._fps,
                "depth_info": dict(self._depth_info),
                "detections": list(self._detections),
            }

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        print("[Processor] Stopped")
