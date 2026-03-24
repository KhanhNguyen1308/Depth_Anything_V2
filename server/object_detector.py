"""
Object Detector - YOLOv8 detection + depth-based distance measurement.
Detects objects in RGB frames and measures distance using depth maps.
"""

import cv2
import numpy as np
import os
import threading
import time

import config


# COCO class names for YOLOv8
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

# Colors for different classes (BGR)
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (64, 255, 64), (255, 64, 64), (64, 64, 255),
]


class ObjectDetector:
    """
    YOLOv8 object detector with depth-based distance measurement.

    Uses ultralytics YOLOv8 for detection and combines with depth maps
    to compute real-world distance to each detected object.
    """

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._detections = []  # Latest detection results
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model."""
        from ultralytics import YOLO

        model_name = getattr(config, "YOLO_MODEL", "yolov8n.pt")
        model_path = os.path.join(
            getattr(config, "MODEL_DIR", "checkpoints"), model_name
        )

        # If model file doesn't exist locally, ultralytics auto-downloads
        if os.path.exists(model_path):
            self._model = YOLO(model_path)
            print(f"[ObjectDetector] Loaded model from {model_path}")
        else:
            self._model = YOLO(model_name)
            print(f"[ObjectDetector] Loaded model: {model_name} (auto-download)")

        # Use GPU if available
        device = getattr(config, "YOLO_DEVICE", "auto")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        print(f"[ObjectDetector] Device: {self._device}")

    def detect(self, frame):
        """
        Run object detection on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            list of dicts: [{class_id, class_name, confidence, bbox: [x1,y1,x2,y2]}]
        """
        conf_thresh = getattr(config, "YOLO_CONFIDENCE", 0.5)
        input_size = getattr(config, "YOLO_INPUT_SIZE", 320)

        results = self._model.predict(
            frame,
            conf=conf_thresh,
            imgsz=input_size,
            device=self._device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })

        with self._lock:
            self._detections = detections

        return detections

    def measure_distances(self, detections, depth_map, has_metric_depth=False):
        """
        Compute distance to each detected object using the depth map.

        Args:
            detections: list from detect()
            depth_map: numpy array (H, W) - metric depth in meters or relative depth
            has_metric_depth: True if depth_map values are in meters

        Returns:
            list of dicts: same as detections + {distance, distance_unit}
        """
        if depth_map is None or len(detections) == 0:
            return detections

        h, w = depth_map.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                det["distance"] = 0
                det["distance_unit"] = "m" if has_metric_depth else ""
                continue

            # Use center 60% of the bounding box to avoid edges
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw = x2 - x1
            bh = y2 - y1
            rx = int(bw * 0.3)
            ry = int(bh * 0.3)

            roi = depth_map[
                max(0, cy - ry):min(h, cy + ry),
                max(0, cx - rx):min(w, cx + rx)
            ]

            if roi.size == 0:
                det["distance"] = 0
                det["distance_unit"] = "m" if has_metric_depth else ""
                continue

            if has_metric_depth:
                valid = roi[roi > 0.1]
                if len(valid) > 0:
                    det["distance"] = round(float(np.median(valid)), 2)
                else:
                    det["distance"] = 0
                det["distance_unit"] = "m"
            else:
                # Relative depth — higher value = closer
                det["distance"] = round(float(np.median(roi)), 3)
                det["distance_unit"] = ""

        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: BGR numpy array (will be modified in-place)
            detections: list from detect() or measure_distances()

        Returns:
            frame with detections drawn
        """
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = _COLORS[det["class_id"] % len(_COLORS)]

            # Bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Label text
            dist = det.get("distance", 0)
            unit = det.get("distance_unit", "")
            if dist > 0 and unit:
                label = f"{cls_name} {conf:.0%} | {dist:.2f}{unit}"
            elif dist > 0:
                label = f"{cls_name} {conf:.0%} | {dist:.3f}"
            else:
                label = f"{cls_name} {conf:.0%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Label background
            label_y = max(y1 - 6, th + 6)
            cv2.rectangle(result, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 2),
                          color, -1)
            cv2.putText(result, label, (x1 + 4, label_y - 2), font, font_scale,
                        (0, 0, 0), thickness, cv2.LINE_AA)

        return result

    def get_detections(self):
        """Get latest detections (thread-safe)."""
        with self._lock:
            return list(self._detections)
