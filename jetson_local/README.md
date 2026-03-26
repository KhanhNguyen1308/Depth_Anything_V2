# Depth Anything V2 – Jetson Nano Local Mode

Hệ thống ước lượng độ sâu thời gian thực chạy hoàn toàn trên **Jetson Nano**, không cần server ngoài. Kết hợp **Depth Anything V2** (metric depth) với **YOLOv8n** (object detection) và streaming qua **web dashboard**.

---

## Tổng quan kiến trúc

```
Camera (USB)
    │  V4L2, 640×480 @ 30fps
    ▼
┌─────────────────────────────────────────┐
│           LocalProcessor (thread)        │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  TensorRTDepthEstimator          │   │
│  │  • Preprocess: resize+center-crop│   │
│  │  • TRT engine: 308×420 FP32     │   │
│  │  • Calibration: scale + offset   │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  ObjectDetector (YOLOv8n)        │   │
│  │  • Chạy mỗi N frame (skip)      │   │
│  │  • Đo khoảng cách qua depth map │   │
│  └──────────────────────────────────┘   │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Flask Web Server (port 8080)     │
│  MJPEG stream + REST API                 │
└─────────────────────────────────────────┘
              │
              ▼
      Browser / Client
```

---

## Những gì hệ thống có thể làm hiện tại

### 1. Ước lượng độ sâu metric (Depth Estimation)
- Mô hình: **Depth Anything V2 ViT-S** (encoder nhỏ nhất, phù hợp Jetson Nano)
- Dataset calibration: **Hypersim** (trong nhà, max 20m)
- Output: độ sâu tuyệt đối tính bằng **mét** tại mỗi pixel
- Hiển thị: colormap **Inferno** (gần = sáng, xa = tối)
- Overlay: crosshair trung tâm + khoảng cách ổn định (median 15 frame)
- Calibration: đã chỉnh hệ số `DEPTH_SCALE=0.741`, `DEPTH_OFFSET=1.186` từ đo thực tế

### 2. Phát hiện vật thể (Object Detection)
- Mô hình: **YOLOv8n** (nano, nhanh nhất)
- Chạy mỗi N frame (cấu hình `YOLO_SKIP_FRAMES`) để tiết kiệm GPU
- Kết hợp với depth map để ước lượng khoảng cách đến từng vật
- Hiện tại tắt (`YOLO_ENABLED = False`) để tối đa FPS cho depth

### 3. Web Dashboard (port 8080)
| URL | Nội dung |
|-----|----------|
| `http://<IP>:8080/` | Trang chủ dashboard |
| `/video_feed/combined` | Camera RGB + depth map ghép ngang |
| `/video_feed/rgb` | Chỉ camera RGB |
| `/video_feed/depth` | Chỉ depth colormap |
| `/video_feed/detection` | Camera với bounding box YOLO |
| `/api/status` | JSON: fps, center_dist, min/max dist, detections |

### 4. REST API `/api/status`
```json
{
  "status": "running",
  "fps": 4.2,
  "model": "vits",
  "metric_depth": true,
  "center_dist": 1.23,
  "min_dist": 0.85,
  "max_dist": 4.10,
  "yolo_enabled": false,
  "detections": []
}
```

---

## Cấu hình phần cứng

| Thành phần | Thông số |
|---|---|
| Board | NVIDIA Jetson Nano (4GB) |
| GPU | 128-core Maxwell, CUDA 10.2 |
| TensorRT | 8.0.1.6 |
| Camera | USB webcam, `/dev/video0`, 640×480 |
| Backend camera | V4L2 (OpenCV biên dịch không có GStreamer) |

---

## Cấu trúc file

```
jetson_local/
├── main.py                          # Entry point
├── config.py                        # Tất cả tham số cấu hình
├── processor.py                     # Camera + depth + YOLO pipeline
├── web_server.py                    # Flask MJPEG streaming + API
│
├── export_onnx.py                   # Xuất checkpoint → ONNX opset-13
├── convert_to_trt.py                # ONNX → TensorRT engine
├── test_trt.py                      # Kiểm tra engine độc lập
│
├── checkpoints/
│   └── depth_anything_v2_metric_hypersim_vits.pth   # 95MB, PyTorch weights
│
├── hypersim_vits_308x420.onnx       # 95MB, ONNX opset-13 (aspect-correct)
├── hypersim_vits_308x420.engine     # 107MB, TensorRT engine (IN USE)
│
├── depth_anything_v2/               # Source gốc từ repo Depth Anything V2
│   ├── depth_anything_v2/           # Module chính (dinov2 + dpt)
│   └── metric_depth/                # Module metric depth (có max_depth)
│
├── static/style.css
├── templates/index.html
└── requirements.txt
```

---

## Pipeline khởi động

```bash
# 1. Lần đầu: export ONNX từ checkpoint (chạy 1 lần, ~3 phút)
python3 export_onnx.py --out hypersim_vits_308x420.onnx

# 2. Lần đầu: build TensorRT engine (chạy 1 lần, ~20 phút trên Jetson Nano)
python3 convert_to_trt.py --no-fp16

# 3. Chạy hệ thống (mỗi lần)
python3 main.py
# → Web dashboard: http://<jetson-ip>:8080
```

---

## Chi tiết kỹ thuật TensorRT

### Tại sao phải re-export thay vì dùng ONNX gốc?
ONNX gốc (`hypersim_vits.onnx`) xuất ở **opset 18** với:
- `LayerNormalization` — op fused, TRT 8.0 không có plugin
- `ReduceMean` dùng axes là **input tensor** (opset-18 style)
- `Resize` có attribute `antialias` / `keep_aspect_ratio_policy` — TRT 8.0 không hỗ trợ

→ Giải pháp: re-export trực tiếp từ PyTorch checkpoint ở **opset 13** bằng `export_onnx.py`. PyTorch tự phân rã `LayerNorm` thành primitives, dùng axes làm attribute (opset-13 style).

### Input shape: 308×420 (không phải vuông)
Camera 640×480 có tỷ lệ 4:3. Nếu resize thẳng về 308×308:
- Width bị nén 33% nhiều hơn height → ảnh bị méo dọc
- Mô hình nhìn thấy vật thể cao bất thường → ước lượng độ sâu sai

→ Giải pháp: export engine ở **308×420** (= `ceil(480×308/480×14)×14 = 308`, `ceil(640×308/480×14)×14 = 420`), preprocessing dùng **center-crop** thay vì `squeeze`.

### Calibration độ sâu
Hypersim được train với camera virtual có FOV khác webcam thực tế. Kết quả raw bị lệch:

```
raw_depth = DEPTH_SCALE × real_depth + DEPTH_OFFSET
corrected  = (raw_depth − DEPTH_OFFSET) / DEPTH_SCALE
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `DEPTH_SCALE` | 0.741 | Độ dốc (slope) |
| `DEPTH_OFFSET` | 1.186 m | Offset cố định (FOV mismatch) |

Đo thực tế: real=0.60m→raw=1.63m, real=1.68m→raw=2.43m.

---

## Cách hiệu chỉnh lại (Recalibration)

```bash
# Đặt vật ở 2+ khoảng cách đã biết, đọc giá trị raw từ log:
# [TRT] Output min=X max=Y  ← khi camera hướng vào vật

# Hoặc đọc center_dist từ API trong khi DEPTH_SCALE=1.0, DEPTH_OFFSET=0.0
# Sau đó tính:
python3 -c "
r1, d1 = 0.60, 1.63   # (real_m, raw_m) đo điểm 1
r2, d2 = 1.68, 2.43   # (real_m, raw_m) đo điểm 2
scale  = (d2-d1)/(r2-r1)
offset = d1 - scale*r1
print(f'DEPTH_SCALE  = {scale:.4f}')
print(f'DEPTH_OFFSET = {offset:.4f}')
"
```

---

## Cấu hình nhanh (`config.py`)

| Tham số | Giá trị hiện tại | Chú thích |
|---|---|---|
| `CAMERA_INDEX` | `0` | `/dev/video0` |
| `CAMERA_WIDTH/HEIGHT` | `640×480` | Độ phân giải camera |
| `INFERENCE_BACKEND` | `"tensorrt"` | `"tensorrt"` \| `"pytorch"` \| `"onnxrt"` |
| `METRIC_DEPTH` | `True` | True = mét, False = tương đối |
| `METRIC_DATASET` | `"hypersim"` | `"hypersim"` (20m) \| `"vkitti"` (80m) |
| `MAX_DEPTH` | `20` | Giới hạn hiển thị (mét) |
| `DEPTH_SCALE` | `0.741` | Hệ số calibration |
| `DEPTH_OFFSET` | `1.186` | Offset calibration (mét) |
| `YOLO_ENABLED` | `False` | Bật YOLO giảm FPS đáng kể |
| `YOLO_SKIP_FRAMES` | `3` | Chạy YOLO mỗi 3 frame |
| `WEB_PORT` | `8080` | Port web dashboard |
| `JPEG_QUALITY` | `80` | Chất lượng stream (0–100) |
| `STREAM_FPS` | `15` | FPS tối đa web stream |

---

## Giới hạn hiện tại

- **FPS**: ~4–6 FPS depth inference trên Jetson Nano (Maxwell GPU, TRT FP32)
- **Độ chính xác**: sai số ~±0.1–0.2m sau calibration hai điểm; dùng nhiều điểm đo hơn để tốt hơn
- **Camera re-enumeration**: Sau khi khởi động TRT, camera có thể đổi index (`/dev/video0` ↔ `/dev/video1`). Kiểm tra bằng `ls /dev/video*` nếu camera không mở được
- **FP16**: Chưa bật (engine FP32) do issue với opset cũ; có thể thử lại với engine mới bằng cách bỏ `--no-fp16`
- **YOLO**: Tắt mặc định; khi bật đồng thời với TRT depth, FPS giảm mạnh
