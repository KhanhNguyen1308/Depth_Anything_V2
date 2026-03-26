# Đề xuất cải tiến hệ thống Depth Anything V2

Tổng hợp các hướng tối ưu, phân loại theo mức độ ưu tiên và độ khó.

---

## Mục lục

1. [Hiệu năng – Nhanh (dưới 1 ngày)](#1-hiệu-năng--nhanh-dưới-1-ngày)
2. [Độ chính xác – Nhanh](#2-độ-chính-xác--nhanh)
3. [Hiệu năng – Trung bình](#3-hiệu-năng--trung-bình)
4. [Độ chính xác – Trung bình](#4-độ-chính-xác--trung-bình)
5. [Hiệu năng – Khó/Dài hạn](#5-hiệu-năng--khódài-hạn)
6. [Độ chính xác – Khó/Dài hạn](#6-độ-chính-xác--khódài-hạn)
7. [Bảng so sánh tổng hợp](#7-bảng-so-sánh-tổng-hợp)

---

## 1. Hiệu năng – Nhanh (dưới 1 ngày)

### 1.1 Bật chế độ hiệu suất tối đa của Jetson

Mặc định Jetson Nano chạy ở 5W (tiết kiệm điện). Chuyển sang 10W tăng FPS đáng kể.

```bash
# Bật chế độ 10W (max-N)
sudo nvpmodel -m 0
sudo jetson_clocks            # Lock CPU+GPU ở xung nhịp tối đa

# Kiểm tra nhiệt độ và xung nhịp
sudo jtop
```

> Kết quả thực tế: thường tăng **30–50% FPS** chỉ bằng lệnh trên. Cần tản nhiệt tốt (quạt chủ động).

---

### 1.2 Sử dụng Pinned Memory cho H2D/D2H transfer

Hiện tại `img.ctypes.data_as(...)` trỏ vào **pageable memory** — mỗi cudaMemcpy phải staging qua DMA buffer tạm. Dùng `numpy` pinned array (page-locked) loại bỏ bước đó.

**File:** `processor.py` — trong `TensorRTDepthEstimator.__init__`:

```python
# Thay dòng:
h_buf = np.empty(shape, dtype=np_dtype)

# Thành: (dùng ctypes để đăng ký pinned memory với CUDA)
import ctypes
h_buf = np.empty(shape, dtype=np_dtype)
self._libcuda.cudaHostRegister(
    h_buf.ctypes.data_as(ctypes.c_void_p),
    ctypes.c_size_t(h_buf.nbytes),
    ctypes.c_uint(0),  # cudaHostRegisterDefault
)
```

Hoặc dùng `torch.cuda.cudart().cudaMallocHost()` API.

> Tiết kiệm ~0.5–2ms mỗi lần transfer (quan trọng khi frame rate thấp).

---

### 1.3 Pre-allocate input buffer để tránh tạo array mới mỗi frame

Hiện tại `estimate()` gọi `img.astype(np.float32)` và `np.ascontiguousarray(...)` mỗi frame → cấp phát và giải phóng bộ nhớ liên tục (GC pressure).

**File:** `processor.py` — thêm vào `TensorRTDepthEstimator.__init__`:

```python
in_shape = self._binding_shapes[0]  # (1, 3, H, W)
self._input_host = np.empty(in_shape, dtype=self._binding_dtypes[0])  # pre-alloc
```

Trong `estimate()`:

```python
# Thay các dòng tạo array mới thành ghi trực tiếp vào self._input_host
np.copyto(self._input_host[0], img.transpose(2, 0, 1))
# → không cấp phát bộ nhớ mới
```

---

### 1.4 Giảm kích thước ảnh stream web

Hiện tại web stream gửi ảnh cùng kích thước gốc (640×480 × 2 cho combined). Việc encode JPEG 1280×480 tốn CPU.

**File:** `config.py`:

```python
STREAM_WIDTH  = 640   # resize trước khi encode JPEG (combined → 640×240)
STREAM_HEIGHT = 240
```

**File:** `web_server.py` — thêm resize trước khi encode frame:

```python
frame = cv2.resize(frame, (config.STREAM_WIDTH, config.STREAM_HEIGHT))
```

> Giảm ~30% thời gian encode, tiết kiệm bandwidth.

---

### 1.5 Tăng buffer camera lên MJPG nếu có thể

Đã bật `FOURCC=MJPG` và `BUFFERSIZE=1`. Đảm bảo camera thực sự output MJPG (không phải YUYV):

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

Nếu camera hỗ trợ, thử `1280×720 MJPG` — kernel decode nhanh hơn `640×480 YUYV` tốn băng thông USB hơn.

---

## 2. Độ chính xác – Nhanh

### 2.1 Calibration nhiều điểm (Least Squares thay vì 2 điểm)

Hiện tại dùng 2 điểm → đường thẳng qua 2 điểm (có thể kém chính xác ở khoảng cách khác). Dùng ≥5 điểm và linear regression cho kết quả tốt hơn đáng kể.

**Tạo script `calibrate_depth.py`:**

```python
#!/usr/bin/env python3
"""
Công cụ calibration nhiều điểm.
Đặt vật ở các khoảng cách đã biết, nhập raw depth từ API /api/status → center_dist
(nhớ tắt calibration: set DEPTH_SCALE=1.0, DEPTH_OFFSET=0.0 trước khi đo)
"""
import numpy as np

measurements = [
    # (real_m, raw_m)   ← đo thực tế
    (0.50, ???),
    (0.75, ???),
    (1.00, ???),
    (1.50, ???),
    (2.00, ???),
    (3.00, ???),
]

real = np.array([m[0] for m in measurements])
raw  = np.array([m[1] for m in measurements])

# Linear regression: raw = scale * real + offset
A = np.vstack([real, np.ones(len(real))]).T
scale, offset = np.linalg.lstsq(A, raw, rcond=None)[0]

print(f"DEPTH_SCALE  = {scale:.4f}   # (raw - offset) / scale = real")
print(f"DEPTH_OFFSET = {offset:.4f}")

# Kiểm tra sai số
pred = (raw - offset) / scale
errors = np.abs(pred - real)
print(f"Mean error: {errors.mean()*100:.1f}cm | Max error: {errors.max()*100:.1f}cm")
```

> Với 6+ điểm đo, sai số trung bình thường giảm từ ±15cm xuống ±5cm.

---

### 2.2 Temporal smoothing dạng EMA thay vì median window cố định

Hiện tại `_depth_history` là list 15 phần tử → pop(0) O(n). Dùng **Exponential Moving Average (EMA)** nhanh hơn và mượt hơn:

**File:** `processor.py` — trong `_overlay_metric`:

```python
# Thay _depth_history bằng EMA
EMA_ALPHA = 0.3   # 0.1 = rất mượt/chậm, 0.5 = nhanh hơn/giật hơn

if not hasattr(self, '_ema_dist'):
    self._ema_dist = center_dist
self._ema_dist = EMA_ALPHA * center_dist + (1 - EMA_ALPHA) * self._ema_dist
stable = self._ema_dist
```

> O(1) thay vì O(n), nhưng quan trọng hơn: phản ứng nhanh hơn khi vật thể di chuyển.

---

### 2.3 Dùng ROI nhỏ hơn + loại outlier để đo khoảng cách trung tâm

Hiện tại ROI trung tâm = `w/5 × h/5` (128×96 pixel) — quá lớn, bao gồm cả background nếu vật thể nhỏ.

**File:** `processor.py` — trong `_overlay_metric`:

```python
# ROI nhỏ hơn: 5% mỗi chiều
rh, rw = max(h // 20, 8), max(w // 20, 8)   # ~24×32 px thay vì 96×128

# Loại outlier (loại bỏ 10% giá trị cực đoan)
if len(valid) > 20:
    p10, p90 = np.percentile(valid, [10, 90])
    valid = valid[(valid >= p10) & (valid <= p90)]
```

---

### 2.4 Dùng INTER_CUBIC khi upscale depth map về kích thước gốc

Hiện tại `cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)` — nhanh nhưng tạo aliasing ở cạnh. `INTER_CUBIC` mượt hơn khi upscale (308×420 → 480×640).

**File:** `processor.py` — cuối hàm `estimate()`:

```python
# Hiện tại:
depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

# Cải tiến:
depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
```

> Chi phí thêm ~1ms. Chất lượng YOLO distance measurement tốt hơn ở biên vật thể.

---

## 3. Hiệu năng – Trung bình

### 3.1 Build engine FP16

Engine hiện tại là FP32 (107MB). Maxwell GPU của Jetson Nano **hỗ trợ FP16 native**. FP16 engine thường nhanh hơn **1.5–2×** trên Maxwell.

```bash
# Bỏ flag --no-fp16
python3 convert_to_trt.py   # dùng FP16 mặc định

# Nếu vẫn lỗi, thêm --workspace 1024 để tăng workspace TRT
```

Nếu vẫn có lỗi accuracy, có thể dùng **mixed precision**: set một số layer về FP32 manually trong `convert_to_trt.py`.

**Config sau khi build xong:**

```python
# config.py
USE_FP16 = True
TENSORRT_ENGINE = "hypersim_vits_308x420_fp16.engine"
```

> Kỳ vọng: ~4–6 FPS → **8–12 FPS** với FP16.

---

### 3.2 Giảm độ phân giải input (252×336 thay vì 308×420)

Model ViT-S dùng patch size 14. Giảm 1 bậc:
- 308→252: bỏ 4 patch hàng (22→18)
- 420→336: bỏ 6 patch cột (30→24)

```bash
# Trong export_onnx.py, sửa size=18 thay vì size=22:
python3 export_onnx.py --size 18 --out hypersim_vits_252x336.onnx
python3 convert_to_trt.py --input hypersim_vits_252x336.onnx
```

> Input nhỏ hơn 33% về pixel → inference nhanh hơn ~20–25%. Độ chính xác giảm nhẹ (~5% độ sâu).

---

### 3.3 CUDA Streams: overlap preprocessing → inference

Hiện tại mỗi frame: `preprocess (CPU) → H2D → execute → D2H` tuần tự.

Với 2 stream CUDA, có thể overlap:
```
Frame N:   │─preprocess─│─H2D─│─execute─│─D2H─│
Frame N+1: │──preprocess──────────│─H2D─│─execute─│─D2H─│
```

**File:** `processor.py` — trong `TensorRTDepthEstimator.__init__`:

```python
import ctypes
self._stream = ctypes.c_void_p()
self._libcuda.cudaStreamCreate(ctypes.byref(self._stream))
```

Thay `cudaMemcpy` → `cudaMemcpyAsync` + `cudaStreamSynchronize`:

```python
self._libcuda.cudaMemcpyAsync(
    self._d_ptrs[0], img.ctypes.data_as(ctypes.c_void_p),
    ctypes.c_size_t(nbytes_in),
    ctypes.c_int(1),   # cudaMemcpyHostToDevice
    self._stream,
)
self._context.execute_async_v2(bindings=bindings, stream_handle=int(self._stream.value))
self._libcuda.cudaStreamSynchronize(self._stream)
```

> Lợi ích thực sự phụ thuộc vào tỷ lệ CPU/GPU time. Nếu preprocessing chiếm >30% time thì đáng kể.

---

### 3.4 Skip depth frame khi camera không di chuyển (motion detection)

Nếu camera tĩnh và scene ít thay đổi, có thể giữ depth cũ thay vì tính lại.

**File:** `processor.py` — trong `_loop`:

```python
MOTION_THRESH = 500   # tổng pixel khác nhau

if hasattr(self, '_prev_gray'):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, self._prev_gray)
    motion = int(diff.sum()) // 255
    if motion < MOTION_THRESH and self._depth is not None:
        # Reuse depth từ frame trước
        depth_color, depth_raw = self._depth_color_cache, self._depth_raw_cache
    else:
        depth_color, depth_raw = self._estimator.estimate(frame)
    self._prev_gray = gray
else:
    self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    depth_color, depth_raw = self._estimator.estimate(frame)
```

> Có thể tiết kiệm 50–70% inference khi camera tĩnh (ví dụ: mounted robot, giám sát).

---

## 4. Độ chính xác – Trung bình

### 4.1 Nâng lên mô hình ViT-B

ViT-S (hiện tại): 25M params, encoder features 64
ViT-B: 97M params, encoder features 128

Cần download checkpoint `depth_anything_v2_metric_hypersim_vitb.pth` (~395MB) và re-export:

```python
# config.py
MODEL_ENCODER = "vitb"

# export_onnx.py sẽ tự detect vitb từ config
```

```bash
python3 export_onnx.py --out hypersim_vitb_308x420.onnx
python3 convert_to_trt.py --no-fp16 --input hypersim_vitb_308x420.onnx
```

> ViT-B thường giảm sai số ~20–30% so với ViT-S, nhưng FPS sẽ giảm thêm (~50%).

---

### 4.2 Camera lens calibration (undistort)

Lens rộng (wide-angle) của USB webcam tạo barrel distortion — làm méo depth ở rìa ảnh. Calibrate camera với checkerboard và undistort trước khi đưa vào model.

```bash
# Bước 1: chụp 20+ ảnh checkerboard (9×6 inner corners)
python3 -c "
import cv2, numpy as np, glob
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints, imgpoints = [], []
for fname in glob.glob('calib/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria))
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.save('camera_matrix.npy', K)
np.save('dist_coeffs.npy', dist)
print('Reprojection error:', ret)
"
```

**File:** `processor.py` — trong `open_camera()`:

```python
import numpy as np
K    = np.load('camera_matrix.npy')
dist = np.load('dist_coeffs.npy')
h, w = config.CAMERA_HEIGHT, config.CAMERA_WIDTH
new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
# Trong _loop:
frame = cv2.undistort(frame, K, dist, None, new_K)
```

> Cải thiện độ chính xác depth ở góc ảnh ~15–25%.

---

### 4.3 Depth map hole filling (inpainting)

Sau calibration, một số vùng depth có thể = 0 (phản chiếu, kính, bóng tối). Dùng `cv2.inpaint` để fill.

**File:** `processor.py` — sau `np.clip(depth, ...)`:

```python
if np.any(depth < 0.1):
    mask = (depth < 0.1).astype(np.uint8) * 255
    # Scale về uint16 để inpaint (inpaint chỉ hỗ trợ uint8/uint16)
    d_scaled = np.clip(depth / config.MAX_DEPTH * 65535, 0, 65535).astype(np.uint16)
    d_filled = cv2.inpaint(d_scaled, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    depth = d_filled.astype(np.float32) / 65535.0 * config.MAX_DEPTH
```

> Hữu ích với các bề mặt phản chiếu (gương, kính cửa).

---

### 4.4 Median filter trên depth map (khử noise per-pixel)

Depth map thô từ TRT có thể có noise pixel-level (đặc biệt ở vùng texture thấp). Median filter 3×3 khử noise tốt mà không làm mờ cạnh như Gaussian.

**File:** `processor.py` — sau `depth = out_buf.squeeze()...`:

```python
# Thêm sau khi có depth raw (trước calibration)
depth = cv2.medianBlur(depth.astype(np.float32), 3)
```

> Không ảnh hưởng đáng kể đến performance (~0.5ms), giảm noise đáng kể ở vùng phẳng.

---

## 5. Hiệu năng – Khó/Dài hạn

### 5.1 INT8 Quantization (TensorRT Calibration)

INT8 thường nhanh hơn FP32 **3–4×** trên Maxwell. Cần **calibration dataset** (~100–500 ảnh thực tế từ webcam) để TRT tính activation scale.

```python
# convert_to_trt.py — thêm INT8 calibrator
class Int8Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, image_list, batch_size=1, cache_file="calib.cache"):
        ...
    def get_batch(self, names):
        ...   # load từng ảnh, preprocess, cudaMemcpy H2D
    def read_calibration_cache(self):
        ...
    def write_calibration_cache(self, cache):
        ...

# builder config:
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Int8Calibrator(image_list)
```

> Nếu thực hiện đúng: ~4–6 FPS → **15–20 FPS**. Accuracy loss thường <2% cho depth task.

---

### 5.2 GPU-accelerated colormap (OpenCV CUDA module)

`cv2.applyColorMap()` chạy trên CPU. Nếu build OpenCV với CUDA:

```python
# Kiểm tra:
print(cv2.cuda.getCudaEnabledDeviceCount())

# Thay:
colormap = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
# Bằng:
gpu_depth = cv2.cuda_GpuMat()
gpu_depth.upload((depth_vis * 255).astype(np.uint8))
gpu_color = cv2.cuda.applyColorMap(gpu_depth, cv2.COLORMAP_INFERNO)
colormap = gpu_color.download()
```

> Tiết kiệm ~2–3ms mỗi frame (640×480 colormap trên CPU ~3ms trên Jetson).

---

### 5.3 NVJPEG encoding thay vì OpenCV imencode

`cv2.imencode('.jpg', frame)` dùng libjpeg-turbo trên CPU. NVJPEG của Jetson encode trên GPU.

```python
# Sử dụng nvjpeg qua PyNvJpeg (nếu cài được):
import pynvjpeg
encoder = pynvjpeg.NvJpeg()
_, buf = encoder.encode(frame, quality=config.JPEG_QUALITY)
```

> Giảm độ trễ web stream ~5–10ms ở 1280×480 frame, giải phóng CPU cho preprocessing.

---

### 5.4 Chuyển sang model chuyên biệt cho indoor: ZoeDepth hoặc Metric3D

Depth Anything V2 metric (Hypersim) train trong nhà nhưng trong môi trường synthetic. Với dữ liệu real-world indoor, các model sau có thể chính xác hơn:

- **ZoeDepth** (NK variant): train trên NYU Depth v2 (real indoor) + KITTI
- **Metric3D v2**: zero-shot metric depth, không cần calibration FOV

> Yêu cầu re-export ONNX + TRT. ZoeDepth ViT-S có thể dùng pipeline tương tự.

---

## 6. Độ chính xác – Khó/Dài hạn

### 6.1 Fine-tune trên dữ liệu thực tế từ webcam

Nếu có depth sensor reference (Intel RealSense, Kinect), thu thập dataset gồm:
- RGB từ webcam
- Depth ground truth từ sensor

Sau đó fine-tune `depth_anything_v2_metric_hypersim_vits.pth` trên dataset này.

```bash
# Xem metric_depth/train.py trong repo gốc
# Thêm dataset tương tự hypersim.py cho data thực tế
cd depth_anything_v2/metric_depth
python3 train.py --encoder vits --dataset custom --data-path /path/to/dataset \
                 --pretrained-from ../../checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

> Fine-tune 1–2 epoch thường cải thiện MAE **30–50%** khi có ~1000 ảnh real.

---

### 6.2 Depth fusion: TRT depth + ToF/stereo sensor

Kết hợp depth từ model (dense nhưng không chính xác tuyệt đối) với cảm biến vật lý (thưa nhưng chính xác) qua weighted fusion:

```python
def fuse_depth(depth_model, depth_sensor_sparse):
    """
    depth_model: (H, W) float, dense, less accurate
    depth_sensor_sparse: (H, W) float, sparse (0 = no measurement)
    """
    mask = depth_sensor_sparse > 0.1
    # Warp sensor depth về model coordinate
    # Scale model output để match sensor ở vùng overlap
    scale  = depth_sensor_sparse[mask].mean() / depth_model[mask].mean()
    fused  = depth_model * scale
    fused[mask] = 0.7 * depth_sensor_sparse[mask] + 0.3 * fused[mask]
    return fused
```

---

### 6.3 Per-frame FOV-aware scale recovery (không cần DEPTH_OFFSET cố định)

Thay vì dùng offset cố định (1.186m), ước lượng scale bù mỗi frame dựa trên matching optical flow với depth gradient. Phức tạp nhưng không cần calibration cố định.

---

## 7. Bảng so sánh tổng hợp

| # | Cải tiến | Loại | Độ khó | FPS gain | Accuracy gain |
|---|---|---|---|---|---|
| 1.1 | Jetson max performance mode | Perf | ⭐ Dễ | **+30–50%** | – |
| 1.2 | Pinned memory H2D/D2H | Perf | ⭐⭐ | +5–10% | – |
| 1.3 | Pre-alloc input buffer | Perf | ⭐ | +3–5% | – |
| 1.4 | Giảm stream resolution | Perf | ⭐ | +CPU 10% | – |
| 2.1 | Multi-point calibration | Acc | ⭐ | – | **sai số ±15cm→±5cm** |
| 2.2 | EMA thay median window | Acc+Perf | ⭐ | +2% | phản ứng nhanh hơn |
| 2.3 | ROI nhỏ + outlier filter | Acc | ⭐ | – | ±3–5cm ít bị lẫn bg |
| 2.4 | INTER_CUBIC upscale depth | Acc | ⭐ | −1% | cạnh mượt hơn |
| 3.1 | FP16 engine | Perf | ⭐⭐ | **+50–100%** | loss <3% |
| 3.2 | Giảm input 252×336 | Perf | ⭐⭐ | +20–25% | loss ~5% |
| 3.3 | CUDA Streams async | Perf | ⭐⭐⭐ | +10–20% | – |
| 3.4 | Motion-based skip | Perf | ⭐⭐ | +50% khi camera tĩnh | – |
| 4.1 | Nâng lên ViT-B | Acc | ⭐⭐ | −50% | **±20–30% tốt hơn** |
| 4.2 | Camera undistort | Acc | ⭐⭐ | −2% | tốt hơn ở góc ảnh |
| 4.3 | Depth inpainting | Acc | ⭐⭐ | −3% | fix vùng 0 |
| 4.4 | Median filter depth | Acc | ⭐ | −1% | giảm noise |
| 5.1 | INT8 quantization | Perf | ⭐⭐⭐⭐ | **+200–300%** | loss <2% |
| 5.2 | GPU colormap | Perf | ⭐⭐⭐ | +3–5ms/frame | – |
| 5.3 | NVJPEG encoding | Perf | ⭐⭐⭐ | giảm latency stream | – |
| 5.4 | Đổi model ZoeDepth | Acc | ⭐⭐⭐ | tương đương | tùy thuộc scene |
| 6.1 | Fine-tune trên real data | Acc | ⭐⭐⭐⭐ | – | **±30–50% tốt hơn** |
| 6.2 | Depth fusion với sensor | Acc | ⭐⭐⭐⭐ | – | chính xác tuyệt đối |

---

## Roadmap gợi ý

```
Tuần 1 (low-hanging fruit):
  ✅ 1.1 jetson_clocks + nvpmodel (0 code, +40% FPS ngay)
  ✅ 2.1 calibrate_depth.py (nhiều điểm)
  ✅ 2.2 EMA smoothing
  ✅ 4.4 Median filter depth

Tuần 2 (engine rebuild):
  🔧 3.1 Build FP16 engine
  🔧 1.3 Pre-alloc input buffer

Tuần 3 (nếu FPS vẫn chưa đủ):
  🔧 3.2 Giảm resolution 252×336
  🔧 3.4 Motion-based skip

Dài hạn (nếu cần accuracy cao):
  🔧 4.2 Camera undistort
  🔧 4.1 ViT-B (+ FP16 để bù FPS)
  🔧 5.1 INT8 (kết quả lớn nhất về FPS)
  🔧 6.1 Fine-tune nếu có ground truth depth
```
