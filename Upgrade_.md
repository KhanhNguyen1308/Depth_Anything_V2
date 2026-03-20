# Gợi ý cải tiến - Depth Anything V2 trên Jetson Nano

> Hệ thống hiện tại: Jetson Nano 4GB, Ubuntu 20.04, Python 3.8, OpenCV 4.8,
> TensorRT 8.0.1 FP16, ViT-S 518×518, ~800ms/frame (~1.3 FPS)

---

## A. Cải thiện hiệu năng (~800ms → mục tiêu <300ms)

### 1. Giảm input size
- Export lại ONNX với input **308×308** hoặc **252×252** thay vì 518×518
- Tốc độ tăng ~3-4x, chất lượng depth vẫn chấp nhận được cho stereo matching
- Cần re-export ONNX từ PyTorch với `input_size` mới, rồi chạy lại `fix_onnx_model.py` và `convert_onnx_to_trt.py`

### 2. INT8 quantization
- Chuẩn bị 100-200 ảnh calibration (chụp từ camera thực tế)
- Build engine INT8 bằng `convert_onnx_to_trt.py --int8 --calib-dir calib_images/`
- Trên Maxwell (Jetson Nano) có thể nhanh hơn FP16 ~30-50%

### 3. Pipeline xử lý song song (3 thread)
- **Thread 1**: Camera grab (liên tục lấy frame mới nhất)
- **Thread 2**: Preprocess + inference camera 0
- **Thread 3**: Preprocess + inference camera 1
- Overlap preprocessing frame tiếp theo với inference frame hiện tại

### 4. Preprocess trên GPU
- Dùng OpenCV CUDA (`cv2.cuda`) để resize, cvtColor, normalize trên GPU
- Tránh bottleneck CPU→GPU memcpy cho mỗi frame
- Ước tính tiết kiệm ~20-50ms/frame

### 5. Double buffering CUDA
- Allocate 2 bộ input/output buffer CUDA
- Xen kẽ memcpy và inference để overlap data transfer với compute
- Giảm thêm ~10-20% latency

### 6. Xử lý xen kẽ 2 camera
- Thay vì inference cả 2 frame mỗi cycle, luân phiên 1 camera/cycle:
  `cam0 → cam1 → cam0 → cam1 → ...`
- Mỗi camera vẫn cập nhật depth, nhưng latency tổng giảm 50%

### 7. Giảm resolution camera
- Dùng **320×240** thay vì 640×480
- Giảm thời gian grab + preprocess
- Kết hợp với input_size nhỏ hơn để tối ưu toàn bộ pipeline

---

## B. Tính toán độ sâu tuyệt đối từ 2 camera (Stereo Depth)

> Depth Anything V2 chỉ cho **relative depth** (monocular).
> Để có **absolute depth (mét)**, cần kết hợp stereo geometry.

### 8. Stereo calibration
- Calibrate cặp camera OV9732 bằng bàn cờ vua (checkerboard)
- Tính intrinsic matrix (K), distortion coefficients cho mỗi camera
- Tính extrinsic (R, T) giữa 2 camera bằng `cv2.stereoCalibrate()`
- Lưu kết quả calibration ra file `.npz` để tái sử dụng

### 9. Stereo rectification
- Dùng `cv2.stereoRectify()` + `cv2.initUndistortRectifyMap()` để warp 2 ảnh về cùng mặt phẳng epipolar
- Chỉ tính rectification map 1 lần khi khởi động, lưu cache
- Áp dụng `cv2.remap()` cho mỗi frame (rất nhanh, ~1-2ms)

### 10. Hybrid depth: Mono relative + Stereo absolute
Đây là cách tiếp cận khả thi nhất trên Jetson Nano:

```
Camera L ──grab──┐                    ┌── Stereo SGBM ── sparse depth (m)
                 ├── rectify ──┤                              │
Camera R ──grab──┘                    └── Depth Anything V2   │
                                          (relative depth)    │
                                               │              │
                                               └── scale ─────┘
                                                  alignment
                                                     │
                                              absolute depth map (m)
```

**Bước thực hiện:**
1. **Stereo SGBM disparity**: Dùng `cv2.StereoSGBM_create()` để tính disparity map từ 2 ảnh rectified
2. **Disparity → depth thật**: `Z = (f × B) / d` (f: focal length px, B: baseline m, d: disparity px)
3. **Mono relative depth**: Chạy Depth Anything V2 trên camera trái → dense relative depth
4. **Scale alignment**: Fit affine transform `Z_abs = a × D_mono + b` bằng least-squares
   - Lấy ~50-100 điểm tin cậy từ stereo depth làm anchor
   - Kết quả: depth map dense (từ mono) + chính xác (từ stereo scale)

### 11. Temporal consistency
- Dùng exponential moving average trên scale factors (a, b) qua các frame:
  `a_t = α × a_new + (1-α) × a_{t-1}` (α ≈ 0.3)
- Tránh jitter khi stereo matching không ổn định giữa các frame

### 12. Confidence filtering cho stereo
- Chỉ dùng stereo points có disparity ổn định làm anchor cho scale alignment
- Left-right consistency check: chạy SGBM cả 2 chiều, loại bỏ điểm lệch > 1px
- Loại bỏ vùng occlusion, vùng texture-less (low variance)

### 13. Depth output format
- Lưu depth map dạng **uint16 (mm)** hoặc **float32 (m)** để downstream processing
- Hỗ trợ xuất point cloud (PCL/Open3D), obstacle detection, SLAM

---

## C. Tối ưu dài hạn

### 14. Nâng cấp Jetson Orin Nano
- GPU Ampere mạnh hơn ~5-10x so với Jetson Nano Maxwell
- TRT 8.6+ hỗ trợ LayerNormalization native (không cần decompose)
- INT8 chất lượng cao hơn nhờ Ampere tensor cores

### 15. Export ONNX dynamic axes
- Cho phép thay đổi input size runtime mà không cần rebuild engine
- Dùng TensorRT optimization profile với min/opt/max shapes

### 16. Depth Anything V2 metric depth
- Repo gốc có nhánh `metric_depth/` đã train với ground-truth metric
- Có thể fine-tune cho indoor/outdoor use case cụ thể
- Giảm/bỏ hoàn toàn nhu cầu stereo alignment

---

## D. Thứ tự ưu tiên thực hiện

| Ưu tiên | Cải tiến | Tác động | Độ phức tạp |
|---------|----------|----------|-------------|
| 🔴 Cao | #1 Giảm input size 308×308 | FPS x3-4 | Thấp |
| 🔴 Cao | #8, #9 Stereo calibration + rectification | Nền tảng cho depth tuyệt đối | Trung bình |
| 🔴 Cao | #10 Hybrid mono+stereo depth | Depth tuyệt đối (mét) | Trung bình |
| 🟡 TB | #3 Pipeline song song | FPS +50-80% | Trung bình |
| 🟡 TB | #6 Xen kẽ 2 camera | Latency -50% | Thấp |
| 🟡 TB | #2 INT8 quantization | FPS +30-50% | Thấp |
| 🟢 Thấp | #4, #5 GPU preprocess + double buffer | FPS +10-20% | Cao |
| 🟢 Thấp | #11, #12 Temporal + confidence filter | Chất lượng depth | Trung bình |
| 🔵 Dài hạn | #14, #16 Orin Nano / Metric depth | Tổng thể | Cao |