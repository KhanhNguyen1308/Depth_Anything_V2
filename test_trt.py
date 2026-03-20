"""
Test TensorRT / ONNX Runtime inference trên Jetson Nano.

Cách dùng:
    # Test với ảnh dummy (không cần camera)
    python3 test_trt.py

    # Test với ảnh thực
    python3 test_trt.py --image path/to/image.jpg

    # Test với camera
    python3 test_trt.py --camera 0

    # Chỉ định backend
    python3 test_trt.py --backend onnxrt
    python3 test_trt.py --backend tensorrt
"""

import argparse
import time
import sys
import os

import cv2
import numpy as np

import config


def check_dependencies():
    """Kiểm tra dependencies cần thiết."""
    print("=== Kiểm tra dependencies ===")

    # ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"  ONNX Runtime: {ort.__version__} OK")
        print(f"    Providers: {ort.get_available_providers()}")
    except ImportError:
        print("  ONNX Runtime: CHƯA CÀI")
        print("    pip3 install onnxruntime-gpu")

    # TensorRT
    try:
        import tensorrt as trt
        print(f"  TensorRT: {trt.__version__} OK")
    except ImportError:
        print("  TensorRT: CHƯA CÀI (optional)")

    # OpenCV
    print(f"  OpenCV: {cv2.__version__}")

    # ONNX file
    onnx_path = config.ONNX_MODEL
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ONNX model: {onnx_path} ({size_mb:.1f} MB)")
    else:
        print(f"  ONNX model: {onnx_path} KHÔNG TÌM THẤY!")
        return False

    # TRT engine
    engine_path = os.path.join(config.MODEL_DIR, config.TENSORRT_ENGINE)
    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"  TRT engine: {engine_path} ({size_mb:.1f} MB)")
    else:
        print(f"  TRT engine: {engine_path} (chưa có, sẽ tự convert)")

    print()
    return True


def test_inference(estimator, frame, num_warmup=3, num_runs=10):
    """Test inference và đo tốc độ."""
    print(f"  Input frame: {frame.shape}")

    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        estimator.estimate(frame)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        depth_color, depth_raw = estimator.estimate(frame)
        elapsed = time.time() - t0
        times.append(elapsed)

    times = np.array(times)
    fps = 1.0 / times.mean()

    print(f"\n  === Kết quả ===")
    print(f"  Thời gian trung bình: {times.mean() * 1000:.1f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Min: {times.min() * 1000:.1f} ms | Max: {times.max() * 1000:.1f} ms")
    print(f"  Output depth_color: {depth_color.shape}")
    print(f"  Output depth_raw: {depth_raw.shape}, range [{depth_raw.min():.3f}, {depth_raw.max():.3f}]")

    return depth_color, depth_raw


def main():
    parser = argparse.ArgumentParser(description="Test Depth Anything V2 inference")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh test")
    parser.add_argument("--camera", type=int, help="Camera index để test")
    parser.add_argument("--backend", type=str, default=None,
                        help="Backend: onnxrt, tensorrt, pytorch (mặc định: theo config)")
    parser.add_argument("--save", type=str, default="test_output.jpg",
                        help="Lưu kết quả ra file (mặc định: test_output.jpg)")
    parser.add_argument("--runs", type=int, default=10, help="Số lần chạy benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("  Depth Anything V2 - Inference Test")
    print("=" * 60)

    if not check_dependencies():
        print("Thiếu dependencies! Cài đặt rồi thử lại.")
        sys.exit(1)

    # Load estimator
    if args.backend:
        config.INFERENCE_BACKEND = args.backend

    backend = getattr(config, "INFERENCE_BACKEND", "onnxrt").lower()
    print(f"\n=== Loading backend: {backend} ===")
    try:
        from depth_estimator import create_estimator
        estimator = create_estimator()
    except Exception as e:
        print(f"Lỗi load estimator: {e}")
        sys.exit(1)

    # Lấy frame test
    if args.image:
        print(f"\n=== Test với ảnh: {args.image} ===")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Không đọc được ảnh: {args.image}")
            sys.exit(1)
    elif args.camera is not None:
        print(f"\n=== Test với camera {args.camera} ===")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            print(f"Không mở được camera {args.camera}")
            sys.exit(1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Không đọc được frame từ camera")
            sys.exit(1)
    else:
        print(f"\n=== Test với ảnh dummy {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} ===")
        frame = np.random.randint(0, 255,
            (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)

    # Run test
    depth_color, depth_raw = test_inference(estimator, frame, num_runs=args.runs)

    # Save output
    if args.save:
        # Ghép RGB + Depth cạnh nhau
        h, w = frame.shape[:2]
        depth_resized = cv2.resize(depth_color, (w, h))
        combined = np.hstack([frame, depth_resized])
        cv2.imwrite(args.save, combined)
        print(f"\n  Đã lưu kết quả: {args.save}")

    print("\n  Inference OK!")
    print("=" * 60)


if __name__ == "__main__":
    main()
