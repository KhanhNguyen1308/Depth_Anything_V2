"""
Convert ONNX model sang TensorRT engine cho Jetson Nano.

Cách dùng:
    # FP16 (khuyến nghị cho Jetson Nano - nhanh, chính xác tốt)
    python3 convert_onnx_to_trt.py --onnx depth_anything_v2_vits.onnx --fp16

    # FP32 (chậm hơn, chính xác hơn)
    python3 convert_onnx_to_trt.py --onnx depth_anything_v2_vits.onnx

    # Tùy chỉnh input size
    python3 convert_onnx_to_trt.py --onnx depth_anything_v2_vits.onnx --fp16 --height 308 --width 308

    # INT8 (nhanh nhất, cần calibration dataset)
    python3 convert_onnx_to_trt.py --onnx depth_anything_v2_vits.onnx --int8 --calib-dir calib_images/

Lưu ý:
    - Phải chạy trên Jetson Nano (TensorRT engine không portable giữa các GPU khác nhau)
    - Jetson Nano JetPack 4.6 có TensorRT 8.0.1
    - Quá trình convert mất 5-30 phút tùy model size
"""

import os
import sys
import argparse
import time
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not found!")
    print("Trên Jetson Nano, TensorRT được cài sẵn với JetPack.")
    print("Kiểm tra: dpkg -l | grep tensorrt")
    print("Nếu chưa có: sudo apt-get install tensorrt python3-libnvinfer-dev")
    sys.exit(1)


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator sử dụng ảnh từ thư mục calibration."""

    def __init__(self, calib_dir, input_shape, cache_file="calibration.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = 1
        self.input_shape = input_shape  # (1, 3, H, W)

        import cv2
        import glob

        self.images = []
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        for pat in patterns:
            self.images.extend(glob.glob(os.path.join(calib_dir, pat)))

        if not self.images:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh calibration trong '{calib_dir}'.\n"
                f"Đặt 50-200 ảnh đại diện vào thư mục này."
            )

        print(f"[Calibrator] Found {len(self.images)} calibration images")
        self.current_idx = 0

        # Allocate device memory
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.device_input = cuda.mem_alloc(
            int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.images):
            return None

        import cv2
        import pycuda.driver as cuda

        img_path = self.images[self.current_idx]
        self.current_idx += 1

        img = cv2.imread(img_path)
        if img is None:
            return self.get_batch(names)

        # Preprocess giống inference
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0).astype(np.float32)
        img = np.ascontiguousarray(img)

        cuda.memcpy_htod(self.device_input, img)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def convert_onnx_to_tensorrt(
    onnx_path,
    engine_path=None,
    fp16=False,
    int8=False,
    calib_dir=None,
    input_height=308,
    input_width=308,
    workspace_mb=1024,
):
    """
    Convert ONNX model sang TensorRT engine.

    Args:
        onnx_path: Đường dẫn file .onnx
        engine_path: Đường dẫn output .engine (mặc định: cùng tên với .engine)
        fp16: Bật FP16 mode
        int8: Bật INT8 mode (cần calib_dir)
        calib_dir: Thư mục chứa ảnh calibration cho INT8
        input_height: Chiều cao input
        input_width: Chiều rộng input
        workspace_mb: Workspace size (MB) cho TensorRT builder
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Không tìm thấy ONNX file: {onnx_path}")

    if engine_path is None:
        basename = os.path.splitext(os.path.basename(onnx_path))[0]
        suffix = "_fp16" if fp16 else ("_int8" if int8 else "_fp32")
        # Lưu engine vào checkpoints/ (match config.py)
        os.makedirs("checkpoints", exist_ok=True)
        engine_path = os.path.join("checkpoints", f"{basename}{suffix}.engine")

    print("=" * 60)
    print("  ONNX -> TensorRT Conversion")
    print("=" * 60)
    print(f"  ONNX model:    {onnx_path}")
    print(f"  Output engine: {engine_path}")
    print(f"  Input shape:   (1, 3, {input_height}, {input_width})")
    print(f"  FP16:          {fp16}")
    print(f"  INT8:          {int8}")
    print(f"  Workspace:     {workspace_mb} MB")
    print(f"  TensorRT ver:  {trt.__version__}")
    print("=" * 60)

    # Tạo builder
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("\n[1/3] Parsing ONNX model...")
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        for i in range(parser.num_errors):
            print(f"  ERROR: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")

    print(f"  Network inputs:  {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    [{i}] {inp.name}: {inp.shape} ({inp.dtype})")

    print(f"  Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    [{i}] {out.name}: {out.shape} ({out.dtype})")

    # Cấu hình builder
    print("\n[2/3] Configuring builder...")
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1 << 20)

    # Set input profile cho dynamic/static shapes
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = (1, 3, input_height, input_width)

    # Nếu model có dynamic shape, set profile
    if any(d == -1 for d in input_tensor.shape):
        print(f"  Dynamic shape detected, setting profile...")
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)
    else:
        print(f"  Static shape: {input_tensor.shape}")

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 mode enabled")
        else:
            print("  WARNING: Platform does not support fast FP16!")

    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calib_dir:
                calibrator = Int8Calibrator(calib_dir, input_shape)
                config.int8_calibrator = calibrator
                print(f"  INT8 mode enabled with calibration from '{calib_dir}'")
            else:
                print("  WARNING: INT8 mode without calibration - quality may be poor")
        else:
            print("  WARNING: Platform does not support fast INT8!")

    # Build engine
    print("\n[3/3] Building TensorRT engine (this may take several minutes)...")
    t_start = time.time()

    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError(
            "Failed to build TensorRT engine!\n"
            "Thử giảm workspace_mb hoặc input size."
        )

    elapsed = time.time() - t_start
    print(f"  Engine built in {elapsed:.1f}s")

    # Serialize engine
    print(f"\n  Saving engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Engine size: {engine_size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print(f"  DONE! Engine saved: {engine_path}")
    print(f"  Cập nhật config.py:")
    print(f"    TENSORRT_ENGINE = \"{engine_path}\"")
    print(f"    USE_TENSORRT = True")
    print("=" * 60)

    return engine_path


def verify_engine(engine_path, input_height=308, input_width=308):
    """Kiểm tra engine có load được không."""
    print(f"\n[Verify] Loading engine: {engine_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("  FAILED to load engine!")
        return False

    print(f"  Bindings: {engine.num_bindings}")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        is_input = engine.binding_is_input(i)
        io = "INPUT" if is_input else "OUTPUT"
        print(f"    [{i}] {io}: {name} {shape} ({dtype})")

    # Dummy inference test
    print("\n[Verify] Running dummy inference...")
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        context = engine.create_execution_context()
        dummy_input = np.random.randn(1, 3, input_height, input_width).astype(np.float32)

        # Allocate memory
        d_input = cuda.mem_alloc(dummy_input.nbytes)

        out_shape = engine.get_binding_shape(1)
        out_size = int(np.prod(out_shape))
        dummy_output = np.empty(out_size, dtype=np.float32)
        d_output = cuda.mem_alloc(dummy_output.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, dummy_input, stream)

        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)

        cuda.memcpy_dtoh_async(dummy_output, d_output, stream)
        stream.synchronize()

        print(f"  Output shape: {out_shape}")
        print(f"  Output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
        print("  PASSED!")
        return True

    except Exception as e:
        print(f"  Verify failed: {e}")
        print("  Engine đã tạo OK, có thể lỗi do pycuda. Thử chạy main.py trực tiếp.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Depth Anything V2 ONNX to TensorRT engine"
    )
    parser.add_argument(
        "--onnx", type=str, required=True,
        help="Đường dẫn file ONNX (vd: depth_anything_v2_vits.onnx)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Đường dẫn output .engine (mặc định: tự tạo từ tên ONNX)"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Bật FP16 mode (khuyến nghị cho Jetson Nano)"
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Bật INT8 mode (cần --calib-dir)"
    )
    parser.add_argument(
        "--calib-dir", type=str, default=None,
        help="Thư mục chứa ảnh calibration cho INT8"
    )
    parser.add_argument(
        "--height", type=int, default=308,
        help="Input height (mặc định: 308)"
    )
    parser.add_argument(
        "--width", type=int, default=308,
        help="Input width (mặc định: 308)"
    )
    parser.add_argument(
        "--workspace", type=int, default=512,
        help="TensorRT workspace size in MB (mặc định: 512, phù hợp Jetson Nano 4GB)"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Bỏ qua bước verify engine"
    )

    args = parser.parse_args()

    if args.int8 and not args.calib_dir:
        print("WARNING: INT8 mode cần --calib-dir để đạt chất lượng tốt.")
        print("Tạo thư mục calib_images/ chứa 50-200 ảnh đại diện.")

    engine_path = convert_onnx_to_tensorrt(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        calib_dir=args.calib_dir,
        input_height=args.height,
        input_width=args.width,
        workspace_mb=args.workspace,
    )

    if not args.no_verify:
        verify_engine(engine_path, args.height, args.width)


if __name__ == "__main__":
    main()
