"""
TensorRT Depth Estimator - Inference Depth Anything V2 bằng TensorRT.
Nhanh hơn ~3-5x so với PyTorch trên Jetson Nano.
"""

import os
import cv2
import numpy as np
import threading
import time

import config

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if TENSORRT_AVAILABLE else None


class TensorRTDepthEstimator:
    """Depth estimation sử dụng TensorRT engine."""

    def __init__(self):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT không khả dụng!\n"
                "Trên Jetson: sudo apt-get install tensorrt python3-libnvinfer-dev\n"
                "pip3 install pycuda"
            )

        self.engine = None
        self.context = None
        self.stream = None
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.output_shape = None
        self.lock = threading.Lock()
        # Lưu CUDA context để push/pop khi gọi từ thread khác
        self.cuda_ctx = cuda.Context.get_current()

        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine từ file. Tự động convert từ ONNX nếu chưa có."""
        engine_path = os.path.join(config.MODEL_DIR, config.TENSORRT_ENGINE)

        if not os.path.exists(engine_path):
            # Tự động convert từ ONNX nếu có file ONNX
            onnx_path = getattr(config, "ONNX_MODEL", "depth_anything_v2_vits.onnx")
            if os.path.exists(onnx_path):
                print(f"[TRT] Engine chưa có, tự động convert từ ONNX: {onnx_path}")
                self._convert_from_onnx(onnx_path, engine_path)
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy TensorRT engine: {engine_path}\n"
                    f"Và không tìm thấy ONNX file: {onnx_path}\n"
                    f"Chạy convert thủ công:\n"
                    f"  python3 convert_onnx_to_trt.py --onnx depth_anything_v2_vits.onnx --fp16"
                )

        print(f"[TRT] Loading engine: {engine_path}")
        t0 = time.time()

        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine!")

        self.context = self.engine.create_execution_context()

        # Lấy thông tin bindings
        input_idx = 0
        output_idx = 1

        input_shape = self.engine.get_binding_shape(input_idx)
        self.output_shape = self.engine.get_binding_shape(output_idx)
        input_dtype = trt.nptype(self.engine.get_binding_dtype(input_idx))
        output_dtype = trt.nptype(self.engine.get_binding_dtype(output_idx))

        print(f"[TRT] Input:  {self.engine.get_binding_name(input_idx)} "
              f"{input_shape} ({input_dtype.__name__})")
        print(f"[TRT] Output: {self.engine.get_binding_name(output_idx)} "
              f"{self.output_shape} ({output_dtype.__name__})")

        # Allocate GPU memory
        input_size = int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize
        output_size = int(np.prod(self.output_shape)) * np.dtype(output_dtype).itemsize

        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.h_output = np.empty(int(np.prod(self.output_shape)), dtype=output_dtype)

        # CUDA stream
        self.stream = cuda.Stream()

        elapsed = time.time() - t0
        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"[TRT] Engine loaded in {elapsed:.2f}s ({engine_size_mb:.1f}MB)")
        print(f"[TRT] Ready for inference")

    def _convert_from_onnx(self, onnx_path, engine_path):
        """Tự động convert ONNX sang TensorRT engine."""
        from convert_onnx_to_trt import convert_onnx_to_tensorrt

        workspace_mb = getattr(config, "TENSORRT_WORKSPACE_MB", 512)
        convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16=config.USE_FP16,
            input_height=config.MODEL_INPUT_SIZE,
            input_width=config.MODEL_INPUT_SIZE,
            workspace_mb=workspace_mb,
        )

    def _preprocess(self, frame):
        """Preprocess frame cho TensorRT inference."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # (1, 3, H, W)
        return np.ascontiguousarray(img, dtype=np.float32)

    def estimate(self, frame):
        """
        Tạo depth map từ một frame BGR sử dụng TensorRT.

        Args:
            frame: numpy array BGR (H, W, 3)

        Returns:
            depth_colormap: numpy array BGR depth colormap (H, W, 3)
            depth_raw: numpy array float32 normalized [0, 1] (H, W)
        """
        h, w = frame.shape[:2]
        input_data = self._preprocess(frame)

        with self.lock:
            # Push CUDA context cho thread hiện tại
            self.cuda_ctx.push()
            try:
                # Copy input to GPU
                cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

                # Inference
                self.context.execute_async_v2(
                    bindings=[int(self.d_input), int(self.d_output)],
                    stream_handle=self.stream.handle,
                )

                # Copy output to CPU
                cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
                self.stream.synchronize()
            finally:
                self.cuda_ctx.pop()

            # Copy output ra trước khi release lock (h_output là shared buffer)
            depth = self.h_output.reshape(self.output_shape).copy()

        # Postprocess ngoài lock — cho phép thread khác inference song song
        depth = np.squeeze(depth)  # Remove batch dim

        # Resize về kích thước gốc (INTER_CUBIC cho depth sắc nét hơn)
        if depth.ndim == 2:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            depth = depth.reshape(-1)
            side = int(np.sqrt(depth.shape[0]))
            depth = depth[:side * side].reshape(side, side)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        # Normalize
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

    def __del__(self):
        """Cleanup CUDA resources."""
        if self.d_input is not None:
            self.d_input.free()
        if self.d_output is not None:
            self.d_output.free()
