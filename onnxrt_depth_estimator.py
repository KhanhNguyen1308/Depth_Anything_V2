"""
ONNX Runtime Depth Estimator - Inference Depth Anything V2 bằng ONNX Runtime.
Hỗ trợ TensorRT EP, CUDA EP, CPU EP trên Jetson Nano.

ONNX Runtime tự động xử lý các op không được TensorRT hỗ trợ
(như LayerNormalization) bằng cách fallback về CUDA EP.
"""

import os
import cv2
import numpy as np
import threading
import time

import config

try:
    import onnxruntime as ort
    ONNXRT_AVAILABLE = True
except ImportError:
    ONNXRT_AVAILABLE = False


class OnnxRTDepthEstimator:
    """Depth estimation sử dụng ONNX Runtime với TensorRT/CUDA EP."""

    def __init__(self):
        if not ONNXRT_AVAILABLE:
            raise RuntimeError(
                "ONNX Runtime không khả dụng!\n"
                "Cài đặt cho Jetson Nano:\n"
                "  pip3 install onnxruntime-gpu\n"
                "Hoặc tải wheel từ NVIDIA Jetson Zoo."
            )

        self.session = None
        self.input_name = None
        self.output_name = None
        self.lock = threading.Lock()

        self._create_session()

    def _create_session(self):
        """Tạo ONNX Runtime session với execution provider phù hợp."""
        onnx_path = getattr(config, "ONNX_MODEL", "depth_anything_v2_vits.onnx")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Không tìm thấy ONNX model: {onnx_path}\n"
                f"Đặt file .onnx vào thư mục project."
            )

        print(f"[OnnxRT] Loading model: {onnx_path}")
        t0 = time.time()

        # Chọn execution provider theo thứ tự ưu tiên
        providers = []
        available_providers = ort.get_available_providers()
        print(f"[OnnxRT] Available providers: {available_providers}")

        # TensorRT EP - nhanh nhất, tự cache engine
        if "TensorrtExecutionProvider" in available_providers:
            trt_cache_dir = os.path.join(config.MODEL_DIR, "trt_cache")
            os.makedirs(trt_cache_dir, exist_ok=True)
            trt_options = {
                "device_id": 0,
                "trt_max_workspace_size": getattr(config, "TENSORRT_WORKSPACE_MB", 512) * 1024 * 1024,
                "trt_fp16_enable": config.USE_FP16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache_dir,
            }
            providers.append(("TensorrtExecutionProvider", trt_options))

        # CUDA EP - fallback cho ops mà TRT không hỗ trợ
        if "CUDAExecutionProvider" in available_providers:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))

        # CPU EP - luôn có
        providers.append("CPUExecutionProvider")

        # Session options tối ưu cho Jetson Nano
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )

        # Lấy thông tin input/output
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.output_name = outputs[0].name
        self.output_shape = outputs[0].shape

        active_provider = self.session.get_providers()
        elapsed = time.time() - t0

        print(f"[OnnxRT] Active providers: {active_provider}")
        print(f"[OnnxRT] Input:  {self.input_name} {self.input_shape}")
        print(f"[OnnxRT] Output: {self.output_name} {self.output_shape}")
        print(f"[OnnxRT] Session created in {elapsed:.2f}s")
        print(f"[OnnxRT] Ready for inference")

    def _preprocess(self, frame):
        """Preprocess frame cho ONNX Runtime inference."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # (1, 3, H, W)
        return img.astype(np.float32)

    def estimate(self, frame):
        """
        Tạo depth map từ một frame BGR sử dụng ONNX Runtime.

        Args:
            frame: numpy array BGR (H, W, 3)

        Returns:
            depth_colormap: numpy array BGR depth colormap (H, W, 3)
            depth_raw: numpy array float32 normalized [0, 1] (H, W)
        """
        with self.lock:
            h, w = frame.shape[:2]

            # Preprocess
            input_data = self._preprocess(frame)

            # Inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )

            depth = outputs[0]
            depth = np.squeeze(depth)  # Remove batch dim

            # Resize về kích thước gốc
            if depth.ndim == 2:
                depth = cv2.resize(depth, (w, h))
            else:
                depth = depth.reshape(-1)
                side = int(np.sqrt(depth.shape[0]))
                depth = depth[:side * side].reshape(side, side)
                depth = cv2.resize(depth, (w, h))

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
