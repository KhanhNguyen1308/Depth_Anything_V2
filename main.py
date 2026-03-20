"""
Main - Entry point cho Depth Anything V2 trên Jetson Nano.
Chạy 2 camera USB OV9732 + depth estimation + web streaming.
"""

import signal
import sys
import time
import threading

import config
from camera_manager import DualCameraManager
from depth_estimator import DualCameraDepthEstimator
from web_server import run_server, set_depth_processor, set_calibrator
from stereo_calibrator import StereoCalibrator


def main():
    print("=" * 60)
    print("  Depth Anything V2 - Jetson Nano Dual Camera System")
    print("=" * 60)
    print(f"  Model: {config.MODEL_ENCODER} | Input: {config.MODEL_INPUT_SIZE}px")
    print(f"  FP16: {config.USE_FP16} | Target FPS: {config.TARGET_FPS}")
    print(f"  Cameras: [{config.CAMERA_0_INDEX}, {config.CAMERA_1_INDEX}]")
    print(f"  Pipeline: 3-thread parallel (grab + 2x inference)")
    print(f"  Web UI: http://0.0.0.0:{config.WEB_PORT}")
    print("=" * 60)

    # Khởi tạo components
    print("\n[Main] Initializing depth estimator...")
    depth_processor = DualCameraDepthEstimator()

    print("[Main] Initializing cameras...")
    camera_mgr = DualCameraManager()
    camera_mgr.start()

    # Đăng ký processor với web server
    set_depth_processor(depth_processor)

    # Khởi tạo calibrator
    calibrator = StereoCalibrator(camera_mgr)
    set_calibrator(calibrator)
    print(f"[Main] Calibration UI: http://0.0.0.0:{config.WEB_PORT}/calibrate")

    # Khởi động pipeline song song (2 inference threads)
    depth_processor.start(camera_mgr)

    # Khởi động web server trong thread riêng
    web_thread = threading.Thread(target=run_server, daemon=True)
    web_thread.start()

    # Graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[Main] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Main] Processing started. Press Ctrl+C to stop.")

    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Main] Cleaning up...")
        depth_processor.stop()
        camera_mgr.stop()
        print("[Main] Done.")


if __name__ == "__main__":
    main()
