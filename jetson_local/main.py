#!/usr/bin/env python3
"""
Jetson Nano Local Mode - Main Entry Point.
Runs Depth Anything V2 + YOLOv8n + Web Dashboard locally.
No remote server required.
"""

import signal
import sys
import threading

import config
from processor import LocalProcessor
from web_server import set_processor, run_server


def main():
    print("=" * 50)
    print("  Depth Anything V2 - Jetson Nano Local Mode")
    print(f"  Camera: /dev/video{config.CAMERA_INDEX} ({config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT})")
    print(f"  Depth:  {config.MODEL_ENCODER} {'metric ('+config.METRIC_DATASET+')' if config.METRIC_DEPTH else 'relative'}")
    print(f"  YOLO:   {'enabled ('+config.YOLO_MODEL+')' if config.YOLO_ENABLED else 'disabled'}")
    print(f"  Web:    http://0.0.0.0:{config.WEB_PORT}")
    print("=" * 50)

    # Start processor (camera + depth + YOLO)
    processor = LocalProcessor()
    processor.start()

    # Wire up web server
    set_processor(processor)

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\n[Main] Shutting down...")
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run web server (blocks)
    run_server()


if __name__ == "__main__":
    main()
