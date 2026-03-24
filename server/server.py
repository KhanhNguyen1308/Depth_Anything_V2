"""
Depth Processing Server - receives camera streams from Jetson and runs
depth estimation + stereo SGBM + web UI.

Usage:
    python3 server.py
    python3 server.py --port 9000 --web-port 5000
"""

import argparse
import signal
import sys
import threading
import time

import config
from camera_receiver import RemoteCameraReceiver
from http_camera_receiver import HttpCameraReceiver
from depth_estimator import SingleCameraDepthEstimator
from web_server import run_server, set_depth_processor


def main():
    parser = argparse.ArgumentParser(description="Depth Processing Server")
    parser.add_argument("--source", default=config.STREAM_SOURCE,
                        choices=["tcp", "tunnel"],
                        help="Camera source: 'tcp' (LAN) or 'tunnel' (cloudflared)")
    parser.add_argument("--port", type=int, default=config.STREAM_PORT,
                        help="TCP port for camera stream (tcp mode)")
    parser.add_argument("--tunnel-url", default=config.TUNNEL_STREAM_URL,
                        help="Tunnel stream URL (tunnel mode)")
    parser.add_argument("--web-port", type=int, default=config.WEB_PORT,
                        help="Web UI port")
    args = parser.parse_args()

    print("=" * 60)
    print("  Depth Anything V2 - Processing Server")
    print("=" * 60)
    print(f"  Source mode: {args.source}")
    if args.source == "tcp":
        print(f"  Camera stream port: {args.port}")
    else:
        print(f"  Tunnel URL: {args.tunnel_url}")
    print(f"  Web UI: http://0.0.0.0:{args.web_port}")
    print(f"  Model: {config.MODEL_ENCODER} | Input: {config.MODEL_INPUT_SIZE}px")
    print(f"  Backend: {config.INFERENCE_BACKEND}")
    print("=" * 60)

    # Initialize depth processor
    print("\n[Server] Initializing depth estimator...")
    depth_processor = SingleCameraDepthEstimator()

    # Initialize camera receiver based on source mode
    print("[Server] Starting camera receiver...")
    if args.source == "tunnel":
        receiver = HttpCameraReceiver(stream_url=args.tunnel_url.rstrip("/") + "/stream")
    else:
        receiver = RemoteCameraReceiver(host="0.0.0.0", port=args.port)
    receiver.start()

    # Register processor with web server
    set_depth_processor(depth_processor)

    # Override web port if specified
    config.WEB_PORT = args.web_port

    # Start depth processing with remote camera source
    depth_processor.start(receiver)

    # Start web server
    web_thread = threading.Thread(target=run_server, daemon=True)
    web_thread.start()

    # Graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[Server] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.source == "tunnel":
        print(f"\n[Server] Ready. Reading stream from {args.tunnel_url}")
    else:
        print(f"\n[Server] Ready. Waiting for Jetson camera connection on port {args.port}...")
    print(f"  Web UI: http://0.0.0.0:{args.web_port}")

    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Server] Cleaning up...")
        depth_processor.stop()
        receiver.stop()
        print("[Server] Done.")


if __name__ == "__main__":
    main()
