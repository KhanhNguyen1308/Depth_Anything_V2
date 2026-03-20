"""
Web Server - Flask streaming server cho Jetson Nano headless.
Hiển thị RGB + Depth map từ 2 camera trên giao diện web.
"""

import cv2
import time
import threading
from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS

import config


app = Flask(__name__)
CORS(app)

_depth_processor = None


def set_depth_processor(processor):
    global _depth_processor
    _depth_processor = processor


def _encode_jpeg(frame):
    """Encode frame thành JPEG bytes."""
    if frame is None:
        return None
    ret, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    )
    if ret:
        return jpeg.tobytes()
    return None


def _generate_stream(stream_key):
    """Generator cho MJPEG streaming."""
    interval = 1.0 / config.STREAM_FPS
    last_send = 0.0
    while True:
        if _depth_processor is None:
            time.sleep(0.1)
            continue

        now = time.monotonic()
        elapsed = now - last_send
        if elapsed < interval:
            time.sleep(interval - elapsed)

        results = _depth_processor.get_results()
        frame = results.get(stream_key)

        if frame is not None:
            jpeg_bytes = _encode_jpeg(frame)
            if jpeg_bytes:
                last_send = time.monotonic()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )


@app.route("/")
def index():
    """Trang chính."""
    return render_template("index.html")


@app.route("/video_feed/combined")
def video_feed_combined():
    """Stream combined view (2x2 grid: RGB + Depth cho 2 camera)."""
    return Response(
        _generate_stream("combined"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed/cam0_rgb")
def video_feed_cam0_rgb():
    """Stream Camera 0 RGB."""
    return Response(
        _generate_stream("cam0_rgb"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed/cam0_depth")
def video_feed_cam0_depth():
    """Stream Camera 0 Depth."""
    return Response(
        _generate_stream("cam0_depth"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed/cam1_rgb")
def video_feed_cam1_rgb():
    """Stream Camera 1 RGB."""
    return Response(
        _generate_stream("cam1_rgb"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed/cam1_depth")
def video_feed_cam1_depth():
    """Stream Camera 1 Depth."""
    return Response(
        _generate_stream("cam1_depth"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    """API trả về trạng thái hệ thống + khoảng cách depth."""
    if _depth_processor is None:
        return jsonify({"status": "initializing", "fps": 0})

    results = _depth_processor.get_results()
    depth_info = results.get("depth_info", {})
    has_stereo = getattr(_depth_processor, '_sgbm', None) is not None
    return jsonify({
        "status": "running",
        "fps": round(results.get("fps", 0), 1),
        "model": config.MODEL_ENCODER,
        "input_size": config.MODEL_INPUT_SIZE,
        "fp16": config.USE_FP16,
        "stream_port": config.STREAM_PORT,
        "stereo": has_stereo,
        "center_dist": depth_info.get("center_dist", 0),
        "min_dist": depth_info.get("min_dist", 0),
        "max_dist": depth_info.get("max_dist", 0),
    })


def run_server():
    """Chạy Flask server trong thread riêng."""
    print(f"[WebServer] Starting on http://{config.WEB_HOST}:{config.WEB_PORT}")
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        threaded=True,
        use_reloader=False,
        debug=False,
    )

