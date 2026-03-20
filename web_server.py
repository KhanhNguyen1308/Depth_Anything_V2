"""
Web Server - Flask streaming server cho Jetson Nano headless.
Hiển thị RGB + Depth map từ 2 camera trên giao diện web.
"""

import cv2
import time
import threading
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS

import config


app = Flask(__name__)
CORS(app)

# Reference tới depth estimator (set từ main.py)
_depth_processor = None
_calibrator = None


def set_depth_processor(processor):
    global _depth_processor
    _depth_processor = processor


def set_calibrator(calibrator):
    global _calibrator
    _calibrator = calibrator


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
    return jsonify({
        "status": "running",
        "fps": round(results.get("fps", 0), 1),
        "model": config.MODEL_ENCODER,
        "input_size": config.MODEL_INPUT_SIZE,
        "fp16": config.USE_FP16,
        "cameras": [config.CAMERA_0_INDEX, config.CAMERA_1_INDEX],
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


# === Calibration routes ===

@app.route("/calibrate")
def calibrate_page():
    """Calibration UI page."""
    return render_template("calibrate.html")


def _generate_calib_preview():
    """Generator for calibration preview MJPEG stream."""
    interval = 1.0 / 15  # 15 FPS for preview
    while True:
        if _calibrator is None:
            time.sleep(0.2)
            continue

        frame = _calibrator.get_preview_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
        time.sleep(interval)


@app.route("/calibrate/preview")
def calibrate_preview():
    """MJPEG stream with checkerboard detection overlay."""
    return Response(
        _generate_calib_preview(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/calibrate/status")
def calibrate_status():
    if _calibrator is None:
        return jsonify({"error": "Calibrator not initialized"}), 503
    return jsonify(_calibrator.get_status())


@app.route("/api/calibrate/settings", methods=["POST"])
def calibrate_settings():
    if _calibrator is None:
        return jsonify({"error": "Calibrator not initialized"}), 503
    data = request.get_json()
    cols = data.get("cols", 9)
    rows = data.get("rows", 6)
    square_size = data.get("square_size", 25.0)
    if not (2 <= cols <= 20 and 2 <= rows <= 20 and 1 <= square_size <= 200):
        return jsonify({"error": "Invalid parameters"}), 400
    _calibrator.set_board_size(cols, rows, square_size)
    return jsonify(_calibrator.get_status())


@app.route("/api/calibrate/capture", methods=["POST"])
def calibrate_capture():
    if _calibrator is None:
        return jsonify({"error": "Calibrator not initialized"}), 503
    success = _calibrator.capture_pair()
    return jsonify({"success": success, **_calibrator.get_status()})


@app.route("/api/calibrate/delete", methods=["POST"])
def calibrate_delete():
    if _calibrator is None:
        return jsonify({"error": "Calibrator not initialized"}), 503
    data = request.get_json()
    index = data.get("index", -1)
    success = _calibrator.delete_pair(int(index))
    return jsonify({"success": success, **_calibrator.get_status()})


@app.route("/api/calibrate/clear", methods=["POST"])
def calibrate_clear():
    if _calibrator is None:
        return jsonify({"error": "Calibrator not initialized"}), 503
    _calibrator.clear_pairs()
    return jsonify(_calibrator.get_status())


@app.route("/api/calibrate/run", methods=["POST"])
def calibrate_run():
    if _calibrator is None:
        return jsonify({"error": "Calibrator not initialized"}), 503

    def _run():
        _calibrator.run_calibration()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"started": True, **_calibrator.get_status()})
