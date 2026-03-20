"""
Jetson Calibration Web Server - minimal Flask server for stereo calibration UI.
Run this on the Jetson to calibrate the cameras via web browser.

Usage:
    python3 calibrate.py
    Then open http://<jetson-ip>:5000/calibrate
"""

import cv2
import signal
import sys
import threading
import time

from flask import Flask, Response, render_template, jsonify, request

import config
from camera_manager import DualCameraManager
from stereo_calibrator import StereoCalibrator

app = Flask(__name__)

_camera_mgr = None
_calibrator = None


@app.route("/")
def index():
    return '<h2>Jetson Camera Tools</h2><p><a href="/calibrate">Stereo Calibration</a></p>'


@app.route("/calibrate")
def calibrate_page():
    return render_template("calibrate.html")


def _generate_calib_preview():
    interval = 1.0 / 15
    while True:
        if _calibrator is None:
            time.sleep(0.2)
            continue
        frame = _calibrator.get_preview_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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


def main():
    global _camera_mgr, _calibrator

    print("=" * 50)
    print("  Jetson Stereo Calibration Tool")
    print("=" * 50)
    print(f"  Cameras: [{config.CAMERA_0_INDEX}, {config.CAMERA_1_INDEX}]")
    print(f"  Web UI: http://0.0.0.0:{config.WEB_PORT}/calibrate")
    print("=" * 50)

    _camera_mgr = DualCameraManager()
    _camera_mgr.start()

    _calibrator = StereoCalibrator(_camera_mgr)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[Calibrate] Shutting down...")
        running = False
        _camera_mgr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Calibrate] Ready. Open the web UI to calibrate.")
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        threaded=True,
        use_reloader=False,
        debug=False,
    )


if __name__ == "__main__":
    main()
