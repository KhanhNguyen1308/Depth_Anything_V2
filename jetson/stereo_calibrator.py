"""
Stereo Calibrator - Calibrate 2 USB cameras via checkerboard pattern.
Captures synchronized image pairs, detects corners, runs stereoCalibrate.
"""

import os
import cv2
import numpy as np
import threading
import time
import json

import config


class StereoCalibrator:
    """Interactive stereo calibration using checkerboard pattern."""

    def __init__(self, camera_mgr):
        self._camera_mgr = camera_mgr
        self._lock = threading.Lock()

        # Checkerboard settings (inner corners)
        self.board_cols = 9
        self.board_rows = 6
        self.square_size = 23.2  # mm

        # Captured image pairs with detected corners
        self._pairs = []  # list of (corners_l, corners_r)
        self._preview_frame = None  # Latest preview with corners drawn
        self._last_capture_status = ""
        self._calibration_result = None
        self._is_calibrating = False

        # Criteria for cornerSubPix
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )

    def get_status(self):
        with self._lock:
            result_info = None
            if self._calibration_result is not None:
                r = self._calibration_result
                result_info = {
                    "rms": r["rms"],
                    "baseline_cm": r["baseline_cm"],
                    "focal_length": r["focal_length"],
                    "saved_to": r["saved_to"],
                }
            return {
                "board_cols": self.board_cols,
                "board_rows": self.board_rows,
                "square_size": self.square_size,
                "num_pairs": len(self._pairs),
                "last_status": self._last_capture_status,
                "is_calibrating": self._is_calibrating,
                "result": result_info,
            }

    def set_board_size(self, cols, rows, square_size):
        with self._lock:
            self.board_cols = int(cols)
            self.board_rows = int(rows)
            self.square_size = float(square_size)
            self._pairs = []
            self._calibration_result = None
            self._last_capture_status = (
                f"Board set to {self.board_cols}x{self.board_rows}, "
                f"square={self.square_size}mm. Pairs cleared."
            )

    def get_preview_frame(self):
        """Get latest camera frames with corner detection overlay."""
        ok, frame_l, frame_r = self._camera_mgr.read_both()
        if not ok:
            return None

        frame_l = frame_l.copy()
        frame_r = frame_r.copy()

        board_size = (self.board_cols, self.board_rows)
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(
            gray_l, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        found_r, corners_r = cv2.findChessboardCorners(
            gray_r, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        # Draw corners
        cv2.drawChessboardCorners(frame_l, board_size, corners_l, found_l)
        cv2.drawChessboardCorners(frame_r, board_size, corners_r, found_r)

        # Status indicator
        color_l = (0, 255, 0) if found_l else (0, 0, 255)
        color_r = (0, 255, 0) if found_r else (0, 0, 255)
        status_l = "DETECTED" if found_l else "NOT FOUND"
        status_r = "DETECTED" if found_r else "NOT FOUND"

        cv2.putText(frame_l, f"Cam0: {status_l}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_l, 2)
        cv2.putText(frame_r, f"Cam1: {status_r}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_r, 2)

        with self._lock:
            n = len(self._pairs)
        cv2.putText(frame_l, f"Pairs: {n}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Combine side by side
        h = min(frame_l.shape[0], frame_r.shape[0])
        w = min(frame_l.shape[1], frame_r.shape[1])
        combined = np.hstack([
            cv2.resize(frame_l, (w, h)),
            cv2.resize(frame_r, (w, h)),
        ])
        return combined

    def capture_pair(self):
        """Capture one calibration pair if corners found in both cameras."""
        ok, frame_l, frame_r = self._camera_mgr.read_both()
        if not ok:
            with self._lock:
                self._last_capture_status = "Failed: cannot read cameras"
            return False

        board_size = (self.board_cols, self.board_rows)
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(
            gray_l, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        found_r, corners_r = cv2.findChessboardCorners(
            gray_r, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not (found_l and found_r):
            status = "Failed: "
            if not found_l and not found_r:
                status += "corners not found in both cameras"
            elif not found_l:
                status += "corners not found in Camera 0"
            else:
                status += "corners not found in Camera 1"
            with self._lock:
                self._last_capture_status = status
            return False

        # Refine corners to sub-pixel accuracy
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1),
                                     self._criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                     self._criteria)

        with self._lock:
            self._pairs.append((corners_l, corners_r))
            n = len(self._pairs)
            self._last_capture_status = f"Captured pair #{n}"

        return True

    def delete_pair(self, index):
        """Delete a captured pair by index."""
        with self._lock:
            if 0 <= index < len(self._pairs):
                self._pairs.pop(index)
                self._last_capture_status = (
                    f"Deleted pair #{index + 1}. {len(self._pairs)} remaining."
                )
                return True
            return False

    def clear_pairs(self):
        with self._lock:
            self._pairs = []
            self._calibration_result = None
            self._last_capture_status = "All pairs cleared"

    def run_calibration(self):
        """Run stereo calibration on captured pairs. Blocking call."""
        with self._lock:
            if len(self._pairs) < 8:
                self._last_capture_status = (
                    f"Need at least 8 pairs, have {len(self._pairs)}"
                )
                return False
            self._is_calibrating = True
            self._last_capture_status = "Calibrating..."
            pairs = list(self._pairs)

        board_size = (self.board_cols, self.board_rows)
        img_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        # 3D object points
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:board_size[0], 0:board_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size / 1000.0  # mm -> meters

        obj_points = [objp] * len(pairs)
        # pairs[i] = (corners_cam0, corners_cam1)
        # L/R labels match USB index: L=USB0, R=USB1
        img_points_l = [p[0] for p in pairs]
        img_points_r = [p[1] for p in pairs]

        # Individual camera calibration first
        ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
            obj_points, img_points_l, img_size, None, None
        )
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
            obj_points, img_points_r, img_size, None, None
        )

        # Stereo calibration
        flags = (
            cv2.CALIB_FIX_INTRINSIC
        )
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6
        )

        rms, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_l, img_points_r,
            mtx_l, dist_l, mtx_r, dist_r, img_size,
            criteria=criteria, flags=flags,
        )

        baseline = float(np.linalg.norm(T))

        # Save calibration
        # mtx_l = cam0 (USB0), mtx_r = cam1 (USB1)
        output_path = config.STEREO_CALIB_FILE
        np.savez(
            output_path,
            mtx_l=mtx_l, dist_l=dist_l,
            mtx_r=mtx_r, dist_r=dist_r,
            R=R, T=T, E=E, F=F,
        )

        # Also update config dimensions
        result = {
            "rms": round(float(rms), 4),
            "baseline_cm": round(baseline * 100, 2),
            "focal_length": round(float(mtx_l[0, 0]), 1),
            "saved_to": output_path,
            "T": T.flatten().tolist(),
        }

        with self._lock:
            self._calibration_result = result
            self._is_calibrating = False
            self._last_capture_status = (
                f"Calibration done! RMS={result['rms']}, "
                f"Baseline={result['baseline_cm']}cm"
            )

        print(f"[Calibrator] Stereo calibration complete:")
        print(f"  RMS error: {result['rms']}")
        print(f"  Baseline: {result['baseline_cm']}cm")
        print(f"  Focal length: {result['focal_length']}px")
        print(f"  Saved to: {output_path}")

        return True
