"""
Benchmark - So sánh tốc độ PyTorch vs TensorRT trên Jetson Nano.

Cách dùng:
    python3 benchmark.py
"""

import time
import numpy as np
import cv2
import config


def benchmark_backend(estimator, name, num_warmup=5, num_runs=50):
    """Benchmark một estimator."""
    # Tạo dummy frame
    dummy = np.random.randint(0, 255,
        (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)

    # Warmup
    print(f"\n[{name}] Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        estimator.estimate(dummy)

    # Benchmark
    print(f"[{name}] Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        t0 = time.time()
        estimator.estimate(dummy)
        elapsed = time.time() - t0
        times.append(elapsed)

    times = np.array(times)
    fps = 1.0 / times.mean()

    print(f"[{name}] Results:")
    print(f"  Mean:   {times.mean() * 1000:.1f} ms  ({fps:.1f} FPS)")
    print(f"  Median: {np.median(times) * 1000:.1f} ms")
    print(f"  Min:    {times.min() * 1000:.1f} ms")
    print(f"  Max:    {times.max() * 1000:.1f} ms")
    print(f"  Std:    {times.std() * 1000:.1f} ms")

    return times.mean(), fps


def main():
    print("=" * 60)
    print("  Depth Anything V2 - Benchmark")
    print(f"  Input: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    print(f"  Model input: {config.MODEL_INPUT_SIZE}x{config.MODEL_INPUT_SIZE}")
    print("=" * 60)

    results = {}

    # TensorRT benchmark
    try:
        from trt_depth_estimator import TensorRTDepthEstimator
        trt_est = TensorRTDepthEstimator()
        mean_t, fps = benchmark_backend(trt_est, "TensorRT")
        results["TensorRT"] = (mean_t, fps)
        del trt_est
    except Exception as e:
        print(f"\n[TensorRT] Skipped: {e}")

    # PyTorch benchmark
    try:
        from depth_estimator import DepthAnythingV2Estimator
        pt_est = DepthAnythingV2Estimator()
        mean_t, fps = benchmark_backend(pt_est, "PyTorch")
        results["PyTorch"] = (mean_t, fps)
        del pt_est
    except Exception as e:
        print(f"\n[PyTorch] Skipped: {e}")

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        for name, (mean_t, fps) in results.items():
            print(f"  {name:12s}: {mean_t * 1000:7.1f} ms  ({fps:.1f} FPS)")

        if "TensorRT" in results and "PyTorch" in results:
            speedup = results["PyTorch"][0] / results["TensorRT"][0]
            print(f"\n  TensorRT speedup: {speedup:.2f}x faster")
        print("=" * 60)


if __name__ == "__main__":
    main()
