"""
Export the metric-depth checkpoint directly to ONNX at opset 13.

At opset 13:
  - nn.LayerNorm  → primitive ReduceMean / Sub / Mul / Sqrt / Div  (no fused op)
  - ReduceMean    → axes as attribute  (not as an input tensor like opset 18)
  - F.interpolate → Resize without 'antialias' / 'keep_aspect_ratio_policy'

All of these are fully supported by TensorRT 8.0, so no post-processing of
the ONNX graph is required to build the engine.

Usage:
    python3 export_onnx.py [--size 308] [--out hypersim_vits_op13.onnx]
"""

import argparse
import os
import sys
import torch

# ------------------------------------------------------------------
# Add the metric_depth folder to sys.path so we can import its copy of
# depth_anything_v2 (the one that has max_depth support).
# ------------------------------------------------------------------
METRIC_DIR = os.path.join(os.path.dirname(__file__),
                          "depth_anything_v2", "metric_depth")
if METRIC_DIR not in sys.path:
    sys.path.insert(0, METRIC_DIR)

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _round_to_multiple(val: int, multiple: int = 14) -> int:
    """Round val UP to the nearest multiple of `multiple`."""
    return ((val + multiple - 1) // multiple) * multiple


def compute_input_shape(cam_w: int, cam_h: int, size: int) -> tuple:
    """Return (H, W) for the TRT export that matches keep_aspect_ratio+lower_bound
    preprocessing used by DepthAnythingV2.image2tensor().
    The shorter dimension is brought to `size` (multiple of 14), the longer
    dimension is scaled proportionally and ceiled to the next multiple of 14.
    """
    scale = size / min(cam_w, cam_h)
    export_w = _round_to_multiple(int(cam_w * scale))
    export_h = _round_to_multiple(int(cam_h * scale))
    return export_h, export_w   # (H, W) — pytorch convention


def export(checkpoint: str, out_onnx: str, export_h: int, export_w: int, max_depth: float, encoder: str):
    assert export_h % 14 == 0, f"export_h must be multiple of 14 (got {export_h})"
    assert export_w % 14 == 0, f"export_w must be multiple of 14 (got {export_w})"

    print(f"[export] Loading checkpoint: {checkpoint}")
    model = DepthAnythingV2(**{**MODEL_CONFIGS[encoder], "max_depth": max_depth})
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Export on CPU to avoid device-mismatch in ONNX constant folding
    device = "cpu"
    model = model.to(device)
    print(f"[export] Running on: {device}")

    dummy = torch.zeros(1, 3, export_h, export_w, device=device)

    print(f"[export] Exporting to ONNX opset 13 → {out_onnx}  (H={export_h} W={export_w})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_onnx,
            opset_version=13,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes=None,   # fully static — best for TRT on Jetson
            do_constant_folding=True,
            verbose=False,
        )

    size_mb = os.path.getsize(out_onnx) / 1024 / 1024
    print(f"[export] Done — {out_onnx}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description="Export DepthAnythingV2 checkpoint to ONNX at opset 13.\n\n"
                    "For a 640×480 camera, the correct export size is:\n"
                    "  H=308, W=420  (matches keep_aspect_ratio+lower_bound preprocessing)\n"
                    "This is computed automatically from --cam-width/--cam-height."
    )
    ap.add_argument("--checkpoint",
                    default="checkpoints/depth_anything_v2_metric_hypersim_vits.pth")
    ap.add_argument("--out", default="hypersim_vits_op13.onnx")
    ap.add_argument("--size", type=int, default=308,
                    help="Shorter-side target size (multiple of 14). Default: 308")
    ap.add_argument("--cam-width",  type=int, default=640, help="Camera frame width")
    ap.add_argument("--cam-height", type=int, default=480, help="Camera frame height")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[export] ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    export_h, export_w = compute_input_shape(args.cam_width, args.cam_height, args.size)
    print(f"[export] Camera {args.cam_width}×{args.cam_height} → export shape H={export_h} W={export_w}")

    export(args.checkpoint, args.out, export_h, export_w, args.max_depth, args.encoder)


if __name__ == "__main__":
    main()
