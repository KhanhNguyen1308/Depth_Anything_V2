"""
Convert Depth Anything V2 ONNX model to TensorRT engine.
Run once on the Jetson before switching INFERENCE_BACKEND to "tensorrt".

Usage:
    python3 convert_to_trt.py [--no-fp16]
"""

import os
import sys
import argparse
import tensorrt as trt
import onnx
import numpy as np
from onnx import version_converter
from onnx.external_data_helper import load_external_data_for_model

import config

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path: str, engine_path: str, workspace_mb: int, fp16: bool) -> None:
    print(f"[TRT] Loading ONNX: {onnx_path}")
    print(f"[TRT] TensorRT version: {trt.__version__}")

    # Step 1 – load ONNX + resolve external data sidecar
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    model = onnx.load(onnx_path, load_external_data=False)

    data_file = None
    for fname in os.listdir(onnx_dir):
        candidate = os.path.join(onnx_dir, fname)
        if fname != os.path.basename(onnx_path) and os.path.isfile(candidate) and fname.endswith(".data"):
            data_file = fname
            break

    if data_file is not None:
        print(f"[TRT] Patching external data references → {data_file}")
        for initializer in model.graph.initializer:
            for entry in initializer.external_data:
                if entry.key == "location":
                    entry.value = data_file
        load_external_data_for_model(model, onnx_dir)

    # Step 2 – opset check.  export_onnx.py produces opset 13 directly so no
    # conversion is needed.  For legacy opset-17/18 models, use version_converter.
    src_opset = model.opset_import[0].version
    if src_opset != 13:
        print(f"[TRT] Converting opset {src_opset} → 13 ...")
        model = version_converter.convert_version(model, 13)
        print("[TRT] Opset conversion done")
    else:
        print(f"[TRT] ONNX already at opset 13, skipping conversion")

    # Step 3 – fix cubic Resize (not supported in TRT 8.0)
    patch_count = 0
    for node in model.graph.node:
        if node.op_type == "Resize":
            for attr in node.attribute:
                if attr.name == "mode" and attr.s == b"cubic":
                    attr.s = b"linear"
                    patch_count += 1
    if patch_count:
        print(f"[TRT] Patched {patch_count} Resize node(s): cubic → linear")

    # Step 4 – write single-file ONNX for TRT parser
    tmp_onnx = onnx_path + ".merged.tmp"
    print(f"[TRT] Writing converted ONNX: {tmp_onnx}")
    onnx.save(model, tmp_onnx)
    del model

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(tmp_onnx, "rb") as f:
        data = f.read()

    try:
        os.remove(tmp_onnx)
    except OSError:
        pass

    if not parser.parse(data):
        print("[TRT] ERROR: Failed to parse ONNX model")
        for i in range(parser.num_errors):
            print(f"  [{i}] {parser.get_error(i)}")
        sys.exit(1)

    print(f"[TRT] Network inputs : {network.num_inputs}")
    print(f"[TRT] Network outputs: {network.num_outputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input [{i}]: {inp.name}  shape={inp.shape}  dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output[{i}]: {out.name}  shape={out.shape}  dtype={out.dtype}")

    builder_config = builder.create_builder_config()
    builder_config.max_workspace_size = workspace_mb * 1024 * 1024

    if fp16:
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 mode enabled")
        else:
            print("[TRT] WARNING: platform does not support fast FP16, building FP32")

    print("[TRT] Building engine... (this may take several minutes on Jetson Nano)")
    engine = builder.build_engine(network, builder_config)

    if engine is None:
        print("[TRT] ERROR: Engine build failed")
        sys.exit(1)

    print(f"[TRT] Serializing engine to: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"[TRT] Done. Engine size: {os.path.getsize(engine_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT conversion for Depth Anything V2")
    parser.add_argument("--onnx", default=config.ONNX_MODEL,
                        help=f"Input ONNX file (default: {config.ONNX_MODEL})")
    parser.add_argument("--engine", default=config.TENSORRT_ENGINE,
                        help=f"Output engine file (default: {config.TENSORRT_ENGINE})")
    parser.add_argument("--workspace", type=int, default=config.TENSORRT_WORKSPACE_MB,
                        help=f"Max workspace MB (default: {config.TENSORRT_WORKSPACE_MB})")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 (build FP32 engine)")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"[TRT] ERROR: ONNX file not found: {args.onnx}")
        sys.exit(1)

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        workspace_mb=args.workspace,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
