"""
Fix ONNX model cho Jetson Nano compatibility.

Vấn đề:
- ONNX IR version 10 không tương thích onnxruntime 1.16.3 (max IR=9)
- External data (.onnx.data) cần merge vào 1 file duy nhất
- Opset 18 Resize attributes không tương thích opset 17
- LayerNormalization không được TensorRT 8.0.1 hỗ trợ

Script này:
1. Load ONNX model (bao gồm external data)
2. Downgrade IR version về 8
3. Downgrade opset 18 -> 17, fix Resize attributes
4. Decompose LayerNormalization thành basic ops (cho raw TensorRT)
5. Merge tất cả weights vào 1 file .onnx duy nhất

Cách dùng:
    python3 fix_onnx_model.py
    python3 fix_onnx_model.py --input depth_anything_v2_vits.onnx --output depth_anything_v2_vits_fixed.onnx
"""

import argparse
import os
import sys
import struct

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np
except ImportError:
    print("Cần cài onnx và numpy: pip3 install onnx numpy")
    sys.exit(1)


def _decompose_layer_norm(model):
    """
    Decompose LayerNormalization thành basic ops cho TensorRT 8.0.1.

    LayerNorm(X, scale, bias, epsilon, axis) =
        mean = ReduceMean(X, axes=[axis:])
        x_centered = X - mean
        variance = ReduceMean(x_centered^2, axes=[axis:])
        x_norm = x_centered / sqrt(variance + epsilon)
        output = scale * x_norm + bias
    """
    graph = model.graph
    nodes = list(graph.node)
    new_nodes = []
    new_initializers = []
    ln_count = 0

    for node in nodes:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue

        ln_count += 1
        prefix = f"ln_decomp_{ln_count}"

        # Get LayerNorm attributes
        epsilon = 1e-5
        axis = -1
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = attr.f
            elif attr.name == "axis":
                axis = attr.i

        # Inputs: X, scale, bias (bias is optional)
        x_input = node.input[0]
        scale_input = node.input[1]
        bias_input = node.input[2] if len(node.input) > 2 else None

        # Output
        output_name = node.output[0]

        # Create epsilon constant
        eps_name = f"{prefix}_epsilon"
        eps_tensor = numpy_helper.from_array(
            np.array([epsilon], dtype=np.float32), eps_name
        )
        new_initializers.append(eps_tensor)

        # Node 1: mean = ReduceMean(X, axes=[axis]) - opset 17 uses axes as attribute
        mean_name = f"{prefix}_mean"
        new_nodes.append(helper.make_node(
            "ReduceMean", [x_input], [mean_name],
            name=f"{prefix}_reducemean", keepdims=1, axes=[axis]
        ))

        # Node 2: x_centered = Sub(X, mean)
        centered_name = f"{prefix}_centered"
        new_nodes.append(helper.make_node(
            "Sub", [x_input, mean_name], [centered_name],
            name=f"{prefix}_sub"
        ))

        # Node 3: centered_sq = Mul(x_centered, x_centered)
        centered_sq_name = f"{prefix}_centered_sq"
        new_nodes.append(helper.make_node(
            "Mul", [centered_name, centered_name], [centered_sq_name],
            name=f"{prefix}_mul_sq"
        ))

        # Node 4: variance = ReduceMean(centered_sq, axes=[axis])
        variance_name = f"{prefix}_variance"
        new_nodes.append(helper.make_node(
            "ReduceMean", [centered_sq_name], [variance_name],
            name=f"{prefix}_reducemean_var", keepdims=1, axes=[axis]
        ))

        # Node 5: var_eps = Add(variance, epsilon)
        var_eps_name = f"{prefix}_var_eps"
        new_nodes.append(helper.make_node(
            "Add", [variance_name, eps_name], [var_eps_name],
            name=f"{prefix}_add_eps"
        ))

        # Node 6: std = Sqrt(var_eps)
        std_name = f"{prefix}_std"
        new_nodes.append(helper.make_node(
            "Sqrt", [var_eps_name], [std_name],
            name=f"{prefix}_sqrt"
        ))

        # Node 7: x_norm = Div(x_centered, std)
        norm_name = f"{prefix}_norm"
        new_nodes.append(helper.make_node(
            "Div", [centered_name, std_name], [norm_name],
            name=f"{prefix}_div"
        ))

        # Node 8: scaled = Mul(scale, x_norm)
        if bias_input:
            scaled_name = f"{prefix}_scaled"
        else:
            scaled_name = output_name
        new_nodes.append(helper.make_node(
            "Mul", [scale_input, norm_name], [scaled_name],
            name=f"{prefix}_mul_scale"
        ))

        # Node 9: output = Add(scaled, bias) [if bias exists]
        if bias_input:
            new_nodes.append(helper.make_node(
                "Add", [scaled_name, bias_input], [output_name],
                name=f"{prefix}_add_bias"
            ))

    # Replace nodes and add new initializers
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)

    return ln_count


def fix_onnx_model(input_path, output_path):
    """Fix ONNX model cho Jetson Nano."""
    print(f"[1/4] Loading model: {input_path}")

    # Load model bao gồm external data
    model = onnx.load(input_path, load_external_data=True)

    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {[(o.domain or 'default', o.version) for o in model.opset_import]}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Weights: {len(model.graph.initializer)}")

    inputs = model.graph.input
    outputs = model.graph.output
    if inputs:
        inp = inputs[0]
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name} {shape}")
    if outputs:
        out = outputs[0]
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {shape}")

    # Fix 1: Downgrade IR version
    print(f"\n[2/4] Downgrading IR version: {model.ir_version} -> 8")
    model.ir_version = 8

    # Fix 2: Downgrade opset nếu cần (opset 18 -> 17 là safe cho LayerNorm)
    for opset in model.opset_import:
        if (opset.domain == "" or opset.domain == "ai.onnx") and opset.version > 17:
            print(f"  Downgrading opset: {opset.version} -> 17")
            opset.version = 17

    # Fix 3: Remove unsupported attributes sau khi downgrade opset
    print(f"\n[3/5] Fixing op attributes for opset 17...")
    fixed_nodes = 0
    # Opset 18 Resize thêm 'antialias' và 'keep_aspect_ratio_policy'
    resize_attrs_to_remove = {"antialias", "keep_aspect_ratio_policy"}
    cubic_fixed = 0
    for node in model.graph.node:
        if node.op_type == "Resize":
            attrs_to_remove = [a for a in node.attribute if a.name in resize_attrs_to_remove]
            for attr in attrs_to_remove:
                node.attribute.remove(attr)
                fixed_nodes += 1
            # TensorRT 8.0.1 không hỗ trợ cubic interpolation, đổi sang linear
            for attr in node.attribute:
                if attr.name == "mode" and attr.s == b"cubic":
                    attr.s = b"linear"
                    cubic_fixed += 1
    if fixed_nodes:
        print(f"  Removed {fixed_nodes} unsupported attribute(s) from Resize nodes")
    if cubic_fixed:
        print(f"  Changed {cubic_fixed} Resize node(s) from cubic -> linear (TRT 8.0.1 compat)")
    if not fixed_nodes and not cubic_fixed:
        print(f"  No Resize fixes needed")

    # Fix 4: Decompose LayerNormalization cho TensorRT 8.0.1
    ln_count = sum(1 for n in model.graph.node if n.op_type == "LayerNormalization")
    print(f"\n[4/6] Decomposing LayerNormalization ({ln_count} nodes)...")
    if ln_count > 0:
        decomposed = _decompose_layer_norm(model)
        print(f"  Decomposed {decomposed} LayerNorm nodes")
        print(f"  Total nodes after: {len(model.graph.node)}")
    else:
        print("  No LayerNormalization nodes found")

    # Fix 5: Đảm bảo tất cả weights có raw_data (không dùng external)
    print(f"\n[5/6] Merging external weights into single file...")
    total_weight_bytes = 0
    for init in model.graph.initializer:
        if init.raw_data:
            total_weight_bytes += len(init.raw_data)
        # Clear external data location flag
        if init.HasField("data_location"):
            init.ClearField("data_location")
        # Clear external data info
        while len(init.external_data) > 0:
            init.external_data.pop()

    print(f"  Total weight data: {total_weight_bytes / 1024 / 1024:.1f} MB")

    # Fix 6: Save fixed model (single file, no external data)
    print(f"\n[6/6] Saving fixed model: {output_path}")

    # Phải dùng save thay vì save_model để tránh protobuf 2GB limit
    if total_weight_bytes > 1.5 * 1024 * 1024 * 1024:
        print("  WARNING: Model > 1.5GB, saving with external data")
        onnx.save_model(
            model, output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_path) + ".data",
        )
    else:
        onnx.save(model, output_path)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    # Verify
    print(f"\n  Verifying fixed model...")
    model2 = onnx.load(output_path)
    print(f"  IR version: {model2.ir_version}")
    print(f"  Opset: {[(o.domain or 'default', o.version) for o in model2.opset_import]}")
    print(f"  Nodes: {len(model2.graph.node)}")
    print(f"  Weights: {len(model2.graph.initializer)}")

    print(f"\n  DONE! Fixed model saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fix ONNX model for Jetson Nano")
    parser.add_argument("--input", type=str, default="depth_anything_v2_vits.onnx",
                        help="Input ONNX model path")
    parser.add_argument("--output", type=str, default="depth_anything_v2_vits_fixed.onnx",
                        help="Output fixed ONNX model path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Không tìm thấy {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  Fix ONNX Model cho Jetson Nano")
    print("=" * 60)

    fix_onnx_model(args.input, args.output)

    print("\n" + "=" * 60)
    print("  Bước tiếp theo:")
    print(f"  1. Cập nhật config.py: ONNX_MODEL = \"{args.output}\"")
    print(f"  2. Test: python3 test_trt.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
