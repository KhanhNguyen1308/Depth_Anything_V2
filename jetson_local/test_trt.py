"""Standalone TRT smoke test — run with: python3 test_trt.py"""
import tensorrt as trt
import ctypes, numpy as np, torch

print("PyTorch CUDA init...")
torch.cuda.synchronize()
print("done")

lib = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so.10.2')

# --- cudaMalloc/cudaMemcpy round-trip ---
ptr = ctypes.c_void_p()
lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))
val = np.array([42.0], dtype=np.float32)
out = np.zeros(1, dtype=np.float32)
lib.cudaMemcpy(ptr, val.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(4), ctypes.c_int(1))
lib.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), ptr, ctypes.c_size_t(4), ctypes.c_int(2))
print(f"cudaMemcpy roundtrip: {out[0]}  (expected 42.0)")

# --- Engine test ---
trt_logger = trt.Logger(trt.Logger.WARNING)
rt = trt.Runtime(trt_logger)
with open('hypersim_vits_op13.engine', 'rb') as f:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

shapes = []
for i in range(engine.num_bindings):
    shapes.append(tuple(engine.get_binding_shape(i)))
    print(i, engine.get_binding_name(i), shapes[-1])

sz_in  = int(np.prod(shapes[0])) * 4
sz_out = int(np.prod(shapes[1])) * 4
inp  = ctypes.c_void_p(); lib.cudaMalloc(ctypes.byref(inp),  ctypes.c_size_t(sz_in))
outp = ctypes.c_void_p(); lib.cudaMalloc(ctypes.byref(outp), ctypes.c_size_t(sz_out))

data = np.random.randn(*shapes[0]).astype(np.float32)
lib.cudaMemcpy(inp, data.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(sz_in), ctypes.c_int(1))
lib.cudaDeviceSynchronize()

ok = ctx.execute_v2([int(inp.value), int(outp.value)])
print('execute_v2 ok:', ok)
lib.cudaDeviceSynchronize()

result_h = np.zeros(shapes[1], dtype=np.float32)
lib.cudaMemcpy(result_h.ctypes.data_as(ctypes.c_void_p), outp, ctypes.c_size_t(sz_out), ctypes.c_int(2))
print(f"output min={result_h.min():.4f} max={result_h.max():.4f} mean={result_h.mean():.4f}")
