import argparse
import timeit

import numpy as np
import torch
import tensorax as ts
import tensorax.functional as F
from tensorax import _C


_ALL_NAMES = ["default", "shared_memory_coalesced", "cache_blocking", "tiled",
              "block_tiling_1d", "block_tiling_2d", "mma",
              "cute_fp16", "cute_fp16_c4", "cute_fp16_pp", "cute_fp16_t256",
              "numpy", "pytorch", "pytorch_fp16", "pytorch_compile_fp16"]

_parser = argparse.ArgumentParser(
    description="matmul benchmark. By default runs every variant; use --only to "
                "run just a subset.")
_parser.add_argument("--only", nargs="+", choices=_ALL_NAMES, metavar="NAME",
                     help=f"benchmarks to run. Choices: {', '.join(_ALL_NAMES)}")
_parser.add_argument("--list", action="store_true",
                     help="print available benchmark names and exit")
_parser.add_argument("--times", "-n", type=int, default=100,
                     help="iterations per benchmark (default: 100)")
_parser.add_argument("--batch", type=int, default=3)
_parser.add_argument("--M", type=int, default=1024)
_parser.add_argument("--K", type=int, default=1024)
_parser.add_argument("--N", type=int, default=1024)
_parser.add_argument("--quiet", action="store_true",
                     help="suppress version/header output")
_args = _parser.parse_args()

if _args.list:
    print("Available benchmarks:")
    for n in _ALL_NAMES:
        print(f"  {n}")
    raise SystemExit(0)

_selected = set(_args.only) if _args.only else set(_ALL_NAMES)

times = _args.times
batch, M, K, N = _args.batch, _args.M, _args.K, _args.N

a_torch = torch.randn((batch, M, K), device='cuda', dtype=torch.float32)
b_torch = torch.randn((batch, K, N), device='cuda', dtype=torch.float32)

a_torch_fp16 = a_torch.half()
b_torch_fp16 = b_torch.half()

a_t = ts.Tensor(a_torch.cpu().numpy(), dtype='float32', device='cuda')
b_t = ts.Tensor(b_torch.cpu().numpy(), dtype='float32', device='cuda')

a_h = F.cast_to_fp16(a_t)
b_h = F.cast_to_fp16(b_t)

a_np = a_torch.cpu().numpy()
b_np = b_torch.cpu().numpy()

def matmul_default():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="default")
    torch.cuda.synchronize()
    return c

def matmul_shared_memory_coalesced():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="shared_memory_coalesced")
    torch.cuda.synchronize()
    return c

def matmul_cache_blocking():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="shared_memory_cache_blocking")
    torch.cuda.synchronize()
    return c

def matmul_tiled():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="tiled")
    torch.cuda.synchronize()
    return c

def matmul_block_tiling_1d():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="block_tiling_1d")
    torch.cuda.synchronize()
    return c

def matmul_block_tiling_2d():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="block_tiling_2d")
    torch.cuda.synchronize()
    return c

def matmul_mma():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="mma")
    torch.cuda.synchronize()
    return c

def matmul_cute_fp16():
    torch.cuda.synchronize()
    c = a_h.matmul(b_h, method="cute_fp16")
    torch.cuda.synchronize()
    return c

def matmul_cute_fp16_c4():
    torch.cuda.synchronize()
    c = a_h.matmul(b_h, method="cute_fp16_c4")
    torch.cuda.synchronize()
    return c

def matmul_cute_fp16_pp():
    torch.cuda.synchronize()
    c = a_h.matmul(b_h, method="cute_fp16_pp")
    torch.cuda.synchronize()
    return c

def matmul_cute_fp16_t256():
    torch.cuda.synchronize()
    c = a_h.matmul(b_h, method="cute_fp16_t256")
    torch.cuda.synchronize()
    return c

def matmul_numpy():
    return np.matmul(a_np, b_np)

def matmul_pytorch():
    torch.cuda.synchronize()
    c = torch.matmul(a_torch, b_torch)
    torch.cuda.synchronize()
    return c

def matmul_pytorch_fp16():
    torch.cuda.synchronize()
    c = torch.matmul(a_torch_fp16, b_torch_fp16)
    torch.cuda.synchronize()
    return c

_matmul_compiled_fp16 = torch.compile(torch.matmul)

def matmul_pytorch_compile_fp16():
    torch.cuda.synchronize()
    c = _matmul_compiled_fp16(a_torch_fp16, b_torch_fp16)
    torch.cuda.synchronize()
    return c

def compute_tflops(time_sec, batch, M, K, N, times):
    total_flops = 2 * batch * M * N * K
    return (total_flops * times) / (time_sec * 1e12)

_BENCHMARKS = {
    "default":                 ("Default matmul",                       matmul_default),
    "shared_memory_coalesced": ("Matmul with shared memory coalescing", matmul_shared_memory_coalesced),
    "cache_blocking":          ("Matmul with shared memory cache blocking", matmul_cache_blocking),
    "tiled":                   ("Tiled matmul",                         matmul_tiled),
    "block_tiling_1d":         ("Matmul with 1D block tiling",          matmul_block_tiling_1d),
    "block_tiling_2d":         ("Matmul with 2D block tiling",          matmul_block_tiling_2d),
    "mma":                     ("Matmul with MMA TF32 (Tensor Cores)",  matmul_mma),
    "cute_fp16":               ("Matmul CUTLASS Hopper (fp16, default)", matmul_cute_fp16),
    "cute_fp16_c4":            ("Matmul CUTLASS Hopper (fp16, Cluster<4,1,1>)", matmul_cute_fp16_c4),
    "cute_fp16_pp":            ("Matmul CUTLASS Hopper (fp16, Pingpong)", matmul_cute_fp16_pp),
    "cute_fp16_t256":          ("Matmul CUTLASS Hopper (fp16, Tile<128,256,64>)", matmul_cute_fp16_t256),
    "numpy":                   ("Numpy matmul",                         matmul_numpy),
    "pytorch":                 ("PyTorch matmul",                       matmul_pytorch),
    "pytorch_fp16":            ("PyTorch matmul fp16 (cuBLAS)",         matmul_pytorch_fp16),
    "pytorch_compile_fp16":    ("PyTorch torch.compile matmul fp16",    matmul_pytorch_compile_fp16),
}

def _accuracy_check_cute_fp16(label: str, method: str) -> bool:
    if M % 8 != 0 or N % 8 != 0 or K % 8 != 0:
        print(f"[{label}] SKIP accuracy check: requires M, N, K all multiples of 8 (got M={M}, N={N}, K={K})")
        return False
    try:
        out_ts = a_h.matmul(b_h, method=method)
        out_fp32 = F.cast_to_fp32(out_ts)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[{label}] SKIP accuracy check: kernel raised: {e}")
        return False
    out_np = np.asarray(_C.tensor_to_list(out_fp32._c_tensor), dtype=np.float32).reshape(out_fp32.shape)
    ref = torch.matmul(a_torch_fp16, b_torch_fp16).float().cpu().numpy()
    ok = np.allclose(out_np, ref, rtol=5e-3, atol=5e-3)
    if ok:
        print(f"[{label}] accuracy check PASS (vs torch matmul fp16)")
    else:
        diff = np.abs(out_np - ref)
        print(f"[{label}] accuracy check FAIL: max_abs={diff.max():.4e} mean_abs={diff.mean():.4e}")
    return ok

for _name, _method in (
    ("cute_fp16",      "cute_fp16"),
    ("cute_fp16_c4",   "cute_fp16_c4"),
    ("cute_fp16_pp",   "cute_fp16_pp"),
    ("cute_fp16_t256", "cute_fp16_t256"),
):
    if _name in _selected and not _accuracy_check_cute_fp16(_name, _method):
        print(f"[{_name}] dropping from benchmark set due to skipped/failed accuracy check")
        _selected.discard(_name)

if not _args.quiet:
    print("Warming up...")
for _name in _ALL_NAMES:
    if _name in _selected:
        _BENCHMARKS[_name][1]()
if not _args.quiet:
    print("Warm-up done.")

print(f"Starting benchmarks... (B={batch}, M={M}, K={K}, N={N})")

for _name in _ALL_NAMES:
    if _name not in _selected:
        continue
    _label, _fn = _BENCHMARKS[_name]
    _t = timeit.timeit(_fn, number=times)
    _tflops = compute_tflops(_t, batch, M, K, N, times)
    print(f"{_label} time over {times} runs: {_t} seconds | "
          f"Time per run: {_t / times:.4f} seconds | TFLOPS: {_tflops:.2f}")
