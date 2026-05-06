import argparse
import timeit

import numpy as np
import torch
import tensorax as ts
import tensorax.functional as F


_ALL_NAMES = ["naive", "tiled", "flash", "mma", "mma_fp16",
              "flash_optimized", "numpy", "pytorch"]

_parser = argparse.ArgumentParser(
    description="SDPA benchmark. By default runs every variant; use --only to "
                "run just a subset.")
_parser.add_argument("--only", nargs="+", choices=_ALL_NAMES, metavar="NAME",
                     help=f"benchmarks to run. Choices: {', '.join(_ALL_NAMES)}")
_parser.add_argument("--list", action="store_true",
                     help="print available benchmark names and exit")
_parser.add_argument("--times", "-n", type=int, default=30,
                     help="iterations per benchmark (default: 30)")
_parser.add_argument("--batch",   type=int, default=4)
_parser.add_argument("--heads",   type=int, default=8)
_parser.add_argument("--seq_len", type=int, default=256)
_parser.add_argument("--d_k",     type=int, default=512)
_parser.add_argument("--d_v",     type=int, default=512)
_parser.add_argument("--quiet", action="store_true",
                     help="suppress version/header output")
_args = _parser.parse_args()

if _args.list:
    print("Available benchmarks:")
    for n in _ALL_NAMES:
        print(f"  {n}")
    raise SystemExit(0)

_selected = set(_args.only) if _args.only else set(_ALL_NAMES)

if not _args.quiet:
    print("Tensorax version:", ts.__version__)
    print("CUDA available:", ts.cuda_is_available())
    print()
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA available:", torch.cuda.is_available())
    print()

times = _args.times

batch, heads, seq_len, d_k, d_v = _args.batch, _args.heads, _args.seq_len, _args.d_k, _args.d_v

q_torch = torch.randn((batch, heads, seq_len, d_k), device='cuda', dtype=torch.float32)
k_torch = torch.randn((batch, heads, seq_len, d_k), device='cuda', dtype=torch.float32)
v_torch = torch.randn((batch, heads, seq_len, d_v), device='cuda', dtype=torch.float32)

q_t = ts.Tensor(q_torch.cpu().numpy(), dtype='float32', device='cuda')
k_t = ts.Tensor(k_torch.cpu().numpy(), dtype='float32', device='cuda')
v_t = ts.Tensor(v_torch.cpu().numpy(), dtype='float32', device='cuda')

q_np = q_torch.cpu().numpy()
k_np = k_torch.cpu().numpy()
v_np = v_torch.cpu().numpy()

# Benchmarking naive SDPA
def sdpa_naive():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking tiled SDPA
def sdpa_tiled():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_tiled(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking flash SDPA
def sdpa_flash():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_flash(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking MMA SDPA
def sdpa_mma():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_mma(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

q_h = F.cast_to_fp16(q_t)
k_h = F.cast_to_fp16(k_t)
v_h = F.cast_to_fp16(v_t)

def sdpa_mma_fp16():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_mma_fp16(q_h, k_h, v_h)
    torch.cuda.synchronize()
    return c

# Benchmarking optimized flash SDPA
def sdpa_flash_optimized():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_flash_optimized(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking NumPy reference SDPA
def sdpa_numpy():
    scores = np.matmul(q_np, k_np.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    c = np.matmul(attn, v_np)
    return c

# Benchmarking PyTorch SDPA
def sdpa_pytorch():
    torch.cuda.synchronize()
    c = torch.nn.functional.scaled_dot_product_attention(q_torch, k_torch, v_torch)
    torch.cuda.synchronize()
    return c

def compute_tflops(time_sec, batch, heads, seq_len, d_k, d_v, times):
    # Total FLOPs for one SDPA run: 4 * B * H * S^2 * Dk + 2 * B * H * S^2 * Dv
    total_flops = (4 * batch * heads * seq_len**2 * d_k) + (2 * batch * heads * seq_len**2 * d_v)
    tflops = (total_flops * times) / (time_sec * 1e12)
    return tflops

_BENCHMARKS = {
    "naive":           ("Naive SDPA",           sdpa_naive),
    "tiled":           ("Tiled SDPA",           sdpa_tiled),
    "flash":           ("Flash SDPA",           sdpa_flash),
    "flash_optimized": ("Optimized Flash SDPA", sdpa_flash_optimized),
    "mma":             ("MMA SDPA",             sdpa_mma),
    "mma_fp16":        ("MMA SDPA fp16",        sdpa_mma_fp16),
    "numpy":           ("Numpy SDPA",           sdpa_numpy),
    "pytorch":         ("PyTorch SDPA",         sdpa_pytorch),
}

if not _args.quiet:
    print("Warming up...")
for _name in _ALL_NAMES:
    if _name in _selected:
        _BENCHMARKS[_name][1]()
if not _args.quiet:
    print("Warm-up done.")

print(f"Starting benchmarks... (B={batch}, H={heads}, S={seq_len}, Dk={d_k}, Dv={d_v})")

for _name in _ALL_NAMES:
    if _name not in _selected:
        continue
    _label, _fn = _BENCHMARKS[_name]
    _t = timeit.timeit(_fn, number=times)
    _tflops = compute_tflops(_t, batch, heads, seq_len, d_k, d_v, times)
    print(f"{_label} time over {times} runs: {_t} seconds | "
          f"Time per run: {_t / times:.4f} seconds | TFLOPS: {_tflops:.2f}")
