# Tensorax

Tensorax is a deep learning framework written from scratch in C++/CUDA with a Python frontend. Every kernel — matmul, attention, elementwise ops, reductions — is hand-written. No PyTorch, no NumPy, no cuBLAS at runtime. The only dependency is `pybind11` for the C++/Python bridge.

The goal is a clean, readable implementation of a DL framework from first principles that also runs fast on real hardware. Both the MMA attention kernel and the MMA matmul kernel use inline PTX assembly to hit Ampere Tensor Cores, and the best matmul variant runs at ~4x NumPy speed — all without calling into any external math library.

[![PyPI](https://img.shields.io/pypi/v/tensorax.svg?style=flat-square&color=blueviolet)](https://pypi.org/project/tensorax/)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/personalized-badge/tensorax?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=orange&left_text=downloads&style=flat-square)](https://pepy.tech/projects/tensorax)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/github/actions/workflow/status/NotShrirang/tensorax/tests.yml?label=tests&style=flat-square)](https://github.com/NotShrirang/tensorax/actions/workflows/tests.yml)

## Quick start

```bash
pip install tensorax
```

The API is intentionally PyTorch-like, so the learning curve is minimal:

```python
from tensorax import Tensor, nn, optim, lr_scheduler, functional as F

# define a model
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.GELU(),
    nn.LayerNorm(8),
    nn.Linear(8, 3),
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# train
for epoch in range(100):
    loss = F.mse_loss(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

More examples in [`examples/`](examples/) and the full API reference in [`docs/USAGE.md`](docs/USAGE.md).

## What's implemented

**Tensor core.** CPU and CUDA backends with automatic fallback. Broadcasting arithmetic, `reshape`, `transpose`, `sum`, `mean`, `exp`, `log`, `sqrt`, `pow`. Reverse-mode autograd through 18+ operations. 13 dtype constants.

**Layers.** `Linear`, `Embedding`, `Sequential`, `Dropout`. Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `SiLU`. Norms: `LayerNorm`, `RMSNorm`, `BatchNorm`.

**Attention.** Scaled dot-product attention, Multi-Head Attention, and Grouped Query Attention — each backed by 5 CUDA kernel variants (naive, tiled, flash, optimized flash, MMA Tensor Core). Causal and padding mask support.

**Training.** SGD with momentum, Adam with bias correction. MSE, cross-entropy, and cross-entropy-from-logits losses. 5 LR schedulers: StepLR, CosineAnnealingLR, ExponentialLR, LinearLR, MultiStepLR.

**CUDA kernels.** 7 matmul implementations (naive through 2D block tiling, plus an MMA TF32 Tensor Core kernel), 5 attention kernels, 14 element-wise ops. Shared memory tiling, coalesced access patterns, and `mma.sync` Tensor Core instructions where it matters.

## Benchmarks

Matmul — fp32, 3x1024x1024, 100 iterations:

```
PyTorch CUDA (cuBLAS)      0.09s  22.6x  (7.4 TFLOPS)
Tensorax MMA TF32          0.52s   3.9x  <- best (1.24 TFLOPS, Tensor Cores)
Tensorax 2D Block Tiling   0.63s   3.2x  (1.02 TFLOPS)
Tensorax 1D Block Tiling   0.76s   2.7x
Tensorax Tiled             0.93s   2.2x
Tensorax Cache Blocking    1.06s   1.9x
Tensorax SM Coalesced      1.25s   1.6x
Tensorax Default           1.25s   1.6x
NumPy CPU (baseline)       2.03s   1.0x
```

Attention — apples-to-apples vs PyTorch fp16 SDPA (cuDNN's fused path),
B=4 H=8, fp16 inputs, RTX 3070 Ti Laptop. Time per run:

```
                  S=256     S=512    S=1024   S=2048   S=4096   S=8192
d=64    tensorax  0.16 ms   0.28 ms  0.76 ms   2.80 ms   8.91 ms  36.59 ms
        PyTorch   0.03 ms   0.08 ms  0.26 ms   1.10 ms   4.29 ms  17.21 ms
        ratio     0.19x     0.30x    0.34x     0.39x    0.48x     0.47x
d=128   tensorax  0.24 ms   0.62 ms  1.55 ms   5.81 ms  21.49 ms  69.29 ms
        PyTorch   0.05 ms   0.14 ms  0.54 ms   2.03 ms   8.49 ms  34.76 ms
        ratio     0.23x     0.23x    0.35x     0.35x    0.40x     0.50x
d=256   tensorax  0.59 ms   1.63 ms  4.82 ms  16.04 ms  51.62 ms 203.3 ms
        PyTorch   0.09 ms   0.32 ms  1.33 ms   4.39 ms  18.36 ms  73.1 ms
        ratio     0.14x     0.20x    0.27x     0.27x    0.36x     0.36x
d=512   tensorax  1.55 ms   4.84 ms 14.08 ms  46.09 ms 178.2 ms  721.6 ms
        PyTorch   0.30 ms   1.11 ms  4.71 ms  19.86 ms  78.0 ms  334.2 ms
        ratio     0.19x     0.23x    0.33x     0.43x    0.44x     0.46x
```

(`d=1024` unsupported on this kernel — `s_q` smem at `d_k=1024` is
128 KB vs sm_86's 99 KB CTA cap.)

cuDNN's fused-attention path is **2.0×-7× faster** than tensorax across
all supported configs. The gap is widest at small problems
(per-CTA setup cost dominates) and tightest at long seq + large batch
(both kernels saturate compute). Tensorax peaks at ~16 TFLOPS, PyTorch
at ~34 TFLOPS — both well below the GPU's 84 TFLOPS fp16 Tensor Core
peak, but cuDNN gets ~2× closer. Full sweep across (B, S, d) is in
`benchmarks/attn_sweep.csv` (run `python benchmarks/attn_sweep.py`).

Other tensorax variants at Dk=Dv=512, S=256 (30 iterations, baselined
to NumPy fp32):

```
Tensorax MMA fp32          0.30s   18x  (0.64 TFLOPS, fp32 inputs)
Tensorax Optim. Flash      0.45s   12x  (0.43 TFLOPS)
Tensorax Flash SDPA        2.93s    2x
NumPy CPU (baseline)       5.47s    1x
Tensorax Tiled SDPA       32.79s    -
Tensorax Naive SDPA       90.47s    -
```

The MMA fp16 kernel uses inline PTX `mma.sync.aligned.m16n8k16` Tensor Core
instructions with online softmax (FA-style), `cp.async` double-buffered K/V
streaming with overlap across kv-tile boundaries, FA-2 split-Q across 4 warps
(each warp owns 16 query rows × full d_v with all warps running QKT in
parallel against shared K), per-warp register-resident softmax via
`__shfl_xor_sync` reductions, lazy output correction (skip the per-row rescale
when the running max barely shifts), Q pre-scaled by `scale·log2(e)` at load
so the softmax uses `exp2.approx` directly (one fmul per exp call dropped),
direct `ldmatrix.x2.trans` of V from the row-major staging buffer (no
explicit transpose pass), and a templated `DV_CHUNKS` parameter so the PV
loop's compile-time bound lets ptxas pin the output accumulator into
registers (verified with `-Xptxas=-v`: zero local-memory stack frame for
`d_v ≤ 256`).

The fp16 path takes pre-cast fp16 Q/K/V (matching how a real KV cache feeds an
inference workload) and skips the per-tile fp32→fp16 cast pass. Apples-to-apples
against PyTorch fp16 SDPA (cuDNN's fused-attention path), tensorax is ~2.5×–7×
behind across the (B, S, d) sweep — the gap is closing as we work through the
items in `docs/profiling/RESULTS.md` (smaller `s_q` smem to fit 2 CTAs/SM,
persistent CTA scheduling, cooperative K loading) but cuDNN remains the
target.

## Project layout

```
csrc/
  cuda/kernels/          elementwise, matmul (x6), reduction, attention (x5)
  cpu/                   CPU fallback for all ops
  tensor_ops.cpp/.h      pybind11 bindings

tensorax/
  tensor.py              Tensor class + autograd engine
  functional.py          F.relu, F.gelu, F.softmax, F.sdpa, losses, ...
  nn/                    Linear, Embedding, norms, dropout, attention (MHA, GQA)
  optim.py               SGD, Adam
  lr_scheduler.py        StepLR, CosineAnnealingLR, ExponentialLR, LinearLR, MultiStepLR
```

## Roadmap

What's here now: core tensor ops, autograd, all the layers/norms/activations listed above, two optimizers, five LR schedulers, three loss functions, five attention kernels, six matmul variants, MHA, GQA, embeddings.

What's next: Conv2D, MaxPool2D, AdamW, tensor indexing/slicing, model serialization, DataLoader, multi-GPU, mixed precision, DDP, ONNX export.

## Profiling

Tensorax includes fine-grained kernel profiling capabilities to measure performance at the section level. This is useful for identifying bottlenecks and understanding kernel behavior.

### Building with profiling support

```bash
TENSORAX_PROFILE=1 pip install -e .
```

This enables device-side clock64 ticks in CUDA kernels, allowing per-section timing measurements.

### Profile section APIs

For **matmul kernels**:
```python
from tensorax import functional as F

a = F.randn((1024, 1024), device='cuda')
b = F.randn((1024, 1024), device='cuda')

# Profile naive matmul
sections = F.profile_sections_matmul_naive(a, b)
# sections is a vector<long long> with clock64 ticks for each kernel section

# Other variants: tiled, shared_memory_coalesced, shared_memory_cache_blocking, 
# 1d_blocktiling, 2d_blocktiling
```

For **attention (SDPA) kernels**:
```python
# Query, Key, Value tensors
q = F.randn((4, 8, 256, 64), device='cuda')   # (B, H, S, Dk)
k = F.randn((4, 8, 256, 64), device='cuda')
v = F.randn((4, 8, 256, 64), device='cuda')

# Profile variants: naive, tiled, flash, mma, flash_optimized
sections = F.profile_sections_sdpa_naive(q, k, v, mask=None)
sections = F.profile_sections_sdpa_mma(q, k, v, mask=None)
sections = F.profile_sections_sdpa_flash_optimized(q, k, v, mask=None)
```

Each function returns a vector of `long long` values representing device clock64 ticks for sequential sections of the kernel. This enables precise measurement of specific computation phases without host-device synchronization overhead per-section.

See [profiling results](docs/profiling/RESULTS.md) for benchmark data and section-by-section breakdowns.

## Docs

- [Usage Guide](docs/USAGE.md) — full API reference with code examples
- [Architecture](docs/ARCHITECTURE.md) — system design, kernel strategy, autograd internals
- [Development](docs/DEVELOPMENT.md) — building from source, testing, contributing
- [Profiling](docs/profiling/RESULTS.md) — kernel profiling results and section analysis
- [Examples](examples/) — runnable scripts

## Citation

```bibtex
@software{tensorax2025,
  title  = {Tensorax: Pure C++/CUDA Tensor Library},
  author = {Shrirang Mahajan},
  year   = {2025},
  url    = {https://github.com/NotShrirang/tensorax}
}
```

## License

[MIT](LICENSE)
