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

Attention — B=4 H=8 S=256 Dk=512 Dv=512, 30 iterations:

```
Tensorax MMA fp16          0.12s    736x  <- best (1.59 TFLOPS)
PyTorch SDPA fp32          0.04s   2415x  (5.15 TFLOPS, internal TF32)
Tensorax MMA fp32          0.30s    301x  (0.64 TFLOPS)
Tensorax Optim. Flash      0.45s    201x  (0.43 TFLOPS)
Tensorax Flash SDPA        2.93s     31x
NumPy CPU (baseline)       5.47s     17x
Tensorax Tiled SDPA       32.79s      3x
Tensorax Naive SDPA       90.47s      1x
```

The MMA kernel uses inline PTX `mma.sync.aligned.m16n8k16` Tensor Core instructions
with online softmax (FA-style), `cp.async` double-buffered K/V streaming with
overlap across kv-tile boundaries, FA-2 split-Q across 4 warps (each warp owns
16 query rows × full d_v with all warps running QKT in parallel against shared
K), per-warp register-resident softmax via `__shfl_xor_sync` reductions, and
register-resident output accumulators (no smem traffic for P or O between tiles).

The fp16 path takes pre-cast fp16 Q/K/V (matching how a real KV cache feeds an
inference workload) and skips the per-tile fp32→fp16 cast pass, giving a clean
~2.5× speedup over the fp32-input variant. Still ~3.2× behind PyTorch's fp32
SDPA (which dispatches to cuDNN's fused-attention path with a tuned schedule we
haven't implemented yet) — closing that gap is ongoing work; tracked in
`ROADMAP.md`.

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
