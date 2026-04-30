# Tensorax

Tensorax is a deep learning framework written from scratch in C++/CUDA with a Python frontend. Every kernel — matmul, attention, elementwise ops, reductions — is hand-written. No PyTorch, no NumPy, no cuBLAS at runtime. The only dependency is `pybind11` for the C++/Python bridge.

The goal is a clean, readable implementation of a DL framework from first principles that also runs fast on real hardware. The MMA attention kernel uses inline PTX assembly to hit Ampere Tensor Cores, and the best matmul variant runs at ~3x NumPy speed — all without calling into any external math library.

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

**CUDA kernels.** 6 matmul implementations (naive through 2D block tiling), 5 attention kernels, 14 element-wise ops. Shared memory tiling, coalesced access patterns, and `mma.sync` Tensor Core instructions where it matters.

## Benchmarks

Matmul — fp32, 3x1024x1024, 100 iterations:

```
PyTorch CUDA (cuBLAS)      0.08s  22.24x
Tensorax 2D Block Tiling   0.58s   2.97x  <- best
Tensorax 1D Block Tiling   0.64s   2.68x
Tensorax Tiled             0.83s   2.05x
Tensorax Cache Blocking    0.98s   1.75x
Tensorax SM Coalesced      1.14s   1.50x
Tensorax Default           1.18s   1.45x
NumPy CPU (baseline)       1.71s   1.00x
```

Attention — B=4 H=8 S=256 Dk=512 Dv=512, 30 iterations:

```
Tensorax MMA fp16          0.14s    644x  <- best (1.37 TFLOPS)
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
overlap across kv-tile boundaries, 4-warp-tiled PV split along d_v, and
register-resident output accumulators (no smem traffic for O between tiles).

The fp16 path takes pre-cast fp16 Q/K/V (matching how a real KV cache feeds an
inference workload) and skips the per-tile fp32→fp16 cast pass, giving a clean
~2.1× speedup over the fp32-input variant. Still ~3.5× behind PyTorch's fp32
SDPA (which dispatches to cuDNN's fused-attention path with multi-warp 64-row
tiles and a tuned schedule we haven't implemented yet) — closing that gap is
ongoing work; tracked in `ROADMAP.md`.

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
