<div align="center">

<br>

# ⚡ Tensorax

### A from-scratch tensor library with hand-written CUDA kernels.

No PyTorch. No NumPy. Pure C++/CUDA + Python.

<br>

[![PyPI](https://img.shields.io/pypi/v/tensorax.svg?style=flat-square&color=blueviolet)](https://pypi.org/project/tensorax/)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/personalized-badge/tensorax?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=orange&left_text=downloads&style=flat-square)](https://pepy.tech/projects/tensorax)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/github/actions/workflow/status/NotShrirang/tensorax/tests.yml?label=tests&style=flat-square)](https://github.com/NotShrirang/tensorax/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen?style=flat-square)](https://github.com/NotShrirang/tensorax/actions/workflows/coverage.yml)

<br>

[Usage Guide](docs/USAGE.md) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](docs/DEVELOPMENT.md) · [Examples](examples/)

<br>

</div>

---

<br>

<table>
<tr>
<td width="50%">

### 🔩 &nbsp; Zero heavy dependencies
Only `pybind11` — no PyTorch, NumPy, or cuBLAS at runtime.

### ⚡ &nbsp; Hand-written CUDA kernels
6 matmul variants, 4 attention kernels, 14 element-wise ops — all from scratch.

### 🧠 &nbsp; Full autograd engine
Reverse-mode autodiff with gradient tracking through 18+ operations.

</td>
<td width="50%">

### 🎯 &nbsp; PyTorch-like API
Familiar `Tensor`, `nn.Module`, `optim.Adam` interface — minimal learning curve.

### 🧱 &nbsp; Batteries included
Linear, ReLU, LayerNorm, BatchNorm, Dropout, GQA, Flash Attention — ready to train.

### 📚 &nbsp; Built to learn from
Clean, readable implementation of a DL framework from first principles.

</td>
</tr>
</table>

<br>

---

<br>

## Get Started

```bash
pip install tensorax
```

```python
from tensorax import Tensor, nn, optim, functional as F

# Build
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.LayerNorm(8), nn.Linear(8, 3))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(100):
    loss = F.mse_loss(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

> **→** Full usage guide with all APIs, code examples, and details: **[docs/USAGE.md](docs/USAGE.md)**

<br>

---

<br>

## What's Inside

<br>

<table>
<tr>
<td width="33%" valign="top">

**Core**
- Tensor with CPU ↔ CUDA
- Broadcasting arithmetic
- `sum`, `mean` with keepdim
- `reshape`, `transpose`
- `exp`, `log`, `sqrt`, `pow`
- 13 dtype constants

</td>
<td width="33%" valign="top">

**Neural Networks**
- `Linear`, `Sequential`
- `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `LayerNorm`, `RMSNorm`, `BatchNorm`
- `Dropout`
- `Module` base class

</td>
<td width="33%" valign="top">

**Training**
- `SGD` with momentum
- `Adam` with bias correction
- `MSE`, `CrossEntropy`, `CE from logits`
- Autograd through 18+ ops

</td>
</tr>
<tr>
<td width="33%" valign="top">

**Attention**
- Scaled dot-product attention
- 4 CUDA kernels (naive → flash)
- Grouped Query Attention
- Causal & padding masks

</td>
<td width="33%" valign="top">

**CUDA Kernels**
- 6 matmul implementations
- 14 element-wise ops
- Parallel reductions
- Tiled + coalesced access

</td>
<td width="33%" valign="top">

**Infra**
- 400 tests, 98% coverage
- CI/CD with GitHub Actions
- `pybind11` bindings
- Automatic CUDA fallback

</td>
</tr>
</table>

<br>

---

<br>

## Performance

**Matrix Multiplication** — fp32, 3×1024×1024, 100 runs:

```
PyTorch CUDA (ref)         ████████████████████████████████████████████  0.41s  (4.51×)
Tensorax 1D Block Tiling   ██████████████████████████████████████████    0.95s  (2.31×)  ← best
Tensorax Tiled             ████████████████████████████████              1.22s  (1.80×)
NumPy CPU (baseline)       █████████████████████████                    1.85s  (1.00×)
```

> **2.31× faster** than NumPy · **43%** of PyTorch's cuBLAS kernels · all hand-written, zero library calls

**Attention Kernels** — 4 implementations from naive to flash, supporting arbitrary batch/heads, asymmetric sequence lengths, and optional masks.

<br>

---

<br>

## Project Structure

```
csrc/                           C++ / CUDA backend
  cuda/kernels/                   elementwise · matmul (×6) · reduction · attention (×4)
  cpu/                            CPU fallback for all ops
  tensor_ops.{cpp,h}             pybind11 bindings

tensorax/                       Python package
  tensor.py                       Tensor class + autograd (1100 lines)
  functional.py                   F.relu, F.softmax, F.sdpa, ...
  nn/                             Linear, norms, dropout, attention, GQA
  optim.py                        SGD, Adam
```

<br>

---

<br>

## Roadmap

| Status | |
|:---:|---|
| ✅ | Core ops · autograd · NN layers · norms · optimizers · losses · attention (4 CUDA kernels) · GQA · matmul (6 variants) |
| 🚧 | Multi-head attention with projections · expanded benchmarking |
| 🔮 | Conv2D · MaxPool2D · GELU/Swish · AdamW · LR schedulers · indexing/slicing · serialization · multi-GPU · mixed precision · DDP |

<br>

---

<br>

## Documentation

| | |
|---|---|
| **[Usage Guide](docs/USAGE.md)** | API reference, code examples, training patterns |
| **[Architecture](docs/ARCHITECTURE.md)** | System design, kernel strategy, autograd internals |
| **[Development](docs/DEVELOPMENT.md)** | Build, test, contribute |
| **[Examples](examples/)** | Runnable scripts for common tasks |

<br>

---

<br>

## Contributing

```
Fork → Branch → Commit → PR
```

See **[DEVELOPMENT.md](docs/DEVELOPMENT.md)** for build instructions and guidelines.

<br>

---

<br>

<div align="center">

## Citation

</div>

```bibtex
@software{tensorax2025,
  title  = {Tensorax: Pure C++/CUDA Tensor Library},
  author = {Shrirang Mahajan},
  year   = {2025},
  url    = {https://github.com/NotShrirang/tensorax}
}
```

<br>

---

<br>

<div align="center">

**[GitHub](https://github.com/NotShrirang)** &nbsp;·&nbsp; **[Issues](https://github.com/NotShrirang/tensorax/issues)** &nbsp;·&nbsp; **[Discussions](https://github.com/NotShrirang/tensorax/discussions)**

Built with ❤️ by [@NotShrirang](https://github.com/NotShrirang)

⭐ Star if you find this useful

</div>
