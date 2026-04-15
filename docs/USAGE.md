# Usage Guide

Complete reference for using Tensorax — from basic tensor operations to training neural networks on GPU.

## Table of Contents

- [Installation](#installation)
- [Tensor Basics](#tensor-basics)
- [Operations](#operations)
- [Automatic Differentiation](#automatic-differentiation)
- [Neural Network Layers](#neural-network-layers)
- [Normalization Layers](#normalization-layers)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Training a Model](#training-a-model)
- [Embedding](#embedding)
- [Attention](#attention)
- [Learning Rate Schedulers](#learning-rate-schedulers)
- [CUDA / GPU Acceleration](#cuda--gpu-acceleration)
- [Matmul Kernel Selection](#matmul-kernel-selection)
- [Functional API Reference](#functional-api-reference)
- [Data Types & Constants](#data-types--constants)

---

## Installation

**From PyPI:**

```bash
pip install tensorax
```

**From source:**

```bash
git clone https://github.com/NotShrirang/tensorax.git
cd tensorax
bash build.sh       # auto-detects CUDA
pip install -e .
```

**Manual build:**

```bash
# CPU only
python setup.py build_ext --inplace

# With CUDA
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

**Platform support:** Linux ✅ · macOS (Intel & Apple Silicon) ✅ · Windows ❌ *(coming soon)*

---

## Tensor Basics

### Creating Tensors

```python
from tensorax import Tensor

# From nested lists
a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

# With explicit shape
c = Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 2))

# Factory methods
zeros = Tensor.zeros((3, 3))
ones = Tensor.ones((2, 4))
rand = Tensor.randn((5, 5))
filled = Tensor.full((3, 3), 7.0)
```

### Tensor Properties

```python
a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

a.shape      # (2, 3)
a.ndim       # 2
a.size       # 6
a.dtype      # 'float32'
a.device     # 'cpu'
a.tolist()   # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

### Specifying Device and Dtype

```python
t = Tensor([[1.0, 2.0]], dtype='float64', device='cpu')
```

---

## Operations

### Element-wise Arithmetic

```python
c = a + b          # addition
d = a - b          # subtraction
e = a * b          # element-wise multiplication
f = a / b          # division
g = a ** 2         # power
```

All arithmetic ops support **broadcasting**:

```python
a = Tensor([[1.0, 2.0], [3.0, 4.0]])   # (2, 2)
b = Tensor([[10.0, 20.0]])              # (1, 2)
c = a + b   # broadcasts b → (2, 2)
```

Scalar operations:

```python
c = a + 5.0
d = a * 2.0
e = a - 1.0
f = a / 3.0
```

### Mathematical Functions

```python
x = Tensor([[1.0, 4.0, 9.0]])

x.sqrt()    # [1.0, 2.0, 3.0]
x.exp()     # element-wise exponential
x.log()     # element-wise natural log
x ** 0.5    # same as sqrt
```

### Matrix Operations

```python
a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

c = a @ b       # matrix multiply (uses block_tiling_2d on CUDA)
d = a.T         # transpose last 2 dims
e = a.transpose(0, 1)  # transpose specific dims
```

### Reductions

```python
x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

x.sum()              # scalar: 21.0
x.sum(dim=0)         # [5.0, 7.0, 9.0]
x.sum(dim=1)         # [6.0, 15.0]
x.mean()             # scalar: 3.5
x.mean(dim=0)        # [2.5, 3.5, 4.5]

# keepdim preserves the reduced dimension
x.sum(dim=1, keepdim=True)   # shape: (2, 1)
x.mean(dim=0, keepdim=True)  # shape: (1, 3)
```

### Shape Operations

```python
x = Tensor.randn((2, 3, 4))

y = x.reshape((6, 4))
z = x.repeat_interleave(3, dim=1)  # repeat along dim 1
```

---

## Automatic Differentiation

Tensorax supports reverse-mode autodiff with a computational graph. Gradients are tracked through 18+ operations.

```python
x = Tensor([[2.0]], requires_grad=True)
w = Tensor([[3.0]], requires_grad=True)
b = Tensor([[1.0]], requires_grad=True)

# Forward pass
y = w * x + b   # y = 3*2 + 1 = 7

# Backward pass
y.backward()

# Access gradients
x.grad   # → 3.0 (dy/dx = w)
w.grad   # → 2.0 (dy/dw = x)
b.grad   # → 1.0 (dy/db = 1)
```

### Supported Autograd Operations

Gradients flow through: `add`, `sub`, `mul`, `div`, `matmul`, `pow`, `sqrt`, `exp`, `sum`, `mean`, `relu`, `sigmoid`, `tanh`, `softmax`, `transpose`, `reshape`, `repeat_interleave`, `mse_loss`, `embedding`.

### Zeroing Gradients

```python
x.zero_grad()  # on a tensor
# or via optimizer
optimizer.zero_grad()  # zeros all parameter gradients
```

---

## Neural Network Layers

### Module Base Class

All layers inherit from `nn.Module` which provides:
- `parameters()` / `named_parameters()` — iterate over trainable params
- `train()` / `eval()` — toggle training mode
- `cuda()` / `cpu()` / `to(device)` — move to device
- `zero_grad()` — zero all parameter gradients

### Linear

```python
from tensorax import nn

linear = nn.Linear(in_features=4, out_features=8, bias=True)
output = linear(x)  # x shape: (batch, 4) → output shape: (batch, 8)
```

Uses Xavier/Glorot initialization.

### Activation Layers

```python
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=-1)
gelu = nn.GELU()
silu = nn.SiLU()

y = relu(x)
y = gelu(x)    # Gaussian Error Linear Unit (used in GPT, BERT)
y = silu(x)    # SiLU/Swish (used in LLaMA, EfficientNet)
```

### Dropout

```python
dropout = nn.Dropout(p=0.5)

# Only active during training
model.train()
y = dropout(x)    # randomly zeroes elements, scales by 1/(1-p)

model.eval()
y = dropout(x)    # identity (pass-through)
```

### Sequential

```python
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(8, 3),
    nn.Sigmoid()
)

output = model(x)

# Access parameters
for param in model.parameters():
    print(param.shape)

# Access layers by index
first_layer = model[0]
```

---

## Normalization Layers

### LayerNorm

```python
ln = nn.LayerNorm(normalized_shape=8, eps=1e-5)
output = ln(x)  # normalizes over the last dimension
```

Learnable parameters: `weight` (scale) and `bias` (shift).

### RMSNorm

```python
rms = nn.RMSNorm(normalized_shape=8, eps=1e-6)
output = rms(x)  # RMS normalization (no mean subtraction)
```

Learnable parameter: `weight` only.

### BatchNorm

```python
bn = nn.BatchNorm(num_features=8, eps=1e-5, momentum=0.1)

model.train()
output = bn(x)   # uses batch statistics, updates running stats

model.eval()
output = bn(x)   # uses running mean/variance
```

---

## Loss Functions

```python
from tensorax import functional as F

pred = Tensor([[0.8, 0.1, 0.1]], requires_grad=True)
target = Tensor([[1.0, 0.0, 0.0]])

# Mean Squared Error
loss = F.mse_loss(pred, target)

# Cross Entropy (from softmax probabilities + one-hot targets)
loss = F.cross_entropy_loss(pred, target)

# Cross Entropy from Logits (numerically stable)
logits = Tensor([[2.0, 1.0, 0.1]])
targets = Tensor([0])   # class indices
loss = F.cross_entropy_from_logits(logits, targets)
loss = F.cross_entropy_from_logits(logits, targets, reduce_mean=False)  # per-sample
```

All losses support `loss.backward()`.

---

## Optimizers

### SGD

```python
from tensorax import optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Adam

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

### Training Loop Pattern

```python
optimizer.zero_grad()   # clear gradients
loss.backward()         # compute gradients
optimizer.step()        # update parameters
```

---

## Training a Model

Complete end-to-end example:

```python
from tensorax import nn, Tensor, optim, functional as F

# Create model
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.LayerNorm(16),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(8, 3),
    nn.Sigmoid()
)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(100):
    output = model(x_train)
    loss = F.mse_loss(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.tolist()[0]:.4f}')

# Inference
model.eval()
predictions = model(x_test)
```

---

## Learning Rate Schedulers

```python
from tensorax import lr_scheduler
```

### StepLR

Decays the learning rate by `gamma` every `step_size` epochs.

```python
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# epoch 0-29: lr=0.001, epoch 30-59: lr=0.0001, ...
```

### ExponentialLR

Decays the learning rate by `gamma` every epoch.

```python
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# lr = base_lr * 0.95^epoch
```

### CosineAnnealingLR

Cosine annealing from base LR to `eta_min` over `T_max` epochs.

```python
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

### LinearLR

Linear warmup/decay between two factors.

```python
# Warmup: start at 1/3 of base LR, linearly reach full LR over 10 epochs
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=10)
```

### MultiStepLR

Decays at specific milestone epochs.

```python
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
# epoch 0-49: lr=0.001, epoch 50-79: lr=0.0001, epoch 80+: lr=0.00001
```

### Usage Pattern

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    output = model(x_train)
    loss = F.mse_loss(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()          # update learning rate

    print(scheduler.get_last_lr())  # current LR
```

---

## Embedding

```python
emb = nn.Embedding(num_embeddings=1000, embedding_dim=64)

# Lookup with 1D indices
indices = Tensor([4, 12, 7], shape=(3,))
vectors = emb(indices)  # shape: (3, 64)

# Lookup with 2D indices (batch of sequences)
indices = Tensor([0, 1, 2, 3, 4, 5], shape=(2, 3))
vectors = emb(indices)  # shape: (2, 3, 64)

# Supports backprop — gradients scatter back to the weight matrix
vectors.sum().backward()
emb.weight.grad  # shape: (1000, 64)
```

---

## Attention

### Scaled Dot-Product Attention

```python
from tensorax import Tensor, functional as F
from tensorax.nn.attention import create_causal_mask, create_padding_mask

batch, heads, seq_len, d_k = 2, 8, 64, 64

Q = Tensor.randn((batch, heads, seq_len, d_k))
K = Tensor.randn((batch, heads, seq_len, d_k))
V = Tensor.randn((batch, heads, seq_len, d_k))

# Basic attention
out = F.scaled_dot_product_attention(Q, K, V)

# High-performance specific kernels:
out = F.scaled_dot_product_attention_mma(Q, K, V)              # Ampere+ Tensor Cores utilizing mma.sync
out = F.scaled_dot_product_attention_flash(Q, K, V)            # Custom flash attention memory optimization
out = F.scaled_dot_product_attention_flash_optimized(Q, K, V)  # Advanced Flash Attention
out = F.scaled_dot_product_attention_tiled(Q, K, V)            # Shared memory block tiling

# Causal (autoregressive) attention
mask = create_causal_mask(seq_len, batch_size=batch, num_heads=heads)
out = F.scaled_dot_product_attention(Q, K, V, mask=mask)

# Padding mask
lengths = [50, 64]  # actual lengths per batch element
mask = create_padding_mask(lengths, max_len=seq_len, num_heads=heads)
```

### Attention Layer

```python
from tensorax.nn.attention import ScaledDotProductAttention

attn = ScaledDotProductAttention()
out = attn(Q, K, V, mask=mask)
```

### Grouped Query Attention (GQA)

```python
from tensorax.nn.attention import GroupedQueryAttention

gqa = GroupedQueryAttention(num_heads=8, num_kv_heads=2)
# num_heads must be divisible by num_kv_heads
# KV heads are automatically replicated

out = gqa(query, key, value, mask=mask)
```

### Multi-Head Attention

```python
from tensorax import nn

mha = nn.MultiHeadAttention(embed_dim=64, num_heads=8, bias=True)

# Input: (batch, seq_len, embed_dim)
q = Tensor.randn((2, 10, 64))
k = Tensor.randn((2, 10, 64))
v = Tensor.randn((2, 10, 64))

output = mha(q, k, v)          # shape: (2, 10, 64)
output = mha(q, k, v, mask=mask)  # with optional attention mask
```

Includes Q/K/V/output linear projections, head splitting/concatenation, and builds on `F.scaled_dot_product_attention`.

### CUDA Attention Kernels

Four hand-written CUDA attention kernels are available:

```python
# Auto-selected (naive on CPU, naive on CUDA)
out = F.scaled_dot_product_attention(Q, K, V)

# Specific kernels (CUDA only)
out = F.scaled_dot_product_attention_tiled(Q.cuda(), K.cuda(), V.cuda())
out = F.scaled_dot_product_attention_flash(Q.cuda(), K.cuda(), V.cuda())
out = F.scaled_dot_product_attention_flash_optimized(Q.cuda(), K.cuda(), V.cuda())
```

| Kernel | Technique | Best For |
|--------|-----------|----------|
| **Naive** | One thread/element, three-pass softmax | Small sequences, correctness baseline |
| **Tiled** | Shared memory K/V tiles, online softmax | Medium sequences |
| **Flash** | Block Q/K/V tiling, online softmax + rescaling | Long sequences, O(1) memory |
| **Flash Optimized** | Optimized flash with tuned parameters | Production workloads |

All kernels support: arbitrary batch/heads, asymmetric seq lengths (`seq_q ≠ seq_k`), separate `d_k`/`d_v`, optional additive masks.

---

## CUDA / GPU Acceleration

### Check Availability

```python
from tensorax import Tensor

Tensor.cuda_is_available()  # True / False
```

### Move Tensors

```python
a = Tensor([[1.0, 2.0], [3.0, 4.0]])

a_gpu = a.cuda()       # move to GPU
a_cpu = a_gpu.cpu()    # move back to CPU
a_dev = a.to('cuda')   # alternative syntax
```

### Move Models

```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
model.cuda()    # moves all parameters to GPU
model.cpu()     # moves all parameters to CPU
```

### Device Enforcement

Operations require tensors on the same device:

```python
a_cpu = Tensor([[1.0]])
b_gpu = Tensor([[2.0]]).cuda()

# a_cpu + b_gpu  # RuntimeError: Tensors on different devices
a.cuda() + b_gpu  # works
```

---

## Matmul Kernel Selection

When running on CUDA, you can choose from 6 matrix multiplication kernel implementations:

```python
a = Tensor.randn((1024, 1024)).cuda()
b = Tensor.randn((1024, 1024)).cuda()

c = a.matmul(b, method="default")                     # basic CUDA matmul
c = a.matmul(b, method="tiled")                        # tiled with shared memory
c = a.matmul(b, method="shared_memory_coalesced")      # coalesced memory access
c = a.matmul(b, method="shared_memory_cache_blocking")  # cache-optimized blocking
c = a.matmul(b, method="block_tiling_1d")              # 1D block tiling (fastest)
c = a.matmul(b, method="block_tiling_2d")              # 2D block tiling (used by @)
```

The `@` operator defaults to `block_tiling_2d`. The `matmul()` method defaults to `default`.

### Benchmark Results (fp32, 3×1024×1024, 100 runs)

| Implementation | Time (s) | vs NumPy |
|---|---|---|
| 1D Block Tiling *(best)* | 0.95 | **2.31× faster** |
| Tiled | 1.22 | **1.80× faster** |
| NumPy CPU *(baseline)* | 1.85 | 1.00× |
| Shared Mem Cache Blocking | 2.18 | 0.85× |
| Default CUDA | 3.37 | 0.55× |
| Shared Mem Coalescing | 3.44 | 0.54× |
| PyTorch CUDA *(reference)* | 0.41 | **4.51× faster** |

---

## Functional API Reference

```python
from tensorax import functional as F
```

### Activations

| Function | Description |
|---|---|
| `F.relu(x)` | max(0, x) |
| `F.sigmoid(x)` | 1 / (1 + exp(-x)) |
| `F.tanh(x)` | hyperbolic tangent |
| `F.softmax(x, dim=-1)` | normalized probabilities along `dim` |
| `F.gelu(x)` | Gaussian Error Linear Unit |
| `F.silu(x)` | SiLU/Swish: x * sigmoid(x) |

### Linear

| Function | Description |
|---|---|
| `F.linear(x, weight, bias=None)` | `x @ weight.T + bias` |

### Losses

| Function | Description |
|---|---|
| `F.mse_loss(pred, target)` | mean squared error |
| `F.cross_entropy_loss(pred, target)` | CE from softmax + one-hot |
| `F.cross_entropy_from_logits(logits, targets, reduce_mean=True)` | CE from raw logits (stable) |

### Attention

| Function | Description |
|---|---|
| `F.scaled_dot_product_attention(Q, K, V, mask=None)` | Auto kernel selection |
| `F.scaled_dot_product_attention_tiled(Q, K, V, mask=None)` | Tiled CUDA kernel |
| `F.scaled_dot_product_attention_flash(Q, K, V, mask=None)` | Flash attention |
| `F.scaled_dot_product_attention_flash_optimized(Q, K, V, mask=None)` | Optimized flash |

### Not Yet Implemented

| Function | Status |
|---|---|
| `F.conv2d(...)` | raises `NotImplementedError` |
| `F.max_pool2d(...)` | raises `NotImplementedError` |

---

## Data Types & Constants

```python
import tensorax as ts

# Integer types
ts.int8, ts.int16, ts.int32, ts.int64

# Unsigned integer types
ts.uint8, ts.uint16, ts.uint32, ts.uint64

# Floating-point types
ts.float16, ts.float32, ts.float64

# Complex types
ts.complex64, ts.complex128

# Device constants
ts.cpu, ts.cuda

# Use in tensor creation
t = Tensor([[1.0]], dtype=ts.float64, device=ts.cpu)
```
