# Tensorax Examples

This directory contains examples demonstrating various features of Tensorax.

## Examples

### 1. Basic Operations (`basic_operations.py`)

Demonstrates fundamental tensor operations:

- Creating tensors
- Addition, multiplication
- Matrix multiplication
- Transpose

Run: `python examples/basic_operations.py`

### 2. Simple Neural Network (`simple_nn.py`)

Shows how to build and train a simple feedforward neural network:

- Defining custom modules
- Using built-in layers (Linear, ReLU, GELU, SiLU)
- Training loop with optimizer and LR scheduler
- Evaluation mode

Run: `python examples/simple_nn.py`

### 3. CUDA Operations (`cuda_example.py`)

Demonstrates GPU acceleration with CUDA:

- Moving tensors between CPU and GPU
- Performing operations on GPU
- Checking CUDA availability

Run: `python examples/cuda_example.py` (requires GPU)

## Available Components

| Category | Components |
|---|---|
| **Layers** | `Linear`, `Embedding`, `Sequential` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `SiLU` |
| **Normalization** | `LayerNorm`, `RMSNorm`, `BatchNorm` |
| **Attention** | `ScaledDotProductAttention`, `GroupedQueryAttention`, `MultiHeadAttention` |
| **Optimizers** | `SGD`, `Adam` |
| **LR Schedulers** | `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LinearLR`, `MultiStepLR` |
| **Losses** | `mse_loss`, `cross_entropy_loss`, `cross_entropy_from_logits` |

## Requirements

Install Tensorax first:

**From PyPI:**

```bash
pip install tensorax
```

**From Source:**

```bash
pip install -e .
```

For CUDA support, ensure you have:

- CUDA Toolkit installed
- Compatible NVIDIA GPU
