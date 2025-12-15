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
- Using built-in layers (Linear, ReLU)
- Training loop with optimizer
- Evaluation mode

Run: `python examples/simple_nn.py`

### 3. CUDA Operations (`cuda_example.py`)

Demonstrates GPU acceleration with CUDA:

- Moving tensors between CPU and GPU
- Performing operations on GPU
- Checking CUDA availability

Run: `python examples/cuda_example.py` (requires GPU)

## Requirements

Install Tensorax first:

```bash
pip install -e .
```

For CUDA support, ensure you have:

- CUDA Toolkit installed
- Compatible NVIDIA GPU
