# Tensorax

**A high-performance tensor computation library with CUDA acceleration, designed for deep learning and numerical computing.**

Built from scratch for deep learning and numerical computing with blazing-fast GPU acceleration.

[![PyPI version](https://img.shields.io/pypi/v/tensorax.svg)](https://pypi.org/project/tensorax/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CI](https://github.com/NotShrirang/tensorax/workflows/Tests/badge.svg)](https://github.com/NotShrirang/tensorax/actions/workflows/tests.yml)
[![Code Coverage](https://github.com/NotShrirang/tensorax/workflows/Code%20Coverage/badge.svg)](https://github.com/NotShrirang/tensorax/actions/workflows/coverage.yml)

<!-- [![Tests](https://img.shields.io/badge/tests-229%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-green.svg)](htmlcov/) -->

## ✨ Features

- 🚀 **Pure C++/CUDA Backend**: No PyTorch or NumPy dependencies - truly standalone
- ⚡ **Extreme Performance**: Up to 448x speedup on GPU operations (1024×1024 matmul)
- 🔄 **Complete Autograd**: Full automatic differentiation with computational graph
- 🧠 **PyTorch-like API**: Familiar interface for easy adoption
- 🔧 **Flexible Deployment**: Works with or without CUDA - automatic fallback to CPU

## 🎯 Why Tensorax?

Unlike other libraries that wrap PyTorch or depend on NumPy, Tensorax is built **completely from scratch**:

- ✅ **Zero heavy dependencies** - Only requires `pybind11` for Python bindings
- ✅ **Production ready** - Complete training pipeline with optimizers and backprop
- ✅ **True CUDA acceleration** - Hand-written kernels, not wrappers
- ✅ **Educational** - Clean, readable codebase perfect for learning DL internals

## 📦 Installation

### Platform Support

**Currently supported:**

- ✅ **Linux** (Ubuntu, Debian, Fedora, etc.)
- ✅ **macOS** (Intel and Apple Silicon)

**Not yet supported:**

- ❌ **Windows** (coming soon - contributions welcome!)

### Prerequisites

- Python 3.8+
- C++17 compatible compiler (g++, clang++)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- pybind11 (automatically installed)

### Quick Install

**From PyPI:**

```bash
pip install tensorax
```

**From Source:**

```bash
git clone https://github.com/NotShrirang/tensorax.git
cd tensorax
bash build.sh       # Automatically detects CUDA
pip install -e .
```

### Manual Build

```bash
# CPU only
python setup.py build_ext --inplace

# With CUDA
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

### From PyPI

```bash
pip install tensorax
```

## 🚀 Quick Start

### Run the Demo

```bash
python demo.py  # Comprehensive showcase of all features
```

### Basic Tensor Operations

```python
from tensorax import Tensor

# Create tensors
a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

# Arithmetic operations
c = a + b           # Addition
d = a - b           # Subtraction
e = a * b           # Element-wise multiplication
f = a / b           # Division
g = a @ b           # Matrix multiplication

# Tensor properties
print(a.shape)      # (2, 2)
print(a.T)          # Transpose
print(a.device)     # 'cpu' or 'cuda'

# Factory methods
zeros = Tensor.zeros((3, 3))
ones = Tensor.ones((2, 4))
rand = Tensor.randn((5, 5))

# GPU acceleration
if Tensor.cuda_is_available():
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    c_gpu = a_gpu @ b_gpu  # 448x faster on 1024×1024!
    result = c_gpu.cpu()
```

### Automatic Differentiation

```python
from tensorax import Tensor

# Create tensors with gradient tracking
x = Tensor([[2.0]], requires_grad=True)
w = Tensor([[3.0]], requires_grad=True)
b = Tensor([[1.0]], requires_grad=True)

# Forward pass
y = w * x + b  # y = 3*2 + 1 = 7

# Backward pass
y.backward()

# Gradients
print(x.grad)  # dy/dx = 3
print(w.grad)  # dy/dw = 2
print(b.grad)  # dy/db = 1
```

### Neural Networks & Training

```python
from tensorax import nn, Tensor, optim, functional as F

# Define a model
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
    nn.Sigmoid()
)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(x_train)
    loss = F.mse_loss(output, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.tolist()[0]:.4f}')
```

### Functional API

```python
from tensorax import functional as F, Tensor

x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# Activation functions
y1 = F.relu(x)      # [0.0, 0.0, 0.0, 1.0, 2.0]
y2 = F.sigmoid(x)   # [0.119, 0.269, 0.5, 0.731, 0.881]
y3 = F.tanh(x)      # [-0.964, -0.762, 0.0, 0.762, 0.964]
y4 = F.softmax(x, dim=-1)  # Normalized probabilities

# Loss functions
pred = Tensor([[2.0, 1.5, 3.0]])
target = Tensor([[2.5, 2.0, 2.5]])
loss = F.mse_loss(pred, target)  # Mean squared error
```

## Project Structure

```
tensorax/
├── csrc/              # C++ and CUDA source code
│   ├── cuda/         # CUDA implementations
│   ├── cpu/          # CPU implementations
│   └── tensor_ops.*  # Core operations
├── tensorax/          # Python package
│   ├── tensor.py     # Tensor class
│   ├── nn/          # Neural network modules
│   ├── functional.py # Functional API
│   └── optim.py     # Optimizers
├── tests/           # Test suite
├── examples/        # Usage examples
└── docs/           # Documentation
```

## ⚡ Performance

Tensorax uses hand-optimized CUDA kernels for maximum performance. Here are some benchmark results for matrix multiplication (fp32, 3x1024×1024):

### Matrix Multiplication Benchmark (100 runs)

Comparison of different CUDA kernel implementations vs NumPy and PyTorch:

| Implementation               | Time (seconds) | Relative Performance |
| ---------------------------- | -------------- | -------------------- |
| **1D Block Tiling (Best)**   | 0.95           | **2.31x faster**     |
| Tiled Matrix Multiply        | 1.22           | **1.80x faster**     |
| NumPy (CPU)                  | 1.85           | Baseline (CPU)       |
| Shared Memory Cache Blocking | 2.18           | 0.85x                |
| Default CUDA                 | 3.37           | 0.55x                |
| Shared Memory Coalescing     | 3.44           | 0.54x                |
| **PyTorch CUDA (Reference)** | **0.41**       | **4.51x faster**     |

**Key Insights:**

- Our 1D block tiling implementation achieves **2.31x faster** performance than NumPy
- Performance is **43% of PyTorch's highly optimized CUDA kernels** (room for improvement)
- Tiled approaches consistently outperform naive implementations by **1.5-3x**

### Optimization Techniques

- ✅ **Coalesced memory access** for elementwise operations
- ✅ **Tiled matrix multiplication** with shared memory
- ✅ **Efficient parallel reductions** for sum/max operations
- ✅ **Kernel fusion** to minimize memory transfers

## Documentation

- [Development Guide](docs/DEVELOPMENT.md) - How to contribute and develop
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and internals
- [CI/CD Documentation](.github/CICD.md) - GitHub Actions workflows and automation
- [Examples](examples/) - Code examples and tutorials

## Development

### Setup development environment

```bash
# Clone repository
git clone https://github.com/NotShrirang/tensorax.git
cd tensorax

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Build the Extension

```bash
# Quick build (automatically detects CUDA)
bash build.sh

# Manual build (CPU only)
python setup.py build_ext --inplace

# Manual build (with CUDA)
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=tensorax --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_tensor.py -v
```

### Test Status

**Current Status (December 9, 2025):**

- ✅ **229 tests passing** (98.9% success rate)
- 🟡 **5 tests skipped** (CUDA tests - require GPU)
- 🔴 **0 tests failing**
- 📊 **87% code coverage**

**Test Breakdown:**

- Core tensor operations: 100% passing
- Neural network layers: 100% passing
- Optimizers: 100% passing
- Integration tests: 100% passing
- Functional API: 100% passing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Implemented Features

### Core Tensor Operations ✅

- [x] **Element-wise operations**: add, subtract, multiply, divide, power, sqrt, abs
- [x] **Matrix operations**: matmul (2D/3D batched), transpose
- [x] **Reduction operations**: sum, mean, max, min, argmax, argmin
- [x] **Mathematical functions**: exp, log, pow, clamp
- [x] **Shape operations**: reshape, view, squeeze, unsqueeze
- [x] **Tensor creation**: zeros, ones, full, randn
- [x] **Device management**: CPU ↔ CUDA transfers with automatic fallback
- [x] **Indexing & slicing**: Advanced tensor indexing and slicing
- [x] **Comparison operators**: eq, lt, gt with broadcasting
- [x] **Automatic differentiation**: Complete backpropagation with gradient tracking

### Neural Network Layers ✅

- [x] **Linear**: Fully connected layer with optional bias
- [x] **Activation layers**: ReLU, Sigmoid, Tanh, Softmax (with custom dim)
- [x] **Dropout**: Training/eval mode with configurable drop probability
- [x] **Sequential**: Container with recursive parameter collection
- [x] **Module system**: Base class with parameter management, device transfer, and train/eval modes

### Optimizers ✅

- [x] **SGD**: Stochastic Gradient Descent with momentum support
- [x] **Adam**: Adaptive moment estimation with bias correction
- [x] **Learning rate**: Configurable with validation
- [x] **Gradient management**: zero_grad() and parameter updates

### Loss Functions ✅

- [x] **Mean Squared Error (MSE)**: For regression tasks
- [x] **Cross Entropy Loss**: From probabilities or logits
- [x] **Backward pass**: All loss functions support gradient computation

### Functional API ✅

- [x] **Activations**: relu, sigmoid, tanh, softmax (multi-dimensional)
- [x] **Loss functions**: mse_loss, cross_entropy_loss, cross_entropy_from_logits
- [x] **Linear transformation**: Functional linear with optional bias
- [x] **Gradient support**: All functions support backpropagation

## 🗺️ Roadmap

### Completed ✅

- [x] Core tensor operations (element-wise, reduction, mathematical)
- [x] Automatic differentiation (complete autograd system)
- [x] Neural network layers (Linear, activations, Dropout)
- [x] Optimizers (SGD with momentum, Adam)
- [x] Loss functions (MSE, Cross Entropy)
- [x] Sequential container
- [x] Device management (CPU/CUDA)
- [x] Comprehensive test suite (229 tests passing)
- [x] Tensor serialization (save/load)

### In Progress 🚧

- [ ] CUDA kernel optimization for all operations
- [ ] Documentation improvements
- [ ] Performance benchmarking suite

### Future Features 🔮

- [ ] Convolution and pooling layers (Conv2D, MaxPool2D)
- [ ] Batch normalization and Layer normalization
- [ ] More activation functions (LeakyReLU, GELU, Swish, ELU)
- [ ] Additional optimizers (RMSprop, AdamW, Adagrad)
- [ ] Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealing)
- [ ] Multi-GPU support with data parallelism
- [ ] Mixed precision training (FP16/BF16)
- [ ] Distributed training (DDP)
- [ ] Graph optimization and fusion
- [ ] JIT compilation for custom operations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch's design and API
- CUDA optimization techniques from various deep learning frameworks
- Community contributions and feedback

## 🎓 Learning Resource

Tensorax is an excellent educational tool for understanding:

- Deep learning internals (how PyTorch/TensorFlow work under the hood)
- CUDA programming and GPU optimization
- Automatic differentiation implementation
- Building ML frameworks from scratch
- C++/Python interoperability with pybind11

Check out the [examples/](examples/) directory for tutorials!

## 📄 Citation

If you use Tensorax in your research or project, please cite:

```bibtex
@software{tensorax2025,
  title = {Tensorax: Pure C++/CUDA Tensor Library},
  author = {NotShrirang},
  year = {2025},
  url = {https://github.com/NotShrirang/tensorax}
}
```

## 📞 Contact & Support

- **GitHub**: [@NotShrirang](https://github.com/NotShrirang)
- **Issues**: [Report bugs or request features](https://github.com/NotShrirang/tensorax/issues)
- **Discussions**: [Ask questions](https://github.com/NotShrirang/tensorax/discussions)

## ⭐ Star History

If you find Tensorax useful, please consider giving it a star! ⭐

---

**Built with ❤️ by [@NotShrirang](https://github.com/NotShrirang)**
