# Tensorax - Project Structure

## 📦 Directory Tree

```
tensorax/
├── 📄 README.md                      # Main project documentation
├── 📄 LICENSE                        # MIT License
├── 📄 setup.py                       # Build configuration
├── 📄 pyproject.toml                 # Python project metadata
├── 📄 MANIFEST.in                    # Package manifest
├── 📄 requirements.txt               # Runtime dependencies (pybind11)
├── 📄 requirements-dev.txt           # Development dependencies
├── 📄 .gitignore                     # Git ignore rules
├── 🔧 build.sh                       # Quick build script
├── 📄 demo.py                        # Comprehensive demo script
├── 📄 PROJECT_STRUCTURE.md           # This file
├── 📄 REFACTORING_SUMMARY.md         # NumPy removal summary
│
├── 📁 csrc/                          # C++ and CUDA source code
│   ├── 📄 tensor_ops.h              # Operation declarations
│   ├── 📄 tensor_ops.cpp            # Python bindings (pybind11)
│   ├── 📁 cpu/                      # CPU implementations
│   │   └── 📄 tensor_cpu.cpp       # CPU operations (add, mul, matmul, etc.)
│   └── 📁 cuda/                     # CUDA implementations
│       ├── 📄 cuda_utils.cuh        # CUDA utilities and macros
│       ├── 📄 tensor_cuda.cu        # CUDA memory management
│       └── 📁 kernels/              # Optimized CUDA kernels
│           ├── 📄 elementwise.cu    # Element-wise ops (add, mul, sqrt, etc.)
│           ├── 📄 reduction.cu      # Reduction ops (sum, max, etc.)
│           └── 📄 matmul.cu         # Tiled matrix multiplication
│
├── 📁 tensorax/                       # Python package
│   ├── 📄 __init__.py               # Package initialization
│   ├── 📄 tensor.py                 # Core Tensor class with autograd
│   ├── 📄 functional.py             # Functional API (F.relu, losses, etc.)
│   ├── 📄 optim.py                  # Optimizers (SGD, Adam)
│   └── 📁 nn/                       # Neural network modules
│       ├── 📄 __init__.py
│       ├── 📄 module.py             # Base Module class
│       └── 📄 layers.py             # Layers (Linear, ReLU, Sequential, etc.)
│
├── 📁 tests/                         # Test suite
│   ├── 📄 __init__.py
│   ├── 📄 test_tensor.py            # Tensor operation tests
│   ├── 📄 test_nn.py                # Neural network tests
│   ├── 📄 test_optim.py             # Optimizer tests
│   └── 📄 test_functional.py        # Functional API tests
│
├── 📁 examples/                      # Usage examples
│   ├── 📄 README.md
│   ├── 📄 basic_operations.py       # Basic tensor ops demo
│   ├── 📄 simple_nn.py              # Neural network example
│   └── 📄 cuda_example.py           # GPU acceleration demo
│
└── 📁 docs/                          # Documentation
    ├── 📄 ARCHITECTURE.md           # System architecture details
    ├── 📄 DEVELOPMENT.md            # Development workflow
    └── 📄 GUIDE.md                  # Complete development roadmap
```

## 🎯 Key Components

### 1. C++/CUDA Backend (csrc/)

**Purpose**: High-performance tensor operations with zero PyTorch/NumPy dependency

**Key Files**:

- `tensor_ops.cpp`: Python bindings using pybind11, high-level operation wrappers
- `tensor_ops.h`: Operation declarations and TensorImpl class
- `cuda/kernels/*.cu`: Optimized CUDA kernels
- `cpu/tensor_cpu.cpp`: CPU fallback implementations

**Operations Fully Implemented**:

- **Element-wise**: add, subtract, multiply, divide, sqrt
- **Matrix**: matmul (tiled CUDA algorithm with 448x speedup), transpose
- **Activations**: ReLU, sigmoid, tanh, softmax
- **Losses**: MSE, cross-entropy
- **Utilities**: random normal distribution, device transfers (CPU↔CUDA)

### 2. Python API (tensorax/)

**Purpose**: User-friendly PyTorch-like interface with automatic differentiation

**Key Classes**:

- `Tensor`: Core multi-dimensional array with full autograd support
  - Operations: `+`, `-`, `*`, `/`, `@` (matmul)
  - Properties: `.T` (transpose), `.shape`, `.device`, `.requires_grad`
  - Methods: `.backward()`, `.zero_grad()`, `.sqrt()`, `.cuda()`, `.cpu()`
  - Factory methods: `.zeros()`, `.ones()`, `.full()`, `.randn()`
- `Module`: Base class for neural network layers
  - Parameter management
  - Device transfer support
- `Optimizer`: Base class for optimization algorithms
  - Parameter updates with gradient descent

**Neural Network Modules** (`nn/`):

- `Linear`: Fully connected layer with Xavier initialization
- `ReLU/Sigmoid/Tanh`: Activation layers
- `Sequential`: Layer container accepting list or varargs

**Optimizers** (`optim.py`):

- `SGD`: Stochastic Gradient Descent with momentum support
- `Adam`: Adaptive moment estimation with bias correction

### 3. Functional API (tensorax/functional.py)

**Purpose**: Stateless operations for functional programming style

**Functions**:

- **Activations**: `relu()`, `sigmoid()`, `tanh()`, `softmax()`
- **Losses**: `mse_loss()`, `cross_entropy_loss()`
- **Operations**: `linear()`

### 4. Automatic Differentiation

**Complete backpropagation system** supporting:

- All arithmetic operations (`+`, `-`, `*`, `/`)
- Matrix operations (matmul, transpose)
- Activation functions (ReLU, sigmoid, tanh)
- Loss functions (MSE)
- Gradient accumulation and parameter updates

**Gradient flow** tracked through computational graph with proper chain rule application.

### 5. Tests (tests/)

**Test Coverage**:

- Tensor operations and device transfers
- Gradient computation and backpropagation
- Layer functionality
- Optimizer behavior (SGD with momentum, Adam)
- Functional API

### 6. Documentation (docs/)

- **ARCHITECTURE.md**: System design, memory management, kernel design
- **DEVELOPMENT.md**: Build process, testing, debugging
- **GUIDE.md**: Complete development roadmap

## 🚀 Quick Start

### Requirements

- Python 3.8+
- C++17 compiler (g++ or clang++)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- pybind11 (automatically installed)

### Installation

```bash
# Clone repository
git clone https://github.com/NotShrirang/tensorax.git
cd tensorax

# Quick build (automatically detects CUDA)
bash build.sh

# Install in development mode
pip install -e .
```

### Run Demo

```bash
# Comprehensive demonstration of all features
python demo.py
```

### Test

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tensorax --cov-report=html
```

### Examples

```bash
# Basic operations
python examples/basic_operations.py

# Neural network
python examples/simple_nn.py

# CUDA demo (requires GPU)
python examples/cuda_example.py
```

## 📊 Implementation Status

### ✅ Completed

- [x] Project structure and build system
- [x] Basic tensor operations (CPU/CUDA)
- [x] Tensor class with device management
- [x] Module system for neural networks
- [x] Common layers (Linear, activations)
- [x] Optimizers (SGD, Adam)
- [x] Test infrastructure
- [x] Documentation and examples

### 🚧 To Implement (See GUIDE.md for details)

- [ ] Complete autograd system
- [ ] Convolution and pooling layers
- [ ] More reduction operations
- [ ] Batch normalization
- [ ] Learning rate schedulers
- [ ] Model serialization
- [ ] Multi-GPU support
- [ ] Mixed precision training

## 🔧 Build System

### Dependencies

**Runtime**:

- Python 3.8+
- NumPy
- PyTorch (for build utilities only)

**Build**:

- C++17 compiler
- pybind11
- CUDA Toolkit 11.0+ (optional)

### Configuration

- `setup.py`: Main build script
- `pyproject.toml`: Modern Python packaging
- `MANIFEST.in`: Files to include in distribution

### CUDA Support

Automatically detected via `CUDA_HOME` environment variable.
Falls back to CPU-only if CUDA not available.

## 📚 Learning Resources

### For CUDA Development

1. CUDA C Programming Guide
2. CUDA Best Practices Guide
3. Nsight Compute/Systems profiling tools

### For Deep Learning

1. PyTorch source code (reference implementation)
2. CS231n course (Stanford)
3. Deep Learning book (Goodfellow et al.)

### Similar Projects to Study

- PyTorch
- TinyGrad
- JAX
- Tinygrad

## 🤝 Contributing

See `docs/DEVELOPMENT.md` for:

- Development environment setup
- Code style guidelines
- Testing procedures
- Pull request process

## 📈 Performance Tips

### CUDA Optimization

1. Use tiled algorithms for matrix ops
2. Maximize shared memory usage
3. Ensure coalesced memory access
4. Profile with Nsight Compute
5. Consider kernel fusion

### Python Optimization

1. Minimize Python/C++ boundary crossings
2. Batch operations when possible
3. Use NumPy vectorization
4. Consider Cython for critical paths

## 🐛 Common Issues

### Build Errors

- Check CUDA_HOME is set correctly
- Verify C++ compiler supports C++17
- Ensure pybind11 is installed

### Runtime Errors

- Check CUDA availability with `cuda_is_available()`
- Verify tensor devices match for operations
- Check array shapes for broadcasting

### Memory Issues

- CUDA memory leaks: Check cudaFree calls
- CPU memory: Let Python GC handle it
- Use smaller batch sizes if OOM

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: docs/ directory
- **Examples**: examples/ directory

---

**Next Steps**: Follow the development roadmap in `docs/GUIDE.md` to implement remaining features!
