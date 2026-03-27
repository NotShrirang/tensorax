# Development Guide

**Status:** Production Ready
**Version:** 0.2.0
**Test Status:** 433 tests passing (100%)
**Coverage:** 95%

This guide covers the development workflow for Tensorax.

## Setting Up Development Environment

1. **Clone the repository:**

```bash
git clone https://github.com/NotShrirang/tensorax.git
cd tensorax
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
pip install -e ".[dev]"
```

## Project Structure

```
tensorax/
├── csrc/                  # C++ and CUDA source code
│   ├── cuda/             # CUDA-specific code
│   │   ├── kernels/     # CUDA kernel implementations
│   │   │   ├── elementwise.cu
│   │   │   ├── reduction.cu
│   │   │   └── matmul.cu
│   │   ├── cuda_utils.cuh
│   │   └── tensor_cuda.cu
│   ├── cpu/             # CPU-only implementations
│   │   └── tensor_cpu.cpp
│   ├── tensor_ops.h     # Header with declarations
│   └── tensor_ops.cpp   # Main C++ entry point with Python bindings
├── tensorax/             # Python package
│   ├── __init__.py
│   ├── tensor.py        # Core Tensor class + autograd
│   ├── functional.py    # Functional API (activations, losses, attention)
│   ├── optim.py         # SGD, Adam
│   ├── lr_scheduler.py  # StepLR, ExponentialLR, CosineAnnealingLR, LinearLR, MultiStepLR
│   └── nn/             # Neural network modules
│       ├── __init__.py
│       ├── module.py    # Base Module class
│       ├── layers.py    # Linear, Embedding, activations, norms, Dropout, Sequential
│       └── attention/   # ScaledDotProductAttention, GQA, MultiHeadAttention
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/              # Documentation
├── setup.py           # Build configuration
├── pyproject.toml     # Project metadata
└── MANIFEST.in        # Package manifest
```

## Building the Extension

### CPU-only build:

```bash
python setup.py build_ext --inplace
```

### With CUDA support:

```bash
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

## Running Tests

### Test Status (March 2026)

- ✅ **433 tests passing** (100% success rate)
- 🔴 **0 tests failing**
- 📊 **95% code coverage**

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=tensorax --cov-report=html --cov-report=term
```

### Run Specific Test Categories

```bash
# Core tensor operations
pytest tests/test_tensor.py -v

# Advanced tensor operations
pytest tests/test_advanced_tensor.py -v

# Neural network layers
pytest tests/test_nn.py -v

# Optimizers
pytest tests/test_optim.py -v

# Functional API
pytest tests/test_functional.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Coverage Analysis

```bash
# Generate HTML coverage report
pytest tests/ --cov=tensorax --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Module Coverage

| Module                    | Coverage | Status         |
| ------------------------- | -------- | -------------- |
| constants/\*.py           | 100%     | ✅ Perfect     |
| nn/\_\_init\_\_.py        | 100%     | ✅ Perfect     |
| nn/module.py              | 100%     | ✅ Perfect     |
| nn/attention/utils.py     | 100%     | ✅ Perfect     |
| utils/shape\_utils.py     | 100%     | ✅ Perfect     |
| utils/type\_checks.py     | 100%     | ✅ Perfect     |
| functional.py             | 98%      | ✅ Excellent   |
| lr\_scheduler.py          | 98%      | ✅ Excellent   |
| optim.py                  | 97%      | ✅ Excellent   |
| nn/layers.py              | 94%      | ✅ Excellent   |
| tensor.py                 | 92%      | ✅ Good        |
| nn/attention/layers.py    | 51%      | ⚠️ Needs tests |

## Code Style

Format code with Black:

```bash
black tensorax/ tests/
```

Check with flake8:

```bash
flake8 tensorax/ tests/
```

Type checking with mypy:

```bash
mypy tensorax/
```

## Adding New Features

### 1. Adding a new CUDA kernel

1. Create kernel in `csrc/cuda/kernels/your_kernel.cu`
2. Declare function in `csrc/tensor_ops.h`
3. Add Python binding in `csrc/tensor_ops.cpp`
4. Expose in Python API in `tensorax/tensor.py` or `tensorax/functional.py`
5. Add tests in `tests/`

### 2. Adding a new layer

1. Create layer class in `tensorax/nn/layers.py`
2. Export in `tensorax/nn/__init__.py`
3. Add tests in `tests/test_nn.py`
4. Add example in `examples/`

### 3. Adding a new optimizer

1. Create optimizer in `tensorax/optim.py`
2. Add tests in `tests/test_optim.py`

## Performance Optimization Tips

### CUDA Kernels:

- Use shared memory for frequently accessed data
- Coalesce global memory accesses
- Minimize divergent branches
- Use appropriate block and grid dimensions
- Profile with `nvprof` or Nsight Compute

### Python Code:

- Minimize Python loops over tensors
- Use vectorized operations when possible
- Cache computed values
- Profile with `cProfile`

## Debugging

### Python debugging:

```python
import pdb; pdb.set_trace()
```

### CUDA debugging:

Use `cuda-gdb` or add debug prints:

```cuda
printf("Debug: value = %f\n", value);
```

### Memory debugging:

Use `cuda-memcheck`:

```bash
cuda-memcheck python your_script.py
```

## Documentation

Build documentation:

```bash
cd docs
make html
```

View docs:

```bash
python -m http.server -d docs/_build/html
```

## Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped in `__init__.py`
- [ ] CHANGELOG updated
- [ ] Code formatted and linted
- [ ] Examples work correctly
- [ ] Build succeeds on all platforms
