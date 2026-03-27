# Tensorax User & Developer Guide

**Status:** Production Ready
**Version:** 0.2.0
**Test Coverage:** 433 tests passing (100%), 95% code coverage

This comprehensive guide covers everything from basic usage to advanced development.

## Project Status: Production Ready ✅

Tensorax has achieved production-ready status with:

- ✅ Complete tensor operations (element-wise, reduction, mathematical)
- ✅ Full autograd system with gradient tracking through 18+ ops
- ✅ Neural network layers (Linear, Embedding, activations, norms, Dropout, Sequential)
- ✅ Activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU, SiLU)
- ✅ Attention (Scaled Dot-Product, Flash, GQA, Multi-Head Attention)
- ✅ Working optimizers (SGD with momentum, Adam)
- ✅ Loss functions (MSE, Cross Entropy, Cross Entropy from logits)
- ✅ Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealingLR, LinearLR, MultiStepLR)
- ✅ Comprehensive test suite (433 passing tests)
- ✅ Device management (CPU/CUDA with automatic fallback)
- ✅ 95% code coverage

## Quick Start Guide

### Installation

**From PyPI (Recommended):**

```bash
pip install tensorax
```

**From Source:**

```bash
# Clone the repository
git clone https://github.com/NotShrirang/tensorax.git
cd tensorax

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Build the extension (auto-detects CUDA)
bash build.sh

# Install in development mode
pip install -e .

# Verify installation
python -c "import tensorax; print(f'Tensorax version: {tensorax.__version__}')"
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=tensorax --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_tensor.py -v          # Core tensor tests
pytest tests/test_nn.py -v              # Neural network tests
pytest tests/test_optim.py -v           # Optimizer tests
pytest tests/test_integration.py -v     # Integration tests
```

### Verify CUDA Setup (Optional)

If you have a GPU:

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Build with CUDA
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace

# Test CUDA functionality
python -c "from tensorax import Tensor; print(f'CUDA available: {Tensor.cuda_is_available()}')"
```

## What's Implemented

All core phases are complete. Here is a summary of what has been built:

### Core Operations (Complete ✅)
- Element-wise: add, sub, mul, div, pow
- Math: exp, log, sqrt
- Reductions: sum, mean, max (with dim, keepdim)
- Shape: reshape, transpose, repeat_interleave
- Broadcasting, scalar ops, operator overloading

### Autograd (Complete ✅)
- Reverse-mode autodiff with computational graph
- Gradients through 18+ ops including embedding
- `loss.backward()` → `param.grad` → `optimizer.step()`

### Neural Network Layers (Complete ✅)
- `Linear`, `Embedding`, `Sequential`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `SiLU`
- Normalization: `LayerNorm`, `RMSNorm`, `BatchNorm`
- `Dropout` with train/eval mode
- Attention: `ScaledDotProductAttention`, `GroupedQueryAttention`, `MultiHeadAttention`

### Optimizers & Schedulers (Complete ✅)
- `SGD` (with momentum), `Adam` (with bias correction)
- `StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LinearLR`, `MultiStepLR`

### CUDA Kernels (Complete ✅)
- 6 matmul variants (naive → 2D block tiling)
- 4 attention kernels (naive → flash optimized)
- 14+ element-wise kernels
- Reduction kernels (sum, mean, max)

---

## Next Phases

### Phase Next: Performance Optimization

### CUDA Optimizations

1. **Kernel Fusion**
   Combine multiple operations into single kernel:

   ```cuda
   // Instead of: y = relu(x + bias)
   // Do: y = max(0, x + bias) in one kernel
   __global__ void fused_add_relu_kernel(...) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < size) {
           output[idx] = fmaxf(0.0f, input[idx] + bias[idx]);
       }
   }
   ```

2. **Stream-based Parallelism**

   ```cpp
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);

   // Launch operations on different streams
   kernel1<<<grid, block, 0, stream1>>>(...);
   kernel2<<<grid, block, 0, stream2>>>(...);
   ```

3. **Memory Pooling**
   Implement custom allocator to reduce allocation overhead

4. **Profiling**

   ```bash
   # Profile with Nsight Compute
   ncu --set full python your_script.py

   # Profile with nvprof
   nvprof python your_script.py
   ```

### Python Optimizations

1. **Cython for Critical Paths**
2. **Numba JIT Compilation**
3. **Reduce Python/C++ boundary crossings**

## Future: Advanced Features

### 1. Mixed Precision Training

```python
# tensorax/amp.py
class GradScaler:
    def scale(self, loss):
        return loss * self.scale_factor

    def step(self, optimizer):
        # Unscale gradients and update
```

### 2. Multi-GPU Support

```python
# tensorax/nn/parallel.py
class DataParallel(Module):
    def __init__(self, module, device_ids):
        self.module = module
        self.device_ids = device_ids
```

### 3. Model Serialization

```python
def save(model, path):
    state_dict = {name: param.numpy()
                  for name, param in model.named_parameters()}
    np.savez(path, **state_dict)

def load(model, path):
    state_dict = np.load(path)
    for name, param in model.named_parameters():
        param._data = state_dict[name]
```

### 4. JIT Compilation

```python
# tensorax/jit.py
@jit
def custom_operation(x, y):
    return x * y + x
```

## Testing & Validation

### Comprehensive Test Suite

1. **Numerical Gradient Checking**

   ```python
   def numerical_gradient(f, x, eps=1e-5):
       grad = np.zeros_like(x)
       for i in range(x.size):
           x_flat = x.flatten()
           x_flat[i] += eps
           f_plus = f(x_flat.reshape(x.shape))
           x_flat[i] -= 2*eps
           f_minus = f(x_flat.reshape(x.shape))
           grad.flat[i] = (f_plus - f_minus) / (2*eps)
       return grad
   ```

2. **Performance Benchmarks**

   ```python
   import time

   def benchmark_op(op, *args, n_runs=100):
       times = []
       for _ in range(n_runs):
           start = time.perf_counter()
           result = op(*args)
           times.append(time.perf_counter() - start)
       return np.mean(times), np.std(times)
   ```

3. **Correctness Tests Against PyTorch**

   ```python
   def test_against_pytorch():
       import torch
       x_np = np.random.randn(100, 100)

       # Tensorax
       x_tr = Tensor(x_np)
       y_tr = F.relu(x_tr)

       # PyTorch
       x_pt = torch.tensor(x_np)
       y_pt = torch.relu(x_pt)

       np.testing.assert_allclose(y_tr.numpy(), y_pt.numpy())
   ```

## Documentation & Examples

### 1. API Documentation

- Use Sphinx for auto-generating docs
- Write docstrings for all public APIs
- Include type hints

### 2. Tutorials

- Getting started guide
- Building CNNs
- Building RNNs/Transformers
- Custom operations
- Performance tuning

### 3. Example Projects

- MNIST classifier
- ResNet on CIFAR-10
- GPT-style language model
- GAN implementation

## Release & Distribution

### 1. Packaging

```bash
# Build source distribution
python setup.py sdist

# Build wheels for different platforms
python setup.py bdist_wheel

# Upload to PyPI
pip install twine
twine upload dist/*
```

### 2. CI/CD Setup

Create `.github/workflows/tests.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/
```

### 3. Version Management

Follow semantic versioning:

- v0.1.x: Initial release — core ops, autograd, NN layers, attention, CUDA kernels
- v0.2.0: Embedding, GELU/SiLU, MultiHeadAttention, LR schedulers
- v1.0.0: Stable API (future)

## Tips for Success

### 1. Start Small

- Get basic operations working first
- Test each component thoroughly
- Don't try to implement everything at once

### 2. Profile Early

- Identify bottlenecks before optimizing
- Compare against PyTorch/TensorFlow
- Use proper profiling tools

### 3. Write Tests

- Test every operation
- Use gradient checking
- Add benchmarks

### 4. Document Everything

- Inline comments for complex code
- API documentation
- Architecture decisions

### 5. Community

- Share your progress
- Get feedback early
- Contribute to similar projects

## Common Pitfalls

1. **Memory Leaks in CUDA**: Always free allocated memory
2. **Gradient Errors**: Use numerical gradient checking
3. **Performance**: Profile before optimizing
4. **API Design**: Keep it simple and consistent
5. **Testing**: Don't skip edge cases

## Resources

### Learning CUDA

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Deep Learning

- [Deep Learning Book](http://www.deeplearningbook.org/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

### Similar Projects

- PyTorch source code
- TinyGrad
- JAX
- Theano (historical)

## Getting Help

- Open issues for bugs
- Discussions for questions
- Stack Overflow for general questions
- Discord/Slack communities

---

Good luck with your tensor library development! 🚀
