#!/usr/bin/env python3
"""
Tensora Library Demo
====================
Demonstrates all core functionalities of the Tensora tensor library.
"""

import time
import timeit
from tensora import Tensor
import tensora.functional as F
from tensora.nn import Linear, ReLU, Sequential, Sigmoid, Tanh
from tensora.optim import SGD, Adam


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def demo_tensor_creation():
    """Demonstrate tensor creation methods."""
    print_section("1. Tensor Creation")
    
    # From nested lists
    print("\n→ From nested lists:")
    t1 = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"  t1 = {t1}")
    print(f"  shape: {t1.shape}, size: {t1.size}, device: {t1.device}")
    
    # Zeros
    print("\n→ Zeros:")
    t2 = Tensor.zeros((2, 3))
    print(f"  {t2}")
    
    # Ones
    print("\n→ Ones:")
    t3 = Tensor.ones((2, 3))
    print(f"  {t3}")
    
    # Full
    print("\n→ Full (filled with 7):")
    t4 = Tensor.full((2, 3), 7.0)
    print(f"  {t4}")
    
    # Random normal
    print("\n→ Random normal distribution:")
    t5 = Tensor.randn((2, 3))
    print(f"  {t5}")


def demo_basic_operations():
    """Demonstrate basic tensor operations."""
    print_section("2. Basic Tensor Operations")
    
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    
    print(f"\nt1 = {t1.tolist()}")
    print(f"t2 = {t2.tolist()}")
    
    # Addition
    print("\n→ Addition (t1 + t2):")
    result = t1 + t2
    print(f"  {result.tolist()}")
    
    # Scalar addition
    print("\n→ Scalar addition (t1 + 10):")
    result = t1 + 10
    print(f"  {result.tolist()}")
    
    # Multiplication
    print("\n→ Element-wise multiplication (t1 * t2):")
    result = t1 * t2
    print(f"  {result.tolist()}")
    
    # Matrix multiplication
    print("\n→ Matrix multiplication (t1 @ t2):")
    result = t1 @ t2
    print(f"  {result.tolist()}")
    
    # Transpose
    print("\n→ Transpose (t1.T):")
    result = t1.T
    print(f"  {result.tolist()}")


def demo_activation_functions():
    """Demonstrate activation functions."""
    print_section("3. Activation Functions")
    
    t = Tensor([[-2, -1, 0, 1, 2]])
    print(f"\nInput: {t.tolist()}")
    
    # ReLU
    print("\n→ ReLU:")
    result = F.relu(t)
    print(f"  {result.tolist()}")
    
    # Sigmoid
    print("\n→ Sigmoid:")
    result = F.sigmoid(t)
    print(f"  {result.tolist()}")
    
    # Tanh
    print("\n→ Tanh:")
    result = F.tanh(t)
    print(f"  {result.tolist()}")
    
    # Softmax
    print("\n→ Softmax:")
    t2 = Tensor([[1, 2, 3, 4, 5]])
    result = F.softmax(t2)
    print(f"  Input: {t2.tolist()}")
    print(f"  Output: {result.tolist()}")
    print(f"  Sum: {sum(result.tolist()[0]):.6f} (should be ~1.0)")


def demo_loss_functions():
    """Demonstrate loss functions."""
    print_section("4. Loss Functions")
    
    # MSE Loss
    print("\n→ Mean Squared Error (MSE):")
    predictions = Tensor([[1, 2, 3], [4, 5, 6]])
    targets = Tensor([[1.5, 2.5, 2.5], [4.5, 5.5, 5.5]])
    loss = F.mse_loss(predictions, targets)
    print(f"  Predictions: {predictions.tolist()}")
    print(f"  Targets: {targets.tolist()}")
    print(f"  MSE Loss: {loss.tolist()}")
    
    # Cross Entropy Loss
    print("\n→ Cross Entropy Loss:")
    logits = Tensor([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]])
    targets = Tensor([[1, 0, 0], [0, 0, 1]])
    loss = F.cross_entropy_loss(logits, targets)
    print(f"  Logits: {logits.tolist()}")
    print(f"  Targets (one-hot): {targets.tolist()}")
    print(f"  Cross Entropy Loss: {loss.tolist()}")


def demo_neural_network_layers():
    """Demonstrate neural network layers."""
    print_section("5. Neural Network Layers")
    
    # Linear layer
    print("\n→ Linear Layer (3 -> 2):")
    linear = Linear(3, 2)
    x = Tensor([[1, 2, 3]])
    output = linear(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output: {output.tolist()}")
    
    # ReLU layer
    print("\n→ ReLU Layer:")
    relu = ReLU()
    x = Tensor([[-1, 0, 1, 2]])
    output = relu(x)
    print(f"  Input: {x.tolist()}")
    print(f"  Output: {output.tolist()}")
    
    # Sigmoid layer
    print("\n→ Sigmoid Layer:")
    sigmoid = Sigmoid()
    x = Tensor([[-2, -1, 0, 1, 2]])
    output = sigmoid(x)
    print(f"  Input: {x.tolist()}")
    print(f"  Output: {output.tolist()}")
    
    # Tanh layer
    print("\n→ Tanh Layer:")
    tanh = Tanh()
    x = Tensor([[-2, -1, 0, 1, 2]])
    output = tanh(x)
    print(f"  Input: {x.tolist()}")
    print(f"  Output: {output.tolist()}")
    
    # Sequential model
    print("\n→ Sequential Model (MLP):")
    model = Sequential([
        Linear(4, 8),
        ReLU(),
        Linear(8, 3),
        Sigmoid()
    ])
    x = Tensor([[1, 2, 3, 4]])
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output: {output.tolist()}")


def demo_optimizers():
    """Demonstrate optimizers."""
    print_section("6. Optimizers")
    
    print("\n→ SGD Optimizer:")
    # Create a simple model
    layer = Linear(2, 1)
    optimizer = SGD(layer.parameters(), lr=0.01, momentum=0.9)
    
    print(f"  Initial weight: {layer.weight.tolist()}")
    
    # Simulate gradient descent step
    x = Tensor([[1, 2]])
    output = layer(x)
    # Manually set gradient (in real training, this comes from backprop)
    layer.weight.grad = Tensor.ones(layer.weight.shape)
    
    optimizer.step()
    print(f"  After 1 step: {layer.weight.tolist()}")
    
    optimizer.step()
    print(f"  After 2 steps: {layer.weight.tolist()}")
    
    print("\n→ Adam Optimizer:")
    layer2 = Linear(2, 1)
    optimizer2 = Adam(layer2.parameters(), lr=0.001)
    
    print(f"  Initial weight: {layer2.weight.tolist()}")
    
    # Simulate steps
    layer2.weight.grad = Tensor.ones(layer2.weight.shape)
    optimizer2.step()
    print(f"  After 1 step: {layer2.weight.tolist()}")


def demo_device_transfer():
    """Demonstrate CPU/CUDA device transfer."""
    print_section("7. Device Transfer (CPU ↔ CUDA)")
    
    try:
        # Check if CUDA is available
        t_cpu = Tensor([[1, 2], [3, 4]], device='cpu')
        print(f"\n→ CPU Tensor: {t_cpu.tolist()}")
        print(f"  Device: {t_cpu.device}")
        
        # Try to move to CUDA
        try:
            t_cuda = t_cpu.cuda()
            print(f"\n→ Moved to CUDA: {t_cuda.tolist()}")
            print(f"  Device: {t_cuda.device}")
            
            # Move back to CPU
            t_cpu_again = t_cuda.cpu()
            print(f"\n→ Moved back to CPU: {t_cpu_again.tolist()}")
            print(f"  Device: {t_cpu_again.device}")
            
            print("\n✓ CUDA is available and working!")
        except RuntimeError as e:
            print(f"\n✗ CUDA not available: {e}")
    except Exception as e:
        print(f"\n✗ Device transfer demo failed: {e}")


def demo_performance_comparison():
    """Demonstrate performance with matrix multiplication."""
    print_section("8. Performance Demo")
    
    print("\n→ Matrix Multiplication Performance:")
    
    # Small matrices
    size = 64
    print(f"\n  Testing with {size}x{size} matrices:")
    t1 = Tensor.randn((size, size))
    t2 = Tensor.randn((size, size))
    
    # CPU timing
    start = time.time()
    result = t1 @ t2
    cpu_time = time.time() - start
    print(f"  CPU Time: {cpu_time*1000:.2f} ms")
    
    # Try CUDA if available
    try:
        t1_cuda = t1.cuda()
        t2_cuda = t2.cuda()
        
        # Warm-up
        _ = t1_cuda @ t2_cuda

        start = time.time()
        result_cuda = t1_cuda @ t2_cuda
        cuda_time = time.time() - start
        print(f"  CUDA Time: {cuda_time*1000:.2f} ms")
        print(f"  Speedup: {cpu_time/cuda_time:.2f}x")
    except RuntimeError:
        print("  CUDA not available for comparison")
    
    # Larger matrices
    size = 128
    print(f"\n  Testing with {size}x{size} matrices:")
    t1 = Tensor.randn((size, size))
    t2 = Tensor.randn((size, size))
    
    start = time.time()
    result = t1 @ t2
    cpu_time = time.time() - start
    print(f"  CPU Time: {cpu_time*1000:.2f} ms")
    
    try:
        t1_cuda = t1.cuda()
        t2_cuda = t2.cuda()
        _ = t1_cuda @ t2_cuda
        
        start = time.time()
        result_cuda = t1_cuda @ t2_cuda
        cuda_time = time.time() - start
        print(f"  CUDA Time: {cuda_time*1000:.2f} ms")
        print(f"  Speedup: {cpu_time/cuda_time:.2f}x")
    except RuntimeError:
        print("  CUDA not available for comparison")


    # Extra large matrices
    size = 1024
    print(f"\n  Testing with {size}x{size} matrices:")
    t1 = Tensor.randn((size, size))
    t2 = Tensor.randn((size, size))
    start = time.time()
    result = t1 @ t2
    cpu_time = time.time() - start
    print(f"  CPU Time: {cpu_time*1000:.2f} ms")

    try:
        t1_cuda = t1.cuda()
        t2_cuda = t2.cuda()
        _ = t1_cuda @ t2_cuda

        start = time.time()
        result_cuda = t1_cuda @ t2_cuda
        cuda_time = time.time() - start
        print(f"  CUDA Time: {cuda_time*1000:.2f} ms")
        print(f"  Speedup: {cpu_time/cuda_time:.2f}x")
    except RuntimeError:
        print("  CUDA not available for comparison")

    # Using Timit for benchmarking
    # Using different methods for matmul
    # 1. Default
    # 2. Shared Memory Coalesced
    # 3. Tiled

    print("\n→ Benchmarking different matmul methods on CUDA:")
    print("  Methods: default, shared_memory_coalesced, tiled")
    try:
        size = 1024
        batch = 3
        a = Tensor.randn((batch, size, size), dtype='float32', device='cuda')
        b = Tensor.randn((batch, size, size), dtype='float32', device='cuda')
        print(f"\n  Benchmarking different matmul methods with {size}x{size} matrices on CUDA:")

        def matmul_func(method: str):
            c = a.matmul(b, method=method)
            return c

        print("\n  Running benchmarks...")

        print("   Method: default")
        time_default = timeit.timeit(lambda: matmul_func("default"), number=1000)
        print(f"      Default matmul time over 1000 runs: {time_default:.4f} seconds")

        print("   Method: shared_memory_coalesced")
        time_shared_memory_coalesced = timeit.timeit(lambda: matmul_func("shared_memory_coalesced"), number=1000)
        print(f"      Matmul with shared memory coalescing time over 1000 runs: {time_shared_memory_coalesced:.4f} seconds")

        print("   Method: tiled")
        time_tiled = timeit.timeit(lambda: matmul_func("tiled"), number=1000)
        print(f"      Tiled matmul time over 1000 runs: {time_tiled:.4f} seconds")
        print("\n✓ Benchmarking complete.")

        print("\n→ Performance Comparison:")
        print(f"  SM Coalesced vs Default Speedup: {time_default/time_shared_memory_coalesced:.2f}x")
        print(f"  Tiled vs Default Speedup: {time_default/time_tiled:.2f}x")

    except RuntimeError:
        print("  CUDA not available for matmul benchmarking")


def demo_simple_training():
    """Demonstrate a simple training scenario."""
    print_section("9. Simple Training Example")
    
    print("\n→ Training a simple linear model:")
    print("  Task: Learn f(x) = 2x + 1")
    
    # Create model
    model = Linear(1, 1)
    optimizer = Adam(model.parameters(), lr=3e-4)  # Small learning rate to prevent explosion
    

    def func(x: Tensor):
        return [2 * xi + 1 for xi in x.tolist()]

    # Training data: y = 2x + 1
    X_train = Tensor.randn((500,)) * 10
    y_train = func(X_train)
    
    print(f"\n  Initial weight: {model.weight.tolist()}")
    print(f"  Initial bias: {model.bias.tolist()}")
    
    # Training loop with backpropagation
    print("\n  Training for 100 epochs...")
    for epoch in range(100):
        total_loss = 0
        for x_val, y_val in zip(X_train, y_train):
            # x_val and y_val are scalars from iteration
            # Reshape to (1, 1) for linear layer: batch_size=1, features=1
            X = Tensor([[x_val]])
            y_true = Tensor([[y_val]])
            
            # Forward pass
            y_pred = model(X)
            
            # Compute loss (MSE)
            loss = F.mse_loss(y_pred, y_true)
            total_loss += loss.tolist()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss:.4f}")
    
    print(f"\n  Final weight: {model.weight.tolist()}")
    print(f"  Final bias: {model.bias.tolist()}")
    print(f"  Target: weight ≈ 2.0, bias ≈ 1.0")


def main():
    """Run all demos."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "TENSORA LIBRARY DEMO" + " "*28 + "█")
    print("█" + " "*15 + "Pure C++/CUDA Tensor Library" + " "*27 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        demo_tensor_creation()
        demo_basic_operations()
        demo_activation_functions()
        demo_loss_functions()
        demo_neural_network_layers()
        demo_optimizers()
        demo_device_transfer()
        demo_performance_comparison()
        demo_simple_training()
        
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + " "*24 + "Demo Complete!" + " "*30 + "█")
        print("█" + " "*68 + "█")
        print("█"*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
