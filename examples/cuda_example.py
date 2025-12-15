"""
Example demonstrating CUDA acceleration (requires GPU).
"""

import numpy as np
from tensorax import Tensor, cuda_is_available

def main():
    print("=== CUDA Operations Example ===\n")
    
    if not cuda_is_available():
        print("CUDA is not available. This example requires a GPU.")
        return
    
    print("CUDA is available!\n")
    
    # Create tensors on CPU
    a_cpu = Tensor(np.random.randn(1000, 1000), dtype='float32')
    b_cpu = Tensor(np.random.randn(1000, 1000), dtype='float32')
    
    print("Tensors created on CPU")
    print(f"a shape: {a_cpu.shape}, device: {a_cpu.device}")
    print(f"b shape: {b_cpu.shape}, device: {b_cpu.device}\n")
    
    # Move to CUDA
    a_cuda = a_cpu.cuda()
    b_cuda = b_cpu.cuda()
    
    print("Tensors moved to CUDA")
    print(f"a shape: {a_cuda.shape}, device: {a_cuda.device}")
    print(f"b shape: {b_cuda.shape}, device: {b_cuda.device}\n")
    
    # Perform operations on GPU
    c_cuda = a_cuda + b_cuda
    d_cuda = a_cuda * b_cuda
    e_cuda = a_cuda @ b_cuda
    
    print("Operations completed on CUDA")
    print(f"Addition result device: {c_cuda.device}")
    print(f"Multiplication result device: {d_cuda.device}")
    print(f"Matrix multiply result device: {e_cuda.device}\n")
    
    # Move result back to CPU
    e_cpu = e_cuda.cpu()
    print(f"Result moved back to CPU: {e_cpu.device}")
    print(f"Result shape: {e_cpu.shape}")

if __name__ == "__main__":
    main()
