"""
Simple example demonstrating basic tensor operations.
"""

import numpy as np
from tensorax import Tensor

def main():
    print("=== Basic Tensor Operations ===\n")
    
    # Create tensors
    a = Tensor([[1, 2], [3, 4]], dtype='float32')
    b = Tensor([[5, 6], [7, 8]], dtype='float32')
    
    print("Tensor a:")
    print(a)
    print(f"Shape: {a.shape}\n")
    
    print("Tensor b:")
    print(b)
    print(f"Shape: {b.shape}\n")
    
    # Addition
    c = a + b
    print("a + b:")
    print(c)
    print()
    
    # Element-wise multiplication
    d = a * b
    print("a * b (element-wise):")
    print(d)
    print()
    
    # Matrix multiplication
    e = a @ b
    print("a @ b (matrix multiplication):")
    print(e)
    print()
    
    # Transpose
    f = a.T
    print("a.T (transpose):")
    print(f)
    print()

if __name__ == "__main__":
    main()
