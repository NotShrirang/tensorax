import timeit

import numpy as np
import torch
import tensorax as ts

times = 100
batch, M, K, N = 3, 1024, 1024, 1024

a_torch = torch.randn((batch, M, K), device='cuda', dtype=torch.float32)
b_torch = torch.randn((batch, K, N), device='cuda', dtype=torch.float32)

a_t = ts.Tensor(a_torch.cpu().numpy(), dtype='float32', device='cuda')
b_t = ts.Tensor(b_torch.cpu().numpy(), dtype='float32', device='cuda')

a_np = a_torch.cpu().numpy()
b_np = b_torch.cpu().numpy()

# Benchmarking matmul with shared memory coalescing
def matmul_shared_memory_coalesced():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="shared_memory_coalesced")
    torch.cuda.synchronize()
    return c

# Benchmarking default matmul
def matmul_default():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="default")
    torch.cuda.synchronize()
    return c

# Benchmarking tiled matmul
def matmul_tiled():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="tiled")
    torch.cuda.synchronize()
    return c

# Benchmarking matmul with shared memory cache blocking
def matmul_cache_blocking():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="shared_memory_cache_blocking")
    torch.cuda.synchronize()
    return c

# Benchmarking matmul with 1D block tiling
def matmul_1d_block_tiling():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="block_tiling_1d")
    torch.cuda.synchronize()
    return c

# Benchmarking matmul with 2D block tiling
def matmul_2d_block_tiling():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="block_tiling_2d")
    torch.cuda.synchronize()
    return c

# Benchmarking matmul with MMA / Tensor Cores
def matmul_mma_tiling():
    torch.cuda.synchronize()
    c = a_t.matmul(b_t, method="mma")
    torch.cuda.synchronize()
    return c

def matmul_numpy():
    c_np = np.matmul(a_np, b_np)
    return c_np

# Benchmarking PyTorch matmul
def matmul_pytorch():
    torch.cuda.synchronize()
    c = torch.matmul(a_torch, b_torch)
    torch.cuda.synchronize()
    return c

def compute_tflops(time_sec, batch, M, K, N, times):
    # Total FLOPs for one matmul run: 2 * B * M * N * K
    total_flops = 2 * batch * M * N * K
    tflops = (total_flops * times) / (time_sec * 1e12)
    return tflops

# Warm-up run
print("Warming up...")
matmul_default()
matmul_shared_memory_coalesced()
matmul_tiled()
matmul_cache_blocking()
matmul_1d_block_tiling()
matmul_2d_block_tiling()
matmul_mma_tiling()
matmul_numpy()
matmul_pytorch()
print("Warm-up done.")

print(f"Starting benchmarks... (B={batch}, M={M}, K={K}, N={N})")

time_default = timeit.timeit(matmul_default, number=times)
print(f"Default matmul time over {times} runs: {time_default} seconds | Time per run: {time_default / times:.4f} seconds | TFLOPS: {compute_tflops(time_default, batch, M, K, N, times):.2f}")

time_shared_memory_coalesced = timeit.timeit(matmul_shared_memory_coalesced, number=times)
print(f"Matmul with shared memory coalescing time over {times} runs: {time_shared_memory_coalesced} seconds | Time per run: {time_shared_memory_coalesced / times:.4f} seconds | TFLOPS: {compute_tflops(time_shared_memory_coalesced, batch, M, K, N, times):.2f}")

time_cache_blocking = timeit.timeit(matmul_cache_blocking, number=times)
print(f"Matmul with shared memory cache blocking time over {times} runs: {time_cache_blocking} seconds | Time per run: {time_cache_blocking / times:.4f} seconds | TFLOPS: {compute_tflops(time_cache_blocking, batch, M, K, N, times):.2f}")

time_tiled = timeit.timeit(matmul_tiled, number=times)
print(f"Tiled matmul time over {times} runs: {time_tiled} seconds | Time per run: {time_tiled / times:.4f} seconds | TFLOPS: {compute_tflops(time_tiled, batch, M, K, N, times):.2f}")

time_1d_block_tiling = timeit.timeit(matmul_1d_block_tiling, number=times)
print(f"Matmul with 1D block tiling time over {times} runs: {time_1d_block_tiling} seconds | Time per run: {time_1d_block_tiling / times:.4f} seconds | TFLOPS: {compute_tflops(time_1d_block_tiling, batch, M, K, N, times):.2f}")

time_2d_block_tiling = timeit.timeit(matmul_2d_block_tiling, number=times)
print(f"Matmul with 2D block tiling time over {times} runs: {time_2d_block_tiling} seconds | Time per run: {time_2d_block_tiling / times:.4f} seconds | TFLOPS: {compute_tflops(time_2d_block_tiling, batch, M, K, N, times):.2f}")

time_mma_tiling = timeit.timeit(matmul_mma_tiling, number=times)
print(f"Matmul with MMA (Tensor Cores) time over {times} runs: {time_mma_tiling} seconds | Time per run: {time_mma_tiling / times:.4f} seconds | TFLOPS: {compute_tflops(time_mma_tiling, batch, M, K, N, times):.2f}")

time_numpy = timeit.timeit(matmul_numpy, number=times)
print(f"Numpy matmul time over {times} runs: {time_numpy} seconds | Time per run: {time_numpy / times:.4f} seconds | TFLOPS: {compute_tflops(time_numpy, batch, M, K, N, times):.2f}")

time_pytorch = timeit.timeit(matmul_pytorch, number=times)
print(f"PyTorch matmul time over {times} runs: {time_pytorch} seconds | Time per run: {time_pytorch / times:.4f} seconds | TFLOPS: {compute_tflops(time_pytorch, batch, M, K, N, times):.2f}")