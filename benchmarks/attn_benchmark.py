import timeit

import numpy as np
import torch
import tensorax as ts
import tensorax.functional as F


print("Tensorax version:", ts.__version__)
print("CUDA available:", ts.cuda_is_available())
print()
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA available:", torch.cuda.is_available())
print()

times = 5

batch, heads, seq_len, d_k, d_v = 4, 8, 256, 512, 512

q_torch = torch.randn((batch, heads, seq_len, d_k), device='cuda', dtype=torch.float32)
k_torch = torch.randn((batch, heads, seq_len, d_k), device='cuda', dtype=torch.float32)
v_torch = torch.randn((batch, heads, seq_len, d_v), device='cuda', dtype=torch.float32)

q_t = ts.Tensor(q_torch.cpu().numpy(), dtype='float32', device='cuda')
k_t = ts.Tensor(k_torch.cpu().numpy(), dtype='float32', device='cuda')
v_t = ts.Tensor(v_torch.cpu().numpy(), dtype='float32', device='cuda')

q_np = q_torch.cpu().numpy()
k_np = k_torch.cpu().numpy()
v_np = v_torch.cpu().numpy()

# Benchmarking naive SDPA
def sdpa_naive():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking tiled SDPA
def sdpa_tiled():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_tiled(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking flash SDPA
def sdpa_flash():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_flash(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking MMA SDPA
def sdpa_mma():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_mma(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking optimized flash SDPA
def sdpa_flash_optimized():
    torch.cuda.synchronize()
    c = F.scaled_dot_product_attention_flash_optimized(q_t, k_t, v_t)
    torch.cuda.synchronize()
    return c

# Benchmarking NumPy reference SDPA
def sdpa_numpy():
    scores = np.matmul(q_np, k_np.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    c = np.matmul(attn, v_np)
    return c

# Benchmarking PyTorch SDPA
def sdpa_pytorch():
    torch.cuda.synchronize()
    c = torch.nn.functional.scaled_dot_product_attention(q_torch, k_torch, v_torch)
    torch.cuda.synchronize()
    return c

def compute_tflops(time_sec, batch, heads, seq_len, d_k, d_v, times):
    # Total FLOPs for one SDPA run: 4 * B * H * S^2 * Dk + 2 * B * H * S^2 * Dv
    total_flops = (4 * batch * heads * seq_len**2 * d_k) + (2 * batch * heads * seq_len**2 * d_v)
    tflops = (total_flops * times) / (time_sec * 1e12)
    return tflops

# Warm-up run
print("Warming up...")
sdpa_naive()
sdpa_tiled()
sdpa_flash()
sdpa_mma()
sdpa_flash_optimized()
sdpa_numpy()
sdpa_pytorch()
print("Warm-up done.")

print(f"Starting benchmarks... (B={batch}, H={heads}, S={seq_len}, Dk={d_k}, Dv={d_v})")

time_naive = timeit.timeit(sdpa_naive, number=times)
print(f"Naive SDPA time over {times} runs: {time_naive} seconds | Time per run: {time_naive / times:.4f} seconds | TFLOPS: {compute_tflops(time_naive, batch, heads, seq_len, d_k, d_v, times):.2f}")

time_tiled = timeit.timeit(sdpa_tiled, number=times)
print(f"Tiled SDPA time over {times} runs: {time_tiled} seconds | Time per run: {time_tiled / times:.4f} seconds | TFLOPS: {compute_tflops(time_tiled, batch, heads, seq_len, d_k, d_v, times):.2f}")

time_flash = timeit.timeit(sdpa_flash, number=times)
print(f"Flash SDPA time over {times} runs: {time_flash} seconds | Time per run: {time_flash / times:.4f} seconds | TFLOPS: {compute_tflops(time_flash, batch, heads, seq_len, d_k, d_v, times):.2f}")

time_mma = timeit.timeit(sdpa_mma, number=times)
print(f"MMA SDPA time over {times} runs: {time_mma} seconds | Time per run: {time_mma / times:.4f} seconds | TFLOPS: {compute_tflops(time_mma, batch, heads, seq_len, d_k, d_v, times):.2f}")

time_flash_optimized = timeit.timeit(sdpa_flash_optimized, number=times)
print(f"Optimized Flash SDPA time over {times} runs: {time_flash_optimized} seconds | Time per run: {time_flash_optimized / times:.4f} seconds | TFLOPS: {compute_tflops(time_flash_optimized, batch, heads, seq_len, d_k, d_v, times):.2f}")

time_numpy = timeit.timeit(sdpa_numpy, number=times)
print(f"Numpy SDPA time over {times} runs: {time_numpy} seconds | Time per run: {time_numpy / times:.4f} seconds | TFLOPS: {compute_tflops(time_numpy, batch, heads, seq_len, d_k, d_v, times):.2f}")

time_pytorch = timeit.timeit(sdpa_pytorch, number=times)
print(f"PyTorch SDPA time over {times} runs: {time_pytorch} seconds | Time per run: {time_pytorch / times:.4f} seconds | TFLOPS: {compute_tflops(time_pytorch, batch, heads, seq_len, d_k, d_v, times):.2f}")
