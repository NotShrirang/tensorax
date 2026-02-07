import timeit

import numpy as np
import torch
import tensorax as ts
import tensorax.functional as F

times = 100

batch, heads, seq_len, d_k, d_v = 4, 8, 512, 512, 512

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

# Warm-up run
print("Warming up...")
sdpa_naive()
sdpa_tiled()
sdpa_flash()
sdpa_numpy()
sdpa_pytorch()
print("Warm-up done.")

print(f"Starting benchmarks... (B={batch}, H={heads}, S={seq_len}, Dk={d_k}, Dv={d_v})")

time_naive = timeit.timeit(sdpa_naive, number=times)
print(f"Naive SDPA time over {times} runs: {time_naive} seconds")

time_tiled = timeit.timeit(sdpa_tiled, number=times)
print(f"Tiled SDPA time over {times} runs: {time_tiled} seconds")

time_flash = timeit.timeit(sdpa_flash, number=times)
print(f"Flash SDPA time over {times} runs: {time_flash} seconds")

time_numpy = timeit.timeit(sdpa_numpy, number=times)
print(f"Numpy SDPA time over {times} runs: {time_numpy} seconds")

time_pytorch = timeit.timeit(sdpa_pytorch, number=times)
print(f"PyTorch SDPA time over {times} runs: {time_pytorch} seconds")
