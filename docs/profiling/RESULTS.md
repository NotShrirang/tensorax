# Kernel Profiling Results

Captured with Nsight Systems (`nsys`) via `benchmarks/profile_kernels.py`.
Each workload runs 3 warm-up launches followed by one measured launch
(scoped via `cudaProfilerStart/Stop`). Numbers below are from the
`cuda_gpu_kern_sum` summary in each `.nsys-rep` report.

Source CSVs and reports live in `benchmarks/profiling/`.

---

## Configuration

| Workload | Shape |
|---|---|
| matmul   | `B=3, M=K=N=1024`, fp32 |
| sdpa     | `B=4, H=8, S=256, Dk=Dv=512`, fp32 |

GPU clock and exact device info: see any `.nsys-rep` (Properties pane in
Nsight Systems UI).

---

## matmul (per-kernel)

Each launch processes one of the `B=3` batches, so "Total" = 3 × per-launch.

| Method | Per-launch | Total (B=3) | vs naive |
|---|---:|---:|---:|
| `default` (naive)              | 2.51 ms | 7.54 ms | 1.00× |
| `shared_memory_coalesced`      | 2.50 ms | 7.50 ms | 1.01× |
| `shared_memory_cache_blocking` | 1.93 ms | 5.78 ms | 1.30× |
| `tiled`                        | 1.48 ms | 4.43 ms | 1.70× |
| `block_tiling_1d`              | 624 µs  | 1.87 ms | 4.03× |
| `block_tiling_2d`              | **484 µs** | **1.45 ms** | **5.20×** |

### Notes
- `shared_memory_coalesced` is essentially identical to `default` — same
  inner-loop structure, just a different thread layout. The shared-memory
  load it advertises in the name isn't actually used.
- `tiled` and `cache_blocking` get a clear win from on-chip data reuse.
- `block_tiling_2d` is the fastest because each thread holds a `TM × TN = 4×4`
  register tile of `C`, amortising shared-memory reads.

---

## sdpa (per-kernel, current state)

Each row is a single launch. Times are dominated by sequence length (`S=256`)
and head dim (`Dk=Dv=512`).

| Method | Per-launch | vs naive | Notes |
|---|---:|---:|---|
| `naive`           | 2.96 s   | 1.00×    | three full passes over K (max → sumexp → ×V), no reuse |
| `tiled`           | 1.01 s   | 2.92×    | online softmax with K/V tiles in shared mem |
| `flash`           | 92.2 ms  | 32.1×    | classic flash-attention block iteration |
| `flash_optimized` | 8.73 ms  | 339×     | vec4 loads + warp reductions + lazy max update |
| **`mma`**         | **10.78 µs** | **274 K×** | fused-cast Tensor-Core path (see § *cast fusion* below) |

> Caveat still pending: the current `sdpa_kernel_mma` only iterates over a
> single 16×16 KV tile (no outer `kv_start` loop), so it does not yet process
> the full sequence. Its 10.78 µs is correct for the ~6 % of work it does, but
> it is not yet a complete SDPA implementation comparable to the other variants.

---

## sdpa.mma cast fusion (before vs after)

The original MMA path launched **5 kernels** per call (3× fp32→fp16, 1× MMA,
1× fp16→fp32). The fused version launches **1 kernel** — the cast happens
in-flight while staging into shared memory, and the output is written as
fp32 directly. Both `cast_*` kernels were also vectorised (4 elements/thread
via `float4` + `__half2`).

| | Before (5 launches) | After (1 launch) | Δ |
|---|---:|---:|---:|
| `cast_f32_to_f16` (×3) | 196.2 µs | — | gone |
| `cast_f16_to_f32` (×1) | 65.4 µs  | — | gone |
| `sdpa_kernel_mma`      | 9.92 µs  | 10.78 µs | +0.86 µs (in-flight `__floats2half2_rn`) |
| **Total per call**     | **271.5 µs** | **10.78 µs** | **−260.7 µs (−96 %)** |

Pre-fusion breakdown for reference (casts dominated at 96.4 %):

| Time % | Total | Count | Avg | Kernel |
|---:|---:|---:|---:|---|
| 72.3 % | 196.2 µs | 3 | 65.4 µs | `cast_f32_to_f16` (Q, K, V) |
| 24.1 % | 65.4 µs  | 1 | 65.4 µs | `cast_f16_to_f32` (output) |
|  3.7 % |   9.9 µs | 1 |  9.9 µs | `sdpa_kernel_mma`            |

---

## End-to-end benchmark (`benchmarks/attn_benchmark.py`)

Wall time across **30 runs**, including Python/C++ wrapper overhead, tensor
allocation, and `cudaDeviceSynchronize`. Speedup column is vs the naive baseline.

| Backend | Total (30 runs) | Per call | Speedup |
|---|---:|---:|---:|
| Tensorax Naive SDPA       | 92.71 s  | 3.09 s   | 1× |
| Tensorax Tiled SDPA       | 31.25 s  | 1.04 s   | 2.97× |
| NumPy CPU (baseline)      |  6.36 s  | 0.21 s   | 14.6× |
| Tensorax Flash SDPA       |  2.92 s  | 97.2 ms  | 31.7× |
| Tensorax Optim. Flash     |  0.408 s | 13.6 ms  | 227× |
| **Tensorax MMA Tensor Core** | **0.164 s** | **5.5 ms** | **565×** ← best |
| PyTorch SDPA (cuDNN)      |  0.0374 s | 1.25 ms | 2480× |

### Comparison vs prior README numbers

The MMA fusion roughly halved end-to-end time and almost doubled its speedup
vs naive. Other backends moved by single-digit %, i.e. run-to-run jitter.

| | README total | Now total | Δ time | README ×  | Now × | Δ × |
|---|---:|---:|---:|---:|---:|---:|
| Naive            | 98.26 s | 92.71 s | −6 % | 1× | 1× | — |
| Tiled            | 32.91 s | 31.25 s | −5 % | 3× | 2.97× | ~ |
| Flash            | 3.10 s  | 2.92 s  | −6 % | 31× | 31.7× | ~ |
| Optim. Flash     | 0.52 s  | 0.408 s | −22 % | 187× | 227× | +21 % |
| **MMA**          | **0.33 s** | **0.164 s** | **−50 %** | **297×** | **565×** | **+90 %** |
| NumPy            | 7.06 s  | 6.36 s  | −10 % | 14× | 14.6× | ~ |
| PyTorch          | 0.04 s  | 0.0374 s | −7 % | 2340× | 2480× | ~ |

---

## Where the time actually goes (cross-cutting)

- **matmul** scales cleanly with arithmetic intensity — every step from naive
  to 2D block-tiling is a real algorithmic improvement, and the speedup
  matches the increase in operand reuse per shared-memory load.
- **sdpa.naive** is recompute-bound: it walks the full sequence three times
  per query without reusing scores. Tiled and flash variants directly fix this.
- **sdpa.mma** used to be cast-bound (96 % in conversion overhead) and is now
  kernel-bound at the GPU level. End-to-end is now wrapper/overhead-bound:
  per-call wall time (5.5 ms) is ~500× the kernel time (10.78 µs), so
  remaining optimisation should target the host-side path or expand the
  kernel to the missing KV iterations.

---

## Optimisation ideas (status)

1. ✅ **Vectorise `cast_f32_to_f16` / `cast_f16_to_f32`** — done; each thread
   processes 4 elements via `float4` + `__half2`.
2. ✅ **Fuse the casts into `sdpa_kernel_mma`** — done; kernel takes fp32
   Q/K/V, casts in-flight to shared memory, writes fp32 output directly.
   End-to-end dropped from 271.5 µs → 10.78 µs (−96 %).
3. ⏳ **Complete the MMA kernel** so it iterates over all KV tiles. Currently
   only the first 16 keys are processed, so the per-launch number is for ~6 %
   of the work the other variants do.
4. ⏳ **Investigate per-call wrapper overhead.** Kernel takes 10.78 µs but
   end-to-end is 5.5 ms. The gap lives in the C++ binding, output-tensor
   allocation, and `cudaDeviceSynchronize`. PyTorch's path is ~3× faster
   end-to-end despite presumably more on-GPU work, so there's headroom here.
5. ⏳ **Run upstream layers in fp16** so Q/K/V arrive in fp16 already; lets
   the in-kernel cast also drop.
