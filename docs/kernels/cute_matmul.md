# cute_matmul — Hopper fp16 GEMM via CUTLASS CollectiveBuilder

## Workload

| | |
|---|---|
| Op | `C = A @ B`, `A: (M, K)`, `B: (K, N)`, `C: (M, N)` |
| Shape | `B=1, M=N=K=4096`, fp16 in / fp16 out (fp32 accumulate) |
| Hardware | NVIDIA H100 (sm_90a) via Modal |
| Bench | `make bench-mm-cute` (`benchmarks/matmul_benchmark.py`, 30 iters, allclose vs torch matmul fp16 with rtol=atol=5e-3) |

**FLOPs / call:** `2·M·N·K = 2·4096³ ≈ 137 GFLOPs`

**Bytes / call** (read A + read B + write C, fp16):
`(M·K + K·N + M·N)·2 = 3·4096²·2 ≈ 100 MB`

**Arithmetic intensity:** `137e9 / 100e6 ≈ 1370 FLOP/B` — solidly compute-bound.

## First-principles bound

| | |
|---|---|
| H100 SXM fp16 TensorCore peak | ~989 TFLOPS |
| Compute lower bound | `137e9 / 989e12 ≈ 138 µs/call` |
| H100 HBM3 BW | ~3.35 TB/s |
| Memory lower bound | `100e6 / 3.35e12 ≈ 30 µs/call` |
| Roofline ridge AI | `989e12 / 3.35e12 ≈ 295 FLOP/B` |

Workload AI (1370) ≫ ridge AI (295). **Compute-bound.** Same regime as SDPA at this shape.

## Bar to beat

| variant | TFLOPS | % of peak |
|---|---:|---:|
| `pytorch_fp16` (cuBLAS dispatch) | **~640** | 65% |
| `pytorch_compile_fp16` | ~540 | 55% |

cuBLAS at 65% of peak is consistent with NVIDIA's published numbers for fp16 GEMM at 4096³ — vendor kernel is hand-tuned over years. CUTLASS's own profiler typically lands within 2–10% of cuBLAS on this shape.

## v0 — Cooperative + Cluster<2,1,1> + Tile<128,128,64> (310 TFLOPS = 31% of peak)

**Instantiation** (`csrc/cuda/kernels/matmul_hopper.cu`):

```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    Sm90, OpClassTensorOp,
    half_t, RowMajor, /*Alignment*/ 8,            // PyTorch contiguous (M,K) is RowMajor
    half_t, RowMajor, /*Alignment*/ 8,            // PyTorch contiguous (K,N) is RowMajor
    float,                                        // accumulator
    Shape<_128,_128,_64>, Shape<_2,_1,_1>,        // tile, cluster
    StageCountAutoCarveout<...>,                  // pick max stages that fit smem
    KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    Sm90, OpClassTensorOp,
    Shape<_128,_128,_64>, Shape<_2,_1,_1>,
    EpilogueTileAuto,
    float, float,                                 // acc, compute
    half_t, RowMajor, /*Alignment*/ 8,            // C
    half_t, RowMajor, /*Alignment*/ 8,            // D
    TmaWarpSpecializedCooperative,
    LinearCombination<half_t, float, half_t, float>  // alpha=1, beta=0
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, PersistentScheduler>;
```

**Measurement:** 310 TFLOPS, allclose PASSES.

**Diagnosis:** algorithm correct, ~2× behind cuBLAS. Same gap pattern as SDPA at v0. Unlike the FMHA collective, all relevant knobs (cluster, schedule, tile) are exposed via `CollectiveBuilder` template parameters — no fork required to probe them.

**Next hypothesis:** A/B sweep across the obvious knobs:
- `Cluster<4,1,1>` — multicast TMA across 4 CTAs, the same trick cuDNN's FMHA uses (and that we couldn't reach in our SDPA wrapper).
- `Pingpong` schedule — won by 6% on SDPA.
- `Tile<128,256,64>` — bigger N tile, more work per CTA.

## v1 — knob sweep (all variants worse than v0)

**Variants added:** `cute_fp16_c4`, `cute_fp16_pp`, `cute_fp16_t256`. Each is one template-arg flip from the v0 default.

**Measurement** (after fixing the variance bug; see "Bug fix" below):

| variant | TFLOPS | δ vs v0 |
|---|---:|---:|
| v0 default (Coop + Cluster<2,1,1> + Tile<128,128,64>) | **309.94** | baseline |
| `cute_fp16_c4` (Cluster<4,1,1>) | 304.79 | −2% (noise) |
| `cute_fp16_pp` (Pingpong) | 265.00 | −14% |
| `cute_fp16_t256` (Tile<128,256,64>) | 210.75 | −32% |

All three lose. **The opposite of the SDPA story.** Reading the data:

- **Cluster<4,1,1> doesn't help.** Multicast-TMA on B helps when B-load is the bottleneck. At AI=1370 (compute-bound), B reads aren't the limit — the cluster optimization buys nothing.
- **Pingpong loses by 14%.** Pingpong's "alternate two warpgroups across output tiles" pattern wins when there's a softmax/GEMM overlap to extract (SDPA). For pure matmul there's no second op; Cooperative's tighter producer amortization wins.
- **Tile<128,256,64> loses by 32%.** 4096/128 × 4096/256 = 512 CTAs vs 1024 CTAs for v0. The 132 SMs see 3.9 waves vs 7.7 waves — fewer CTAs underutilize the GPU at this shape.

**Diagnosis:** v0 is already a good template choice for this compute-bound shape. The SDPA-era intuitions don't transfer to plain matmul. Need to either (a) profile to find a non-obvious bottleneck or (b) try a knob we haven't tested (B-layout TN-style, larger K tile).

## Bug fix during v1 — per-call cudaMalloc against PyTorch's caching allocator

While running the v1 sweep, the bench numbers were wildly unstable across runs:

| run | `cute_fp16` default | `cute_fp16_pp` | `pytorch_fp16` |
|---|---:|---:|---:|
| #1 | 307.15 | — | 649.70 |
| #2 | 46.64 | 185.29 | 643.60 |
| #3 | 265.43 | 60.80 | 660.79 |
| #4 (post-fix) | 309.94 | 265.00 | 630.95 |

PyTorch's bar was rock-stable (~640–660 TFLOPS, ~3% spread), but our cute kernels swung 6× across runs. The kernel itself is deterministic — the variance came from the launcher.

**Root cause:** the launcher called `cudaMalloc(workspace_size)` and `cudaDeviceSynchronize()` per kernel call. PyTorch uses its caching allocator (cudaMallocAsync-backed); our raw `cudaMalloc` competes with it for the same GPU memory pool, and the driver allocator's response time is non-deterministic at fp16-GEMM timescales (~100–500 µs alloc cost vs ~200–500 µs kernel time).

**Fix:**
1. Cached workspace in a static, sized up on demand, never freed (leaks at process exit, fine for a kernel library).
2. Removed the redundant internal `cudaDeviceSynchronize()` — Python's `torch.cuda.synchronize()` already wraps the timed function, and any subsequent CUDA op serializes via stream 0 anyway.

```cpp
static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;
static void* ensure_workspace(size_t bytes) {
    if (bytes > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, bytes);
        g_workspace_size = bytes;
    }
    return g_workspace;
}
```

Same fix applied to `attn_hopper.cu` (workspace + LSE buffer).

**After fix:** v0 default holds at ~310 TFLOPS run-over-run with the same ~3% noise band as PyTorch.

This is a footgun worth never repeating. Saved as `memory/feedback_per_call_cuda_malloc.md`.

## Where we are

**Best variant:** v0 default ≈ 310 TFLOPS = 31% of fp16 peak = **0.49× of cuBLAS bar.**

**Knob ceiling reached** for the simple sweep (cluster M, schedule, N-tile size). The remaining 2× to cuBLAS is real and not closeable from these knobs.

## Untried — promising next steps

These are concrete experiments with hypotheses, not desperation moves.

1. **B layout TN-style (LayoutB = ColumnMajor).** WGMMA fp16 atoms prefer K-major B. Our RowMajor B has K stride = N (large), N stride = 1 — N-major. Switching requires either pre-transposing B in the bench (`b_torch_fp16.t().contiguous()`) or accepting that the kernel ABI changes shape semantics. **Expected gain: +5–15%.** This is the one knob with a clear CUTLASS-vs-cuBLAS hypothesis attached.

2. **Larger K tile (`Tile<128,128,128>`).** Doubles WGMMA K depth per stage; reduces stage count proportionally but improves MMA throughput per stage. **Expected gain: ±10%, sign unclear.**

3. **`Tile<256,128,64>`** (M instead of N bigger). Different occupancy/wave pattern. Might find the sweet spot for 132 SMs at 4096³. **Expected gain: ±10%.**

4. **ncu profile** of `cute_fp16` vs cuBLAS — same playbook as SDPA. Would identify the actual bottleneck (tensor pipe saturation? smem throughput? register pressure? wave imbalance?) and replace guessing with data.

The cuBLAS gap at 0.49× is large enough to be worth profiling before committing to (1)–(3). For SDPA, profiling found the bottleneck (cluster shape) was outside our reach. For matmul, all knobs are reachable, so a profile-driven decision is more likely to land a real win.

## Build / bench commands

```
make build           # CPU container, ~1.5 min
make bench-mm-cute   # H100, runs cute_fp16 + 3 variants + pytorch baselines at M=N=K=4096
```

(No `profile-mm-cute` target yet; would mirror `profile-cute` for SDPA.)
