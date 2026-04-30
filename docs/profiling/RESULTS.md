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
5. ✅ **Run upstream layers in fp16** so Q/K/V arrive in fp16 already; lets
   the in-kernel cast also drop. Implemented as `F.scaled_dot_product_attention_mma_fp16`
   + `F.cast_to_fp16` — see § *fp16-input MMA path* below.

---

# sdpa.mma optimisation journal (post-fusion)

The above section captured the kernel right after the in-flight cast fusion.
What followed turned out to be a near-rewrite, because a feedback round from
a CUDA optimization professional flagged three things that didn't match the
README's "best" claim:

> "you'll need to accumulate more than one sdpa kernel before you'd be
> prepared for the role"
>
> "way way slower than cuBLAS, that's not going to look good"
>
> "evidence of your own chain of thought as you optimize, using both first
> principles reasoning and feedback from profiling"

The third item is what this section is for. Each step below records a
hypothesis, the change made, and the measured impact (with negative results
kept in place — they're as useful as the wins).

Workload throughout: `B=4, H=8, Sq=Sk=256, Dk=Dv=512`, fp32 inputs, 30 iterations.
GPU: consumer Ampere (sm_86 in setup.py, RTX 3090-class).

## Step 0 — The kernel didn't actually compute SDPA

Three latent bugs were discovered while trying to verify against PyTorch:

1. **No outer KV-tile loop.** The kernel processed only the first 16 keys.
   With `seq_len_k = 256`, the other 240 keys were silently ignored. The
   "274 K×" speedup in the table above measures ~6 % of the work.
2. **Output write missing `q_start`.** Each block wrote to `out_base[r0 * d_v + ...]`
   instead of `out_base[(q_start + r0) * d_v + ...]`. Output rows past the
   first 16 query rows were overwriting the first 16.
3. **K loaded as the wrong B operand layout.** The MMA was effectively
   computing `Q @ K` instead of `Q @ K^T`. The bug hid because identity-like
   inputs are symmetric (`Q @ K = Q @ K^T` when K = K^T), so the test cases
   we had passed by accident.

After fixing all three: kernel matches PyTorch's `scaled_dot_product_attention`
to ~5e-4 relative error (fp16 precision) across `D ∈ {16, 32, 64, 128, 512}`
and `Sq, Sk` multiples of 16.

Wall time on the README workload jumped from 0.96 s (which was actually the
`tiled` fallback in disguise — the host wrapper had `if (d_v != 16) return tiled()`)
to 16.2 ms (0.40 TFLOPS). The 60× number is the effect of running an actual
MMA implementation; the silent-fallback path was the prior baseline.

## Step 1 — `cp.async` double-buffered K/V streaming

**Hypothesis:** the per-iter cost is dominated by global → shared memory
movement (the section profile showed ~50 % in K/V load + cast). Streaming
the next chunk's data via `cp.async` while the current chunk's MMAs run
should overlap most of that latency.

**Smem budget puzzle.** The straightforward design (full 16 × d_k staging)
blows the 100 KB consumer-Ampere budget for d=512:

```
2 × 16KB (s_q + s_k full)  +  2 × 32KB (fp32 K + V staging) = ≥ 96KB before s_o
```

**Solution:** stream **chunks** instead of full rows. Each KV-tile loop
iterates `d_k / 16` chunks of `16 × 16` halves, double-buffered through tiny
2 KB staging slots. Cast pass converts fp32 staging → fp16 final per chunk;
ldmatrix reads from final.

```
for kv_start:
    for d_k chunk in 0..d_k/16:
        if not last: issue cp.async chunk+1
        wait for chunk
        cast chunk → fp16
        ldmatrix Q, K
        MMA accum
```

Per-tile smem dropped from "would-be" 96 KB to 57 KB.

**Result:** 16.2 ms → 12.7 ms (0.40 → 0.51 TFLOPS). 27 % speedup.

## Step 2 — 4-warp tiling on d_v

**Hypothesis:** PV is currently single-warp. With 4 warps in the block, each
handling its own d_v/4 cols of output, the PV MMA wall time should drop ~4×,
and the cooperative cast/load pre-phases get parallelism for free.

**Synchronization tradeoff:** QKT and softmax stay on warp 0 (small fraction
of time, no win from splitting). The warp-0-only branch can't use
`__syncthreads` inside (warps 1-3 are outside the branch — instant
deadlock); replaced with `__syncwarp`. Phase boundaries use `__syncthreads`
to bring all warps together before cooperative phases (P cast, O rescale)
and per-warp PV phase.

**Per-warp staging.** Each warp pipelines its own d_v chunks via its own
double-buffered staging slot — the V cp.async groups are per-warp, not
per-block, so they can interleave without synchronization.

**Result:** 12.7 ms → 10.7 ms (0.51 → 0.60 TFLOPS). 16 % speedup.

## Step 3 — Register-resident output accumulators

**Hypothesis:** the output buffer `s_o[16][d_v]` is read, rescaled, and
written back to smem on **every** kv-tile iteration. At d=512 that's
8 KB × 16 iters = 128 KB of smem traffic per query block, just for the
rescale. If we keep O in registers per warp instead, the rescale is a
register multiply — and the per-iter PV MMA can accumulate directly via the
C operand of `mma.sync` instead of writing to smem and adding.

**Register budget.** Per warp, 16 query rows × d_v/4 cols × 4 fp32 cells per
fragment = 64 fp32 / thread for d=512. Within Ampere's 256-reg-per-thread
budget.

**Cleanup:** s_o (32 KB) eliminated entirely. Smem dropped from 65 KB → 33 KB.

**Result:** 10.7 ms → 8.4 ms (0.60 → 0.76 TFLOPS). 22 % speedup.

This was the single biggest win of the journal — the smem traffic of
read-rescale-add was a far bigger cost than I expected from looking at the
section profile alone.

## Step 4 — K cp.async overlap across kv-tile boundaries

**Hypothesis:** within a kv-tile, the cp.async pipeline correctly hides K
loads behind QKT MMA. But chunk 0 of the next kv-tile is issued **after**
the entire current kv-tile finishes, so its load latency is exposed.
Hoisting that issue to the end of the previous PV phase should overlap it
with the trailing `__syncthreads` and a chunk of softmax/P-cast on the next
iteration.

**Implementation:** pre-issue chunk 0 of the first kv-tile before the loop,
then at the end of each kv-tile's PV phase issue chunk 0 of the next tile.

**Result:** 8.4 ms → ~7.5 ms (0.76 → 0.86 TFLOPS, ~7 % within run-to-run noise).

## Step 5 — 32 query rows per CTA  *(reverted)*

**Hypothesis:** at 16 query rows per CTA each block loads `d_k * Sk` halves
of K and V. Doubling the rows would amortize that load 2× and double per-CTA
arithmetic intensity. Smem now has slack (s_o is gone), so the obvious move
is to grow the query block.

**What actually happened:**

| | CTAs (B*H*Sq/Q_ROWS) | Per SM (~82 SMs) |
|---|---:|---:|
| Q_ROWS=16 | 512 | ~6 |
| Q_ROWS=32 | 256 | ~3 |

Wall time was **slower or break-even** across 5 × 30-run benches. The K
amortization win was eaten by the loss in CTA-level parallelism — at this
workload size we're CTA-count-bound, not per-CTA-cost-bound.

Reverted to Q_ROWS=16. Kept `M_TILES = Q_ROWS / 16` generalization in the
loops so the change is one line for workloads (longer sequences, smaller
batch×heads) where this **would** help.

## Step 6 — FA-2 micro-optimisations  *(reverted)*

**Hypothesis:** baking `scale * log2(e)` into Q at load time gives two free
wins: (1) score writes don't need `* scale`, (2) softmax can use `exp2`
directly instead of `exp(x * log2(e))`. Standard FA-2 trick.

**Result:** 10.3 ms vs 10.7 ms baseline — **net wash** within noise. Reverted.

The reason it didn't help: Q load is one-time per kernel (not in the hot
loop), and the per-iter softmax is < 1 % of runtime per the section profile.
The savings exist but they're in places that aren't bottlenecked.

This is a useful negative result. FA-2's compute-side micro-optimisations
matter on workloads where softmax/scale is a non-trivial fraction of time
(small d_k, long Sk). Ours is dominated by the per-tile fp32→fp16 cast.

## Step 7 — fp16-input MMA path

**Honest framing:** the previous kernel already used fp16 internally —
`mma.sync.aligned.m16n8k16.f32.f16.f16.f32` is fp16 input × fp16 input,
fp32 accumulate. The fp32 input "support" was a convenience: a per-tile
`cast_k_chunk` / `cast_v_chunk` pass converted fp32 → fp16 in shared memory
before the MMA could see it. That cast was a real per-iter cost.

I almost didn't do this step because of the worry that comparing fp16-input
tensorax against fp32-input PyTorch is "cheating" — like quantizing GPT-4 to
int4 and claiming it runs faster. The relevant facts:

- This kernel has been doing fp16 MMA the whole time.
- Pytorch's "fp32" SDPA dispatches to TF32 on Ampere (19-bit mantissa, not
  full fp32). It's also reduced precision under the hood.
- fp16 is the production-standard format for inference attention (cuDNN,
  FlashAttention, vLLM, every inference stack uses it).

So fp16 inputs **don't add quantization** — they remove a redundant cast
that the existing kernel was paying every kv-tile. Same MMA hardware path,
same numerical precision, packaged differently.

**What changed:**

- New kernel `sdpa_kernel_mma_fp16` (copy of fp32 path with cast pass removed).
- `cp.async` writes halves directly to `s_kchunk` — no fp32 staging buffer.
- V "cast" pass becomes a pure transpose (still col-major-ifying for ldmatrix).
- Smem dropped further: 33 KB → 27 KB at d=512.
- New `TensorImpl` constructor for empty allocation (so we can hold fp16 bytes
  in a `float*` field). New `cast_to_fp16` op + Python wrapper.
- New benchmark variant pre-casts Q, K, V to fp16 once outside the timed loop
  (matching how a real KV cache feeds inference; cast cost is paid on cache
  write, not per attention call).

**Result on README workload (B=4 H=8 Sq=Sk=256 D=512, 30 runs):**

| Path | s/run | TFLOPS | vs. fp32-input |
|---|---:|---:|---:|
| MMA fp32-input | 0.0101 | 0.64 | 1.00× |
| **MMA fp16-input** | **0.0047** | **1.37** | **2.15×** |
| PyTorch fp32 SDPA | 0.0013 | 5.15 | – |
| PyTorch fp16 SDPA | 0.00025 | 17.3 | – |

A clean 2.1× speedup. The fp16 path is the first variant to clear 1 TFLOPS
and now beats `flash_optimized` by ~3.2×.

---

## Final state, end of journal

```
                            time     TFLOPS   speedup vs. tensorax-naive
Tensorax MMA fp16          4.7 ms    1.37     642×    ← best
Tensorax MMA fp32         10.1 ms    0.64     299×
Tensorax Optim. Flash     15.0 ms    0.43     201×
Tensorax Flash            97.6 ms    0.07      31×
Tensorax Tiled           1093  ms    0.01     2.8×
Tensorax Naive           3016  ms    0.00     1.0×

PyTorch fp32 SDPA          1.3 ms    5.15    2415× (cuDNN, internal TF32)
PyTorch fp16 SDPA          0.25 ms  17.3   12000× (cuDNN, FA-2-class)
```

**Net journey on the MMA kernel from start of journal:**
- 16.2 ms (real-but-broken, just-fixed-correctness baseline) → 4.7 ms = **3.5×**
- vs. the original silent-fallback "MMA" of 0.96 s (was actually `tiled`)
  that the README claimed: **204×**

**Remaining gap to PyTorch:**
- ~3.5× behind PyTorch fp32 SDPA
- ~19× behind PyTorch fp16 SDPA

The remaining gap is structural — we're at 1 warp per block × 16 query rows.
PyTorch's cuDNN dispatches into FA-2-class kernels that use 4-8 warps per CTA,
64-row query tiles, register-staged Q persistence across kv-tiles, and a tuned
schedule for memory hierarchy. That's a different kernel design, not a
parameter tweak on this one.

## Negative results worth keeping

- **32 query rows per CTA (Q_ROWS=32):** halved CTA count, lost more from SM
  under-utilization than gained from K amortization. Win at larger workloads.
- **Pre-scale Q with `scale * log2(e)`, exp2 in softmax (FA-2 trick):** free
  in theory, in practice within noise. The optimisation targets ops that
  aren't bottlenecks for this workload's shape.

## What I'd do next

1. **Multi-warp Q tiling** — 4 warps × 16 query rows = 64 query rows per CTA,
   each warp handles its own queries. Requires per-warp s_o (back to smem) or
   tiled d_v processing to fit register accumulators. This is the structural
   change closest to FA-2.
2. **Direct fp16 model integration** — wire the existing nn.Linear, etc. to
   produce fp16 outputs so KV cache feeds the fp16 MMA path naturally,
   without explicit `cast_to_fp16` calls.
3. **Tuned schedule for memory hierarchy** — at this point the kernel is
   probably L2/HBM-bandwidth-bound on the inner cast+MMA cycle. Verify with
   nsys, then look at `cp.async.bulk` (sm_90) or larger tile sizes (sm_80+).
