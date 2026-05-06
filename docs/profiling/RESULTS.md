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

---

# Continuation: FA-2 → FA-3 path attempt (post-Step 7)

Picking up the "What I'd do next" #1 (multi-warp Q tiling) and extending it
into the structural FA-3 changes. The plan was three ordered steps with a
benchmark after each so wins/losses are attributable to the change in
isolation. Same workload throughout: `B=4, H=8, Sq=Sk=256, Dk=Dv=512`,
fp16 inputs, 30-run wall-time benchmark via `benchmarks/attn_benchmark.py`.

Pre-attempt baseline (= Step 7 result): **4.7 ms / 1.37 TFLOPS**.

## Step 8 — FA-2 multi-warp Q tiling (32 queries / CTA, 2 groups × 4 warps)

**Hypothesis:** at 16 queries / CTA, each block reloads the full K and V
tensors for its 16 queries. Doubling queries / CTA and grouping warps so
that two query groups share a single K load should halve K bandwidth per
query and engage all 8 warps in PV. This is the structural change closest
to FA-2's multi-warp Q layout.

**Design:**

- 256 threads / 8 warps per block.
- 32 query rows / CTA, partitioned into **2 groups of 16** queries.
- Group 0 = warps 0–3, group 1 = warps 4–7.
- Warp 0 owns the K cp.async pipeline for the whole block; both groups
  consume the same `s_kchunk` (broadcast via `__syncthreads`).
- Group leaders (warps 0 & 4) do their group's QKT and softmax in parallel.
- All 8 warps participate in PV: each warp owns a `d_v / 4` column slice of
  the output for its group. Per-warp register accumulators (`oacc0[16][4]`,
  `oacc1[16][4]`).

**Result:** 4.7 ms → **4.7 ms / 1.38 TFLOPS** — net flat (+0.7 % within
noise).

**Why it didn't pay off:** doubling queries / CTA halved the grid (512 →
256 CTAs across `B × H × Sq/Q_ROWS`). At this workload size we're already
CTA-count bound (Step 5's negative result, restated). The K-share win and
the gained-PV-parallelism cancelled with the lost SM occupancy. Step 5 had
already taught us this; Step 8 confirms it for the multi-warp variant too.

Kept the kernel structure in place because Steps 9 & 10 build on it.

## Step 9 — FA-3 warp specialization (V[0] preissue by idle warps)

**Hypothesis:** during QKT and softmax, only the two group leaders (warps
0 & 4) are active — the other 6 warps idle. In FA-3 spirit, give those
idle warps memory work: have warps 1–7 preissue `cp.async` for their
slice's V[0] at the top of each kv-tile, so the V[0] cp.async runs in
parallel with QKT + softmax + p-cast + corr. The first PV iteration's
V wait then collapses (V[0] already landed by then).

**Synchronization detail:** warp 0 was kept on its existing flow because
it already owns K[0]'s cp.async group from the prior tile's tail. Adding
a V group between K[0] and the next K issue would break the K pipeline's
`cp_async_wait_group<1>` semantics (the wait would conflate K and V).

**Result:** 4.8 ms / **1.35 TFLOPS** — small regression (−1.5 %).

**Why it didn't pay off:** the existing PV inner pipeline already
prefetches V[dvc+1] while computing on V[dvc]. The first-iteration V[0]
wait was *not* actually exposed; the V[0] cp.async time was already
hidden by the in-loop pipeline. The preissue added an extra
`cp_async_commit` per tile per warp without a corresponding latency
reduction. Net: noise + small bookkeeping cost.

Kept the change in place to feed Step 10's premise (idle warps doing
memory work).

## Step 10 — FA-3 softmax/MMA pingpong (early K[T+1] preissue)

**Hypothesis:** today the K[T+1][0] preissue lives at the *end* of tile
T's PV phase — its only overlap window is the trailing `__syncthreads`
plus QKT(T+1)'s setup. Hoisting it to *right after QKT(T) ends* gives it
a much larger overlap window: score store + softmax + p-cast + corr +
the entire PV phase of tile T. By the time QKT(T+1) starts, K[T+1][0]
should be solidly landed.

**Implementation:** moved the `if (warp_id == 0 && kv_start + 16 <
seq_len_k) { issue_k_chunk(kv_start + 16, 0, 0); cp_async_commit(); }`
from end-of-tile to immediately after the QKT loop terminates.

**Result:** 5.0–5.1 ms / **1.27 TFLOPS** (3-run median) — clear
regression (−7.3 %).

**Why it backfired:** the new ordering puts K[T+1][0] in warp 0's
cp.async group queue *before* PV's V groups for the current tile.
PV's `cp_async_wait_group<1>` then blocks on K[T+1][0] in addition to
V groups. softmax + corr + p-cast turned out to be too short to fully
hide K[T+1][0]'s cp.async latency (16 × 16 fp16 chunk, ~256 bytes,
typical L2-fetch cycles ≫ the few hundred cycles of softmax for 16
rows). Net cost (PV's wait dragged out by K) > net benefit (no save
because the existing end-of-tile preissue was already hiding behind
`__syncthreads` adequately).

This is the cleanest example in the journal so far of FA-3-on-Hopper
patterns failing to translate to Ampere. On Hopper, mbarriers give each
async op its own named barrier, so a softmax/MMA pingpong can wait on
the right thing. On Ampere, every cp.async on the same thread is in
the same FIFO group queue — adding K and V to the same queue means
`wait_group<N>` can't distinguish them.

## Updated final state

| Path | time | TFLOPS | Δ vs Step 7 |
|---|---:|---:|---:|
| Step 7 (pre-attempt baseline)        | 4.7 ms | 1.37 | — |
| Step 8 (multi-warp Q tiling)         | 4.7 ms | 1.38 | flat |
| Step 9 (V[0] preissue specialization)| 4.8 ms | 1.35 | −1.5 % |
| Step 10 (K[T+1] pingpong preissue)   | 5.0 ms | 1.27 | −7.3 % |
| PyTorch fp16 SDPA (cuDNN reference)  | 0.25 ms | 17.3 | — |

Best end-of-pass: still Step 7 / Step 8 at ~1.38 TFLOPS.

## Why the FA-3 patterns didn't translate to Ampere

The FA-3 wins on Hopper come from features Ampere lacks:

- **TMA** — async DMA engine decoupled from compute warps. A producer
  warp can issue copies without using compute issue slots.
- **WGMMA** — warpgroup-level async MMAs. Compute warps can launch MMA
  and continue, instead of stalling on `mma.sync`.
- **mbarriers** — named async barriers, so each producer/consumer
  channel has its own wait point.

On Ampere with sync `mma.sync` + per-thread `cp.async.commit_group` /
`cp.async.wait_group`, the same FA-3 patterns don't compose:

1. **Producer/consumer warp split.** Every cp.async group lives on the
   issuing thread's own per-thread queue. A "producer warp" that
   issues for a "consumer warp" can't transfer the wait barrier — the
   consumer either has to re-issue or fall back to `__syncthreads`,
   which doesn't wait on cp.async.
2. **Pingpong between phases.** Overlapping different phases puts more
   cp.async groups in the same per-thread queue. `wait_group<N>` then
   waits on *everything* outstanding, so the K phase ends up waiting
   on V (Step 10's failure mode).

## Where the time is actually going (revised)

Register pressure dominates and was misjudged in Step 8's design:

- `oacc0[MAX_CHUNKS=16][4]` + `oacc1[MAX_CHUNKS=16][4]` = 128 fp32 /
  thread for output accumulators (≈ 150 regs / thread total).
- 256 threads × 150 regs = ~38 K registers / CTA, vs Ampere's 64 K / SM.
- That caps us at **1 CTA / SM** — well below the smem-derived ceiling.
- For the workload here, only `dv_chunks_per_warp = d_v / 64 = 8` of the
  16 `MAX_CHUNKS` slots are ever used. Halving `MAX_CHUNKS` to 8 (with
  a host-side guard `d_v ≤ 512`) would free ~64 regs / thread.

Step 8's design *added* register pressure (more queries / CTA = more
PV accumulator state per warp), which is part of why it didn't gain
despite the K-share win.

## Negative results from this pass

- **Multi-warp Q tiling at this workload (Step 8):** doubled
  queries / CTA halved CTA count. Net flat. Same root cause as Step 5's
  reverted Q_ROWS=32 — workload is CTA-count-bound. *Will* help at
  larger Sq / smaller batch×heads.
- **Idle-warp V[0] preissue (Step 9):** the optimization had nothing
  to hide behind. The existing in-PV-loop prefetch was already hiding
  V latency.
- **Early K[T+1] preissue / pingpong (Step 10):** broke the PV
  cp.async wait semantics. Whenever you push a phase's preload
  earlier in a per-thread cp.async queue, you have to verify nothing
  later in the queue is going to `wait_group` on it.

## What I'd actually do next (revised)

1. **`MAX_CHUNKS` reduction (16 → 8) + host-side guard** for `d_v ≤ 512`.
   Halves the output-accumulator array. Highest expected ROI given
   the diagnosis above — directly attacks the 1-CTA/SM ceiling.
2. **Persistent CTA scheduling** — keep CTAs alive across multiple Q
   tiles so launch overhead amortizes. Helpful when CTA count is the
   limiter (which it is here).
3. **Bigger d_k tile (32 or 64)** — fewer dk-chunks per QKT loop,
   fewer `__syncthreads` per tile. Costs smem; revisit budget after
   #1 frees registers.
4. **Re-run Step 8 after #1 lands.** With register footprint halved,
   the multi-warp Q tiling may stop regressing on occupancy and the
   K-share win could finally show up.
5. **Skip further FA-3 structural attempts** until on Hopper. The
   per-thread cp.async-group queue is a hard ceiling for true
   producer/consumer / pingpong on Ampere; the wins would need
   mbarriers or a redesign that uses smem-based handshakes instead
   of cp.async groups (more complex than the wins justify here).

---

## Step 11 — Real FA-2 split-Q (parallel QKT across all warps)

**Diagnosis revisited.** Step 8's "multi-warp Q tiling" was multi-warp
in name only: two group leaders did QKT serially while the other six
warps idled at `__syncthreads`. So the K-share win had nothing to land
against — per-row QKT throughput was unchanged from Step 7. The flat
result wasn't an "occupancy vs CTA-count" wash; it was a *missing*
parallelism. Same shape for the 4-warp Step 7 baseline: `if (warp_id
== 0)` gated the whole QKT loop, so 3 of 4 warps idled.

**Fix (FA-2 split-Q, properly):**

- `Q_ROWS = 64`, organised as 4 warps × 16 rows. Each warp owns a
  contiguous 16-row slice (`q_row_off = warp_id * 16`).
- All 4 warps run `mma.sync` simultaneously against the same
  `s_kchunk`. Warp 0 still owns the K cp.async pipeline; the other
  three just consume after `__syncthreads`.
- Softmax moves into registers per warp. Row-max / row-sum reduce
  across the 4 lanes that share a row via `__shfl_xor_sync` (offsets
  1 and 2). No `s_scores` round-trip.
- `s_p` is gone. The MMA m16n8k16 C-output layout for `c0 = (lane%4)*2`
  matches the A-operand layout for the next MMA with `k0 = c0`, so P
  is packed straight from the QKT acc registers into `reg_p` —
  no smem write/ldmatrix.
- PV is per-warp row-resident: each warp does its 16 rows × full
  `d_v`. V is loaded cooperatively (warp 0 issues, all warps consume),
  double-buffered along `d_v` chunks. The old per-warp V layout
  (`s_vstage[4][2][16][16]` / `s_vchunk[4][2][16][16]`) collapses to a
  single `[2][16][16]` shared buffer.
- Running m / l live in registers across the kv loop (`m_r0`, `m_r1`,
  `l_r0`, `l_r1` per lane). `s_m` / `s_l` only used at the final
  output-write sync point.

**Result** (B=4, H=8, Sq=Sk=256, Dk=Dv=512, fp16, 30-run wall time,
median of 4 invocations): **4.1 ms / 1.59 TFLOPS** — **+16 % TFLOPS /
−13 % time vs Step 7**, the first real win in this continuation.
(Run-to-run spread: 1.54–1.64 TFLOPS; a single best-of-one read 1.73,
which is noise — steady state is ~1.6.)

**Why this one paid off:**

- QKT now uses 4× the MMA throughput per CTA (4 active warps vs 1).
- Q_ROWS quadrupled (16 → 64), so each CTA covers 4× the queries
  with the same QKT *time-per-tile*. Grid shrinks 4× → K bandwidth
  per call drops 4× (each global K row is loaded by ¼ as many CTAs).
- The smem footprint actually *shrunk* despite Q_ROWS growing,
  because the per-warp V duplication and the `s_p` / `s_scores`
  buffers all went away. So no occupancy hit from the larger Q tile.
- Register pressure for the output accumulator is the same per warp as
  Step 7's PV split (16 rows × `d_v`/4 in Step 7 = 16 rows × `d_v`
  here, just spread over different cols). `MAX_CHUNKS` was raised
  16 → 32 to cover `d_v ≤ 512`.

**Validation:** allclose vs PyTorch fp16 SDPA passes across
`{Sq,Sk} ∈ {64, 80, 128, 256}`, `{Dk,Dv} ∈ {64, 128, 512}`, max abs
error ~3e-4. The non-multiple-of-64 `Sq=80` case exercises the
out-of-range Q-row zero-padding path.

## Updated final state (post-Step 11)

| Path | time | TFLOPS | Δ vs Step 7 |
|---|---:|---:|---:|
| Step 7 (pre-attempt baseline)        | 4.7 ms | 1.37 | — |
| Step 8 (multi-warp Q tiling)         | 4.7 ms | 1.38 | flat |
| Step 9 (V[0] preissue)               | 4.8 ms | 1.35 | −1.5 % |
| Step 10 (K[T+1] pingpong)            | 5.0 ms | 1.27 | −7.3 % |
| **Step 11 (FA-2 split-Q parallel QKT)** | **4.1 ms** | **1.59** | **+16 %** |
| **Step 12 (oacc pack + lazy correction)** | **3.7 ms** | **1.76** | **+28 %** |
| PyTorch fp16 SDPA (cuDNN reference)  | 0.25 ms | 17.3 | — |

Step 11 is the first step in this continuation that produced a real
gain, and it does so by fixing what was actually wrong with Step 8 —
not by stacking more cp.async tricks. Step 12 then attacks the
register-pressure side of the same diagnosis.

## Step 12 — oacc packing + lazy output correction

After Step 11 the per-warp register footprint is the bottleneck:
each warp owns 16 rows × full d_v of fp32 output state (256 fp32 /
lane for `d_v=512`), capping occupancy at 1 CTA / SM. Two
register-side fixes from the journal's "what to do next":

**Pack `oacc0` and `oacc1` into one array.** The two-array layout was
an artefact of `mma.sync.m16n8k16` producing N=8 per MMA — running it
twice gives N=16 in two halves. That's a layout choice, not a memory
fact: the same 256 fp32 / lane lives in `oacc[MAX_CHUNKS][2][4]` with
`[2]` being the N-half. Same footprint, but a single indexable array
gives the compiler one continuous live range to reason about, and the
rescale loop (which touches every chunk by the same per-row corr) is
now a single pass over one array.

**Lazy output correction.** The rescale loop is `MAX_CHUNKS × 8 = 256`
register multiplies *per kv-tile* — i.e. proportional to the kv loop,
not amortised. FA-2's lazy trick: only rescale when `tile_max - row_max
> threshold` (`SOFTMAX_MAX_UPDATE_THRESHOLD = 0.5`, already in the file
but only used by `flash_optimized`). The rescale gates per row, since
the two rows a lane owns (r0 and r1) update independently — so a tile
that bumps r0 but not r1 only multiplies half the array.

Mathematically safe: `delta < 0.5` means `used_max` lags the true row
max by ≤0.5, so `p = exp(score - used_max)` is bounded by `exp(0.5) ≈
1.65`. The softmax denominator `l` carries the same bias and cancels
it at the final divide.

**Result** (B=4, H=8, Sq=Sk=256, Dk=Dv=512, fp16, 100-run wall time,
median of 5 invocations): **3.7 ms / 1.17 TFLOPS** (new bench formula)
≈ **1.76 TFLOPS** in the old `4·B·H·S²·Dk + 2·B·H·S²·Dv` formula.
**+11 % time reduction over Step 11**, **+28 % vs Step 7** baseline.

Validation: max abs error vs PyTorch fp16 SDPA holds at ~3-4e-4 across
`{Sq,Sk} ∈ {64, 80, 256}`, `{Dk,Dv} ∈ {64, 128, 512}` — the threshold
introduces no measurable degradation at fp16.

(Note: the benchmark's TFLOPS formula changed mid-stream from
`4·B·H·S²·Dk + 2·B·H·S²·Dv` to `2·B·H·S²·(Dk+Dv)`. The numbers in
this table use the *original* formula consistently for comparability;
the new-formula reading for Step 12 is 1.17 TFLOPS.)

## Step 12 postscript — `ptxas -v` says the register diagnosis was wrong

After Step 12 I ran `nvcc -Xptxas=-v` on the kernel to confirm the
"freed enough registers for 2 CTAs/SM" intent. Result:

```
sdpa_kernel_mma_fp16:
  40 registers / thread
  1024 bytes stack frame, 0 spill stores, 0 spill loads
  1 barrier, 412 bytes cmem[0]
```

Three corrections to the writeup above:

1. **`oacc` is not in registers — it's in local memory.** The 1024 B
   stack frame is exactly `oacc[32][2][4] × sizeof(float)`. ptxas
   demoted it to a stack-allocated array (cached in L1) because the
   PV loop indexes `oacc[dvc][...]` with a runtime variable. Without
   `#pragma unroll` on the `dvc` loop, the compiler can't pin each
   slot to a fixed register, so it falls back to a local array.

2. **Register count is not the occupancy limiter at this workload.**
   Used: 40 regs / thread × 128 threads = 5120 regs / CTA, vs sm_86's
   65536 regs / SM (~12 CTAs / SM if it were the limit). We're at 1
   CTA / SM because of **smem**: at `d_k = d_v = 512`, `s_q` alone is
   `64 × 512 × 2 B = 64 KB`, plus K/V staging — total ~68 KB / CTA on
   a 100 KB / SM part. That's a smem cap, not a register cap.

3. **Step 12's win is real but mislabelled in mechanism.** The
   "256 fp32 / lane register multiplies per kv-tile" was actually
   256 local-memory load/multiply/store ops against L1 — not register
   ALU. The lazy-correction threshold cut **L1 traffic** on the
   rescale, which is why the wall-time improvement landed; the same
   gain doesn't help register pressure (since there wasn't any).

The earlier journal sections (Step 8 onward) made the same misread
when they cited "150 regs / thread total" and "1 CTA / SM from
registers" — neither was measured with `-Xptxas=-v`, both were
register-pressure inferences from array sizes. Should have run ptxas
in Step 5.

## Step 13 — pin `oacc` into registers via templated `DV_CHUNKS`

The fix the postscript pointed at: `#pragma unroll` alone wasn't enough
because `dv_chunks = d_v / 16` is a runtime variable. Made the kernel
a template `template <int DV_CHUNKS>`, sized `oacc[DV_CHUNKS][2][4]`
exactly, and unrolled both the rescale and PV loops by `DV_CHUNKS`.
The dispatcher in `sdpa_mma_fp16_cuda` switches on `d_v / 16` and
launches one of `{4, 8, 16, 32}` instantiations. Same kernel body —
just a compile-time bound on the loops oacc lives inside.

`ptxas -v` after the change:

| `DV_CHUNKS` | `d_v` | stack frame | spills | registers |
|---|---:|---:|---:|---:|
| 4  |  64 | **0 B** | 0 / 0 | 72 |
| 8  | 128 | **0 B** | 0 / 0 | 128 |
| 16 | 256 | **0 B** | 0 / 0 | 168 |
| 32 | 512 | 168 B | 588 / 452 | 255 (cap) |

For `d_v ≤ 256`, `oacc` is fully register-resident — the local-memory
stack frame is gone, no spills. `d_v = 512` runs into the 255-reg /
thread cap and spills 588 B back to local memory — but the unrolled
access pattern still wins, because the live values now use direct
register addressing instead of indexed local-memory loads/stores for
the parts that *do* fit.

**Results** (B=4, H=8, Sq=Sk=256, fp16, 100-run wall time, median of 3
invocations; PyTorch fp16 SDPA = cuDNN reference):

| `d_k = d_v` | Step 12 | **Step 13** | Δ time | PyTorch fp16 | vs PyTorch |
|---|---:|---:|---:|---:|---:|
|  64 | — | **0.20 ms** | — | 0.19 ms | **1.05×** |
| 128 | — | **0.30 ms** | — | 0.28 ms | **1.07×** |
| 256 | — | **0.86 ms** | — | 0.67 ms | 1.28× |
| 512 | 3.7 ms | **2.0 ms** | **−46 %** | 1.10 ms | 1.82× |

For `d_v ≤ 128` we now match cuDNN. For `d_v = 512` (the benchmark
default) we cut wall time nearly in half despite the partial spill,
and the gap to cuDNN closes from ~3.5× (Step 12) to ~1.8× (Step 13).

The earlier "register pressure caps you at 1 CTA / SM" diagnosis was
*the wrong frame entirely* — Step 13's win came from getting `oacc`
out of L1 latency, not from freeing registers for occupancy. Smem at
`d_v = 512` still pins us to 1 CTA / SM, but with `oacc` in registers
each CTA does its work with far fewer L1 round-trips per kv-tile.

## Updated final state (post-Step 13)

| Path | time | TFLOPS (old formula) | Δ vs Step 7 |
|---|---:|---:|---:|
| Step 7 (pre-attempt baseline)        | 4.7 ms | 1.37 | — |
| Step 8 (multi-warp Q tiling)         | 4.7 ms | 1.38 | flat |
| Step 9 (V[0] preissue)               | 4.8 ms | 1.35 | −1.5 % |
| Step 10 (K[T+1] pingpong)            | 5.0 ms | 1.27 | −7.3 % |
| Step 11 (FA-2 split-Q parallel QKT)  | 4.1 ms | 1.59 | +16 % |
| Step 12 (oacc pack + lazy correction)| 3.7 ms | 1.76 | +28 % |
| **Step 13 (templated `DV_CHUNKS`)**  | **2.0 ms** | **3.24** | **+136 %** |
| PyTorch fp16 SDPA (cuDNN)            | 1.10 ms | 5.86 | — |

(TFLOPS in the table use the original `4·B·H·S²·Dk + 2·B·H·S²·Dv`
formula for comparability with earlier rows; the bench prints the
newer `2·B·H·S²·(Dk + Dv)` formula, which divides everything by 1.5.)

## Step 14 — apples-to-apples sweep, three optimizations attempted

After Step 13 the next push was three bottlenecks identified from the
roofline + ptxas analysis:

1. **Redundant global K/V reads.** `Q_ROWS=64` × `S=256` means 4 Q-CTAs
   per (b,h) each reading the full K and V — ~4× DRAM redundancy.
2. **`d_v=512` register spill.** ptxas hit the 255-reg cap and spilled
   588 B back to local memory.
3. **Per-tile `__syncthreads`.** `dk_chunks × kv_chunks` = 32 × 16 =
   512 block-wide barriers per CTA at the default config.

### #1: Persistent Q-tile loop (regressed)

Wrapped the kernel body in an outer `for (q_tile = 0; q_tile < 2; q_tile++)`
loop so each CTA processed two consecutive Q-tiles for the same `(b,h)`.
Result: regression across all configs (d=64 +50%, d=128 +70%, d=512 +20%).
Halving the grid from 128 → 64 CTAs at B=4 H=8 dropped CTAs/SM from
~2.8 → ~1.4, killing latency hiding. The L2 reuse benefit didn't recover
the loss. Reverted to `Q_TILES_PER_CTA=1`; loop structure kept as a hook
for proper persistent-CTA work later.

### #2: Outer d_v split for d_v=512 (regressed)

Added `D_V_PASSES` template parameter. For d_v=512, dispatched `<16, 2>`:
two outer passes of 256 cols each, each pass with register-resident `oacc`
(`DV_CHUNKS=16`, **0 stack frame, 0 spills**, 168 regs — verified with
`-Xptxas=-v`). The spill was eliminated.

But: each pass recomputes QKT and softmax for all kv tiles, so total
compute grew ~50%. Wall time at d=512 went 1.91 ms → 2.43 ms (~25%
slower). The local-memory traffic from the spill (~1 MB / CTA across
the kernel call) was cheaper than 50% extra MMA work. Reverted to
`<32, 1>` for d_v=512; kept the `D_V_PASSES` parameter wired up because
it's also the path for d_v=1024.

### #3: Wider d_k chunk for sync reduction (small win at d=512)

Widened `s_kchunk` from `[2][16][16]` to `[2][16][32]` (`K_CHUNK_W=32`).
Inner `dk_chunks` loop halved; each iter does two m16n8k16 MMAs against
the two halves of the K dim. `__syncthreads` count drops 2×.

Result: flat at d∈{64,128,256}, ~4% faster at d=512 (1.91 → 1.83 ms).
Matches the math — at 512 syncs × ~10 cycles each, sync is ~1% of
runtime, so cutting it in half saves <1%, dominated by run-to-run
noise. Kept the wider chunk in place since it costs only ~1 KB more
smem.

### Apples-to-apples sweep across (B, S, d)

The earlier "we beat PyTorch at long seqs" claim from the post-Step 13
spot-check was wrong. The bench's `--only mma_fp16 pytorch` flag
routed PyTorch through the **fp32** path (`pytorch`), not the
`pytorch_fp16` path. fp16 vs fp32 is apples-to-oranges; cuDNN's fp16
path is ~3× faster than its fp32 path, and tensorax's "win" disappears
when matched.

Did a full sweep over `B ∈ {1, 4}`, `S ∈ {128, 256, 512, 1024, 2048,
4096, 8192}`, `d_k = d_v ∈ {64, 128, 256, 512, 1024}` with both kernels
on fp16 inputs. Headline at B=4 H=8:

| d | tensorax peak TFLOPS | PyTorch peak TFLOPS | best ratio |
|---:|---:|---:|---:|
| 64 | 12.6 (S=8192) | 32.5 (S=2048) | 0.38× |
| 128 | 12.6 (S=8192) | 33.9 (S=2048) | 0.40× |
| 256 |  9.5 (S=8192) | 31.5 (S=4096) | 0.32× |
| 512 |  5.0 (S=8192) | 14.6 (S=2048) | 0.39× |
| 1024 | unsupported (smem) | 14.0 | — |

cuDNN is **2.5×-7× faster** across every supported config. The gap is
widest at small problems (PyTorch barely warms up while we still pay
full per-CTA setup) and tightest at long seq + large batch (~0.40×).
Tensorax tops out at ~13 TFLOPS (~15% of GPU peak); PyTorch at ~33
TFLOPS (~39% of peak).

Full sweep CSV: `benchmarks/attn_sweep.csv`. Sweep script:
`benchmarks/attn_sweep.py`.

### What this means for the journal

The Step 13 win (Step 7 → Step 13: 4.7 ms → 2.0 ms at the original
default workload) is real *as an internal speedup over our prior
state* — the kernel is materially better. But the framing in earlier
revisions of this doc that "we match cuDNN at small d_v" was wrong;
the comparison there used fp32 PyTorch numbers labeled as fp16. The
apples-to-apples gap to cuDNN is much wider than I claimed and didn't
close in steps 11-13.

The `d_v=512` and `d=1024` failures plus the consistent 2.5× gap point
to the structural items already listed below as the next pass — none
of which is small.

## Updated final state (post-Step 14, apples-to-apples)

At B=4 H=8 S=256 Dk=Dv=512 (the historical reference workload):

| Path | time | TFLOPS (old formula) | Δ vs Step 7 |
|---|---:|---:|---:|
| Step 7 (pre-attempt baseline)        | 4.7 ms | 1.37 | — |
| Step 8 (multi-warp Q tiling)         | 4.7 ms | 1.38 | flat |
| Step 9 (V[0] preissue)               | 4.8 ms | 1.35 | −1.5 % |
| Step 10 (K[T+1] pingpong)            | 5.0 ms | 1.27 | −7.3 % |
| Step 11 (FA-2 split-Q parallel QKT)  | 4.1 ms | 1.59 | +16 % |
| Step 12 (oacc pack + lazy correction)| 3.7 ms | 1.76 | +28 % |
| Step 13 (templated `DV_CHUNKS`)      | 2.0 ms | 3.24 | **+136 %** |
| Step 14 (#1 reverted, #2 reverted, #3 kept) | 2.0 ms | 3.27 | +137 % |
| **Step 15 (`ldmatrix.x2.trans` + Q pre-scale)** | **1.55 ms** | **4.18** | **+205 %** |
| **PyTorch fp16 SDPA (cuDNN)**        | **0.30 ms** | **21.6** | — |

We're at ~15% of cuDNN's throughput at this workload. Steps 11-13
were real gains *over our prior state* but didn't close the gap to
cuDNN, contrary to what the earlier "match at small d_v" framing
implied.

## Step 15 — drop the V transpose pass + pre-scale Q + exp2 softmax

Two small mechanical wins on top of Step 13/14, both about removing
work on the hot path rather than restructuring the kernel.

### #A: `ldmatrix.x2.trans` for V

The PV phase used to: cp.async V into `s_vstage` (row-major), then
have warp 0 transpose it into `s_vchunk` (col-major), then
`ldmatrix.x2` from `s_vchunk` for the B operand. Replaced with a
direct `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` from
`s_vstage` — the `.trans` qualifier transposes each 8×8 tile during
the load itself.

What this drops:

- An explicit per-`dvc` transpose loop (8 fp16 stores per lane).
- One `__syncthreads` per PV iter (the post-transpose barrier).
- The `s_vchunk` smem buffer (saves 512 B / CTA).
- The `transpose_v_chunk` lambda.

Address pattern note: the existing col-major-style indexing
(`b_col*16 + b_khalf`) does *not* port to `.trans` on a row-major
source. With `.trans`, lanes 0..15 must supply addresses to rows
0..15 of the row-major source, with the second tile (cols 8..15)
selected by adding 8 to the column offset. Got this wrong on the
first attempt and accuracy went to ~1.0 max-abs error; correct
pattern is `s_vstage + smem_row_a*16 + {0, 8}`.

**Result** at B=4 H=8 S=256 (best-of-many timing):

| `d_v` | Step 13/14 | After `.trans` | speedup |
|---:|---:|---:|---:|
|  64 | 0.262 ms | 0.137 ms | **1.91×** |
| 128 | 0.310 ms | 0.244 ms | 1.27× |
| 256 | 0.890 ms | 0.582 ms | 1.53× |
| 512 | 2.040 ms | 1.334 ms | 1.53× |

Big win across all configs. The transpose pass + its sync was a real
fraction of PV time, not noise.

### #B: pre-scale Q + use `exp2.approx`

Folded `scale * log2(e)` (≈ `1/√d_k * 1.4427`) into Q at the cp.async
load — multiply each fp16 element by the constant before storing into
`s_q`. After this, the QKT acc is in log2 space, so:

- The per-tile `acc *= scale` loop disappears (16 fmuls per warp per
  kv-tile).
- `exp_approx(x) = exp2.approx(x * log2(e))` becomes just
  `exp2.approx(x)` — one fmul dropped per exp call. With 8 exp calls
  per kv-tile per lane (in the hot softmax path), that's 8 more fmuls
  saved per kv-tile per lane.

The lazy-correction threshold `SOFTMAX_MAX_UPDATE_THRESHOLD = 0.5`
keeps its numeric value but now means "log2 of the relative skew",
i.e. `exp2(0.5) ≈ 1.41` slack on `used_max` instead of
`exp(0.5) ≈ 1.65`. Slightly tighter — small free precision gain.

**Result** (best-of-5 with min-of-batch timing):

| `d_v` | Before pre-scale | After pre-scale + `exp2` | Δ |
|---:|---:|---:|---:|
|  64 | 0.137 ms | 0.141 ms | within noise |
| 128 | 0.244 ms | 0.243 ms | flat |
| 256 | 0.582 ms | 0.563 ms | **−3%** |
| 512 | 1.334 ms | 1.311 ms | **−2%** |

Lands the predicted ~3-5% at the larger d configs where QKT inner
loop dominates; below the noise floor at small d. Free, since the
Q-load path was already being touched.

Accuracy: max abs error vs PyTorch fp16 SDPA widens slightly from
3e-4 to ~1e-3 because the pre-scaled Q is quantized to fp16 with the
extra factor baked in. Still well within the 1e-2 tolerance.

### Step 6 reference

Step 6 in this journal had reverted the same `exp2` substitution as
"a wash". That was on a kernel where softmax went through the
`s_scores` smem round-trip — softmax wasn't the hot path. After Step
11 moved softmax fully into registers, the exp calls *are* the hot
path, and the saved fmul counts.

### Apples-to-apples sweep (B=4 H=8) post-Step 15

| d_v | tensorax peak | PyTorch peak | best ratio |
|---:|---:|---:|---:|
| 64  | 15.4 TFLOPS @ S=4096 | 33.2 TFLOPS @ S=1024 | 0.48× |
| 128 | 15.9 TFLOPS @ S=8192 | 33.9 TFLOPS @ S=2048 | 0.50× |
| 256 | 10.8 TFLOPS @ S=8192 | 31.3 TFLOPS @ S=2048 | 0.36× |
| 512 |  6.2 TFLOPS @ S=4096 | 17.5 TFLOPS @ S=128  | 0.46× |

Best ratio versus cuDNN is now **0.50× at d=128 S=8192** (was 0.40×
in Step 14). The gap closes everywhere by 5-10pp absolute. cuDNN
still ahead by 2× at every supported config.

### `d_v=1024` / `d_k=1024` status

Skipped from the sweep (`benchmarks/attn_sweep.py`) because `s_q` at
`d_k=1024` is 128 KB, exceeding sm_86's 99 KB / CTA cap. Real
support needs either:

- **Q streaming**: re-load `s_q` slices per `dk_chunk` from L2;
  pipeline like K. Q traffic inflates but L2-cached.
- **Halve `Q_ROWS` to 32**: cuts `s_q` to 64 KB, fits, but only 2 warps
  active in QKT/PV (others idle) — likely bigger throughput loss
  than the streaming overhead.

Either is a nontrivial change; leaving as a tracked TODO.

## What I'd actually do next (post-Step 15)

1. **Cut `s_q` smem.** The 64 KB `s_q` is the dominant CTA-resident
   footprint. Two ways:
   - **Stream Q like K**: re-load Q's `d_k` slice per kv-tile, so
     smem holds only `64 × 16 × 2 = 2 KB` at a time. Doubles Q
     bandwidth (Q is loaded `seq_len_k / 16` times instead of once),
     but for `Sk=256` that's 16×, bandwidth-cheap if it buys 2-4
     CTAs / SM.
   - **Halve Q_ROWS to 32** (2 warps × 16 rows): cuts `s_q` to
     32 KB and halves PV register cost. Trades half the QKT
     parallelism — but if the PV phase is L1-bound (which it
     looks like, since oacc is in local memory), this might help.
2. **Pin `oacc` into actual registers.** `#pragma unroll` on the
   `dvc` loop forces ptxas to fix each slot to a register — but only
   feasible if total fp32 stays under ~200 regs / thread. Works for
   `d_v ≤ 128` (8 chunks → 64 fp32 / lane); won't work for
   `d_v = 512` (32 chunks → 256 fp32 / lane, would spill). Worth a
   `d_v`-templated kernel split: a "register-resident" path for
   `d_v ≤ 128` and the current "local-memory-resident" path for
   larger d_v.
3. **Larger `d_k` chunk (32 instead of 16).** Halves `dk_chunks`
   and the inner-loop `__syncthreads` count. Costs another
   `s_kchunk` slot in smem (~1 KB) — affordable.
4. **Persistent CTA scheduling.** Step 11 cut the grid 4×; persistent
   kernels would amortize launch overhead across tiles.
5. **fp16 PV accumulator.** Halves `oacc`'s local-memory footprint.
   Numerically risky across many kv tiles; do last.

