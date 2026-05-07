# cute_sdpa — Hopper SDPA via CUTLASS FMHA collective

## Workload

| | |
|---|---|
| Op | `O = softmax(Q @ K^T / sqrt(d)) @ V` |
| Shape | `B=4, H=8, S=2048, Dk=Dv=128`, fp16 in / fp16 out |
| Hardware | NVIDIA H100 (sm_90a) via Modal |
| Bench | `make bench-cute` (`benchmarks/attn_benchmark.py`, 30 iters, allclose vs torch SDPA fp16 with rtol=atol=5e-3) |

**FLOPs / call** (matches `attn_benchmark.py`):
`2·B·H·S²·(Dk + Dv) = 2·4·8·2048²·256 ≈ 68.7 GFLOPs`

**Bytes / call** (Q, K, V, O each `B·H·S·D·2 B`):
`4·B·H·S·D·2 = 4·4·8·2048·128·2 ≈ 67 MB`

**Arithmetic intensity:** `68.7e9 / 67e6 ≈ 1024 FLOP/B` — solidly compute-bound.

## First-principles bound

| | |
|---|---|
| H100 SXM fp16 TensorCore peak | ~989 TFLOPS |
| Compute lower bound | `68.7e9 / 989e12 ≈ 69 µs/call` |
| H100 HBM3 BW | ~3.35 TB/s |
| Memory lower bound | `67e6 / 3.35e12 ≈ 20 µs/call` |
| Roofline ridge AI | `989e12 / 3.35e12 ≈ 295 FLOP/B` |

Workload AI (1024) ≫ ridge AI (295). **Compute-bound.** Ceiling determined by tensor-pipe utilization, not memory.

## Bar to beat

| variant | TFLOPS | % of peak |
|---|---:|---:|
| `pytorch_fp16` (cuDNN flash dispatch) | **~470** | 48% |
| `pytorch_compile_fp16` | ~385 | 39% |

Both baselines run the same FlashAttention-3 algorithm (TMA producer + WGMMA consumer warps + online softmax). torch.compile loses to native cuDNN dispatch — confirmed via runs and saved in `memory/project_sdpa_bar.md`. The real bar is **cuDNN's FA-3 at ~470 TFLOPS = 0.48× of peak.** Our gap to that is tuning, not algorithm.

## v0 — Cooperative dispatch (188 TFLOPS = 19% of peak)

**Instantiation** (`csrc/cuda/kernels/attn_hopper.cu`):

```cpp
using Kernel = cutlass::fmha::kernel::FmhaBuilder<
    half_t, /*AccQK*/ float, /*AccPV*/ float,
    Shape<_128, _128, _128>,                      // BlockQO × BlockKV × BlockHead
    StrideQ, StrideK, StrideV,                    // (int, _1, (int, int)) — row-major B,H,S,D
    cutlass::fmha::collective::DefaultFusion,     // no causal/residual mask
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::Kernel;
```

- TMA producer warpgroup loads Q, K, V into smem
- 2 consumer warpgroups run WGMMA + online softmax + LSE
- 5 KV pipeline stages (default), 2 Q stages
- Cluster shape hardcoded to `<1,1,1>` inside the collective

**Measurement:** 188 TFLOPS, allclose PASSES.

**Diagnosis at this stage:** algorithm correct, ~2.5× behind cuDNN. The collective wraps TMA + WGMMA + warp-spec correctly. Next: A/B against the other dispatch policy that example 88 ships for D=128.

**Next hypothesis:** `KernelTmaWarpSpecializedPingpong` adds two `Option`s — `kIsPersistent=true` (persistent tile scheduler reuses 132 CTAs across waves) and `kLoadsQSeparately=true` (each consumer warpgroup loads its own Q, halving Q tile size per warpgroup but doubling Q-load throughput). Should help moderately.

## v1 — Pingpong (200 TFLOPS = 20% of peak, +6% over Cooperative)

**Change:** flip dispatch policy template arg to `KernelTmaWarpSpecializedPingpong`. ~5 lines of code (added a second entry point).

**Measurement:** 200 TFLOPS, +6% over v0. Reproducible across multiple runs (allowing for ~3% Modal H100 binning noise on the absolute number).

**Diagnosis:** small win, in the expected direction. Still 2.3× behind cuDNN. Time to profile instead of guessing further.

## v2 — ncu profile (no code change, diagnostic only)

`make profile-cute` runs cute_fp16_pp and cuDNN flash side-by-side at the same shape, single-launch each, ncu --set full.

| metric | cute_fp16_pp | cuDNN flash | δ |
|---|---:|---:|---|
| Duration | 172.86 µs | 138.98 µs | +24% |
| **Tensor pipe utilization** | **54.9%** | **71.0%** | **−16 pp** |
| Compute SM throughput | 52.0% | 67.0% | −15 pp |
| Warp cycles per instruction | 8.28 | 5.34 | +55% |

**Stall reasons (cycles per issued instruction):**

| reason | cute_fp16_pp | cuDNN | δ |
|---|---:|---:|---|
| barrier | **3.19** | 1.50 | +113% |
| wait | 1.40 | 0.87 | +61% |
| gmma | **0.22** | 0.07 | +200% |
| long_scoreboard | 0.75 | 0.69 | +9% |

**Launch shape:** identical (Block(384,1,1), Grid(132,1,1), 168 reg/thread, 1 wave/SM, 215KB dyn smem).

**One key difference:** cuDNN uses **CGA `4×1×1`** (cluster grid array — 4-CTA cluster with multicast TMA). Ours uses `1×1×1`. cuDNN's kernel name encodes it: `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1x1_kernel0_0`.

**Diagnosis:**
- Bottleneck is mainloop-internal: shallower producer/consumer pipeline + no cluster-multicast TMA.
- `barrier` stalls (warp groups waiting on async barrier completion) translate to "TMA→WGMMA pipeline isn't keeping consumers fed."
- `gmma` stalls (fewer in-flight WGMMA instructions) confirm the same.
- L1/TEX hit rate: 0% (cuDNN, full TMA bypass) vs 71% (ours, some loads going through L1) — cuDNN has tighter TMA-only path.

**Next hypothesis:** the FMHA `Tag::` enum exposes `kStagesQ`, `kStagesKV`, `kClusterM`, `kBlocksPerSM`. If `kClusterM` is wired, setting it to 4 closes the headline gap. If `kStagesQ` ≥ 3 alleviates the barrier pressure, that closes the secondary one.

## v3 — read the source: kClusterM is dead, kStagesQ is alive

Reading `cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma_warpspecialized.hpp:80`:

```cpp
using ClusterShape = Shape<_1, _1, _1>;   // hardcoded; kClusterM is declared but never read
```

`grep -rn "kClusterM\|kBlocksPerSM" /tmp/cutlass_ref/` returns matches only in `fmha_options.hpp` (the enum declaration) — never used elsewhere. **Both are dead enum values.** Same dead-end for `kBlocksPerSM`.

`kStagesQ` and `kStagesKV` ARE wired:

```cpp
static constexpr int StageCount  = find_option_t<Tag::kStagesKV, Int<5>, Options...>::value;
static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<NumMmaWarpGroups>, Options...>::value;
```

So we can deepen the Q pipeline. KV stages already at 5; bumping further likely smem-overflows (we're at 215 KB / 228 KB).

**Diagnosis:** the headline cuDNN gap (cluster shape) is **unreachable** without forking the collective. We can probe the secondary gap (`kStagesQ`) cheaply.

**Next hypothesis:** `kStagesQ=3` (vs default 2) adds one more Q tile in flight. Each Q tile is `BlockQO × BlockHead × 2B = 128×128×2 = 32 KB`, but with `kLoadsQSeparately` each consumer warpgroup loads only half, so +16 KB total. Should fit. Expected: alleviates `barrier` stall, +5–10%.

## v4 — kStagesQ=3 (≈200 TFLOPS, no win)

**Change:** pass `Option<Tag::kStagesQ, Int<3>>` to the FmhaBuilder via the existing variadic Options pack.

**Measurement (4 runs):** `cute_fp16_pp_q3` ≈ 199 TFLOPS, `cute_fp16_pp` ≈ 200 TFLOPS. **Within Modal H100 binning noise.**

**Diagnosis:** Q-load isn't the gate. Softmax/GEMM overlap and producer/consumer queue depth must be more nuanced than "bump Q stages." The collective is doing something specific in the Q load schedule that another stage doesn't help.

**Side experiment that compile-failed:** `kNumMmaWarpGroups=3` produces `BlockQO/3 = 128/3 = 42` non-integer per-warpgroup tile, which breaks the epilogue's TMA-store smem layout. ~100 nvcc errors all rooted in `cute::C<42>`. Saved as `memory/feedback_fmha_warpgroup_divisibility.md`.

## Where we are

**Best variant:** `cute_fp16_pp` ≈ 200 TFLOPS = 20% of fp16 peak = **0.43× of cuDNN bar.**

**Knob ceiling reached** for what `FmhaBuilder` exposes. Pingpong wins by ~6%, kStagesQ doesn't move, kNumMmaWarpGroups=3 is incompatible with M=128, kClusterM/kBlocksPerSM are dead enums, kStagesKV likely smem-bound at 5.

**The bar's headline advantage (CGA 4×1×1 cluster + multicast TMA) is unreachable from outside the collective.** Closing it requires forking `fmha_collective_tma_warpspecialized.hpp`, hardcoding `ClusterShape = Shape<_4,_1,_1>`, and setting up cluster-aware TMA descriptors — ~2-3 days of work with non-trivial correctness risk on the TMA descriptor side.

## Build / bench commands

```
make build            # CPU container, ~1.5 min
make bench-cute       # H100, runs cute_fp16, cute_fp16_pp, cute_fp16_pp_q3 + pytorch baselines at S=2048,D=128
make profile-cute     # H100, ncu --set full of cute_fp16_pp + cuDNN, report saved to /build_vol
```
