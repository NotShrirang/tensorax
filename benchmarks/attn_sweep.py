"""Sweep tensorax SDPA fp16 vs PyTorch fp16 SDPA across (B, S, d) configs.

Runs serially. Writes a CSV with one row per config. Usage:
    python benchmarks/attn_sweep.py --out sweep.csv
"""
import argparse
import csv
import gc
import time

import torch

import tensorax as ts
import tensorax.functional as F


def time_call(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def iters_for(S, d):
    # Bumped: 3-iter runs at S>=4096 were noisy enough that single sweep
    # numbers misled on parity claims. Trade longer wall time for stable means.
    if S >= 8192 and d >= 256:
        return 10
    if S >= 8192:
        return 30
    if S >= 4096 and d >= 256:
        return 15
    if S >= 4096:
        return 50
    if S >= 2048:
        return 50
    if S >= 1024:
        return 50
    return 100


def _intlist(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="benchmarks/attn_sweep.csv")
    p.add_argument("--heads", type=int, default=8)
    # Subset / single-config selection (handy for ncu profiling).
    p.add_argument("--batches",  type=_intlist, default=[1, 4])
    p.add_argument("--seq-lens", type=_intlist,
                   default=[128, 256, 512, 1024, 2048, 4096, 8192])
    # d_k=1024 needs s_q smem of 128 KB which exceeds sm_86's 99 KB CTA cap.
    # A future Q-streaming or Q_ROWS=32 path would cover it.
    p.add_argument("--dims", type=_intlist, default=[64, 128, 256, 512])
    p.add_argument("--iters", type=int, default=0,
                   help="override iters; 0 = auto-scale by problem size")
    p.add_argument("--no-pytorch", action="store_true",
                   help="skip the PyTorch SDPA timing (useful under ncu)")
    args = p.parse_args()

    H = args.heads
    batches  = args.batches
    seq_lens = args.seq_lens
    dims     = args.dims

    fields = [
        "B", "H", "S", "d_k", "d_v", "iters",
        "tensorax_ms", "tensorax_tflops",
        "pytorch_ms",  "pytorch_tflops",
        "speedup_vs_pytorch",
        "tensorax_status",
    ]

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)

        for B in batches:
            for S in seq_lens:
                for d in dims:
                    iters = args.iters if args.iters > 0 else iters_for(S, d)
                    flops = 2.0 * B * H * S * S * (d + d)

                    try:
                        q_torch = torch.randn((B, H, S, d), device="cuda", dtype=torch.float32)
                        k_torch = torch.randn((B, H, S, d), device="cuda", dtype=torch.float32)
                        v_torch = torch.randn((B, H, S, d), device="cuda", dtype=torch.float32)
                        q_h_pt = q_torch.half()
                        k_h_pt = k_torch.half()
                        v_h_pt = v_torch.half()

                        # Tensorax fp16 path
                        ts_status = "ok"
                        ts_ms = float("nan")
                        ts_tf = float("nan")
                        try:
                            q_t = ts.Tensor(q_torch.cpu().numpy(), dtype="float32", device="cuda")
                            k_t = ts.Tensor(k_torch.cpu().numpy(), dtype="float32", device="cuda")
                            v_t = ts.Tensor(v_torch.cpu().numpy(), dtype="float32", device="cuda")
                            q_h = F.cast_to_fp16(q_t)
                            k_h = F.cast_to_fp16(k_t)
                            v_h = F.cast_to_fp16(v_t)

                            def run_ts():
                                F.scaled_dot_product_attention_mma_fp16(q_h, k_h, v_h)

                            t = time_call(run_ts, iters)
                            ts_ms = t * 1000
                            ts_tf = flops / (t * 1e12)
                        except Exception as e:
                            ts_status = f"error: {type(e).__name__}: {str(e)[:80]}"

                        # PyTorch fp16 SDPA (skip under --no-pytorch, e.g. ncu runs)
                        if args.no_pytorch:
                            pt_ms = float("nan")
                            pt_tf = float("nan")
                        else:
                            def run_pt():
                                torch.nn.functional.scaled_dot_product_attention(q_h_pt, k_h_pt, v_h_pt)
                            t = time_call(run_pt, iters)
                            pt_ms = t * 1000
                            pt_tf = flops / (t * 1e12)

                        speedup = pt_ms / ts_ms if (ts_status == "ok" and not args.no_pytorch) else float("nan")

                        row = [
                            B, H, S, d, d, iters,
                            f"{ts_ms:.4f}" if ts_status == "ok" else "",
                            f"{ts_tf:.3f}" if ts_status == "ok" else "",
                            f"{pt_ms:.4f}",
                            f"{pt_tf:.3f}",
                            f"{speedup:.3f}" if ts_status == "ok" else "",
                            ts_status,
                        ]
                        w.writerow(row)
                        fh.flush()
                        if ts_status == "ok":
                            print(f"B={B} H={H} S={S:5d} d={d:4d}  "
                                  f"ts={ts_ms:8.3f} ms ({ts_tf:5.2f} TF)   "
                                  f"pt={pt_ms:8.3f} ms ({pt_tf:5.2f} TF)   "
                                  f"speedup={speedup:5.2f}x")
                        else:
                            print(f"B={B} H={H} S={S:5d} d={d:4d}  "
                                  f"ts={ts_status:50s} "
                                  f"pt={pt_ms:8.3f} ms ({pt_tf:5.2f} TF)")
                    except Exception as e:
                        print(f"B={B} H={H} S={S} d={d}  HARD ERROR: {type(e).__name__}: {e}")
                        w.writerow([B, H, S, d, d, iters, "", "", "", "", "", f"hard error: {e}"])
                        fh.flush()

                    # Cleanup
                    for name in ("q_torch", "k_torch", "v_torch",
                                 "q_h_pt", "k_h_pt", "v_h_pt",
                                 "q_t", "k_t", "v_t",
                                 "q_h", "k_h", "v_h"):
                        try:
                            del locals()[name]
                        except KeyError:
                            pass
                    torch.cuda.empty_cache()
                    gc.collect()


if __name__ == "__main__":
    main()
