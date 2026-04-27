"""Single-op workload runner driven by profile_kernels.py under nsys.

Warm-up runs happen before cudaProfilerStart, so nsys (with
--capture-range=cudaProfilerApi --capture-range-end=stop) only captures the
one measured launch.
"""
import argparse
import torch
import tensorax as ts
import tensorax.functional as F


MATMUL_METHODS = [
    "default",
    "shared_memory_coalesced",
    "tiled",
    "shared_memory_cache_blocking",
    "block_tiling_1d",
    "block_tiling_2d",
]

SDPA_FNS = {
    "naive": F.scaled_dot_product_attention,
    "tiled": F.scaled_dot_product_attention_tiled,
    "flash": F.scaled_dot_product_attention_flash,
    "mma": F.scaled_dot_product_attention_mma,
    "flash_optimized": F.scaled_dot_product_attention_flash_optimized,
}


def run_matmul(method, warmup, B, M, K, N):
    a = torch.randn((B, M, K), device="cuda", dtype=torch.float32).cpu().numpy()
    b = torch.randn((B, K, N), device="cuda", dtype=torch.float32).cpu().numpy()
    a_t = ts.Tensor(a, dtype="float32", device="cuda")
    b_t = ts.Tensor(b, dtype="float32", device="cuda")

    for _ in range(warmup):
        a_t.matmul(b_t, method=method)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    a_t.matmul(b_t, method=method)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def run_sdpa(method, warmup, B, H, S, Dk, Dv):
    q = torch.randn((B, H, S, Dk), device="cuda", dtype=torch.float32).cpu().numpy()
    k = torch.randn((B, H, S, Dk), device="cuda", dtype=torch.float32).cpu().numpy()
    v = torch.randn((B, H, S, Dv), device="cuda", dtype=torch.float32).cpu().numpy()
    q_t = ts.Tensor(q, dtype="float32", device="cuda")
    k_t = ts.Tensor(k, dtype="float32", device="cuda")
    v_t = ts.Tensor(v, dtype="float32", device="cuda")

    fn = SDPA_FNS[method]
    for _ in range(warmup):
        fn(q_t, k_t, v_t)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    fn(q_t, k_t, v_t)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="op", required=True)

    m = sub.add_parser("matmul")
    m.add_argument("--method", required=True, choices=MATMUL_METHODS)
    m.add_argument("--warmup", type=int, default=3)
    m.add_argument("-B", type=int, default=3)
    m.add_argument("-M", type=int, default=1024)
    m.add_argument("-K", type=int, default=1024)
    m.add_argument("-N", type=int, default=1024)

    s = sub.add_parser("sdpa")
    s.add_argument("--method", required=True, choices=list(SDPA_FNS.keys()))
    s.add_argument("--warmup", type=int, default=3)
    s.add_argument("-B", type=int, default=4)
    s.add_argument("-H", type=int, default=8)
    s.add_argument("-S", type=int, default=256)
    s.add_argument("--dk", type=int, default=512)
    s.add_argument("--dv", type=int, default=512)

    args = p.parse_args()
    if args.op == "matmul":
        run_matmul(args.method, args.warmup, args.B, args.M, args.K, args.N)
    else:
        run_sdpa(args.method, args.warmup, args.B, args.H, args.S, args.dk, args.dv)


if __name__ == "__main__":
    main()
