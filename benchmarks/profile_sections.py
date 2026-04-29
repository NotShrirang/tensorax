"""Per-section histograms for instrumented kernels.

Build with:
    TENSORAX_PROFILE=1 bash build.sh

Each section corresponds to one logical operation between two TX_TICK calls.
Names follow the kernel source: a section is "what happens between tick N and
tick N+1", named after the single op that runs there.
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import tensorax as ts
import tensorax._C as _C


SECTIONS = {
    "matmul.naive": [
        "Setup",
        "Inner K-loop, first quarter",
        "Inner K-loop, second quarter",
        "Inner K-loop, third quarter",
        "Inner K-loop, fourth quarter",
        "Write result to memory",
    ],
    "matmul.tiled": [
        "Setup",
        "Read A tile into shared memory",
        "Read B tile into shared memory",
        "Multiply the two tiles",
        "Repeat for remaining tiles",
        "Write result to memory",
    ],
    "matmul.shared_memory_coalesced": [
        "Setup",
        "Inner K-loop, first quarter",
        "Inner K-loop, second quarter",
        "Inner K-loop, third quarter",
        "Inner K-loop, fourth quarter",
        "Write result to memory",
    ],
    "matmul.shared_memory_cache_blocking": [
        "Setup",
        "Read A block into shared memory",
        "Read B block into shared memory",
        "Multiply the two blocks",
        "Repeat for remaining blocks",
        "Write result to memory",
    ],
    "matmul.block_tiling_1d": [
        "Setup",
        "Read A block into shared memory",
        "Read B block into shared memory",
        "Multiply (1D register tiling)",
        "Repeat for remaining blocks",
        "Write result to memory",
    ],
    "matmul.block_tiling_2d": [
        "Setup",
        "Read A block into shared memory",
        "Read B block into shared memory",
        "Multiply (2D register tiling)",
        "Repeat for remaining blocks",
        "Write result to memory",
    ],
    "sdpa.naive": [
        "Setup",
        "Softmax part 1: find row max over keys",
        "Softmax part 2: compute sum of exponentials",
        "Softmax part 3: normalize and multiply by V",
        "Write output to memory",
    ],
    "sdpa.tiled": [
        "Setup",
        "Read K and V tile into shared memory",
        "First key: Q dot K score",
        "First key: softmax update (max, exp)",
        "First key: accumulate weighted V",
        "Continue remaining keys in tile",
        "Repeat for remaining tiles",
        "Write output to memory",
    ],
    "sdpa.flash": [
        "Setup and load Q",
        "Read K tile into shared memory",
        "Read V tile into shared memory",
        "First query: score keys (Q dot K)",
        "First query: softmax max + rescale",
        "First query: accumulate weighted V",
        "Continue remaining queries",
        "Repeat for remaining KV tiles",
        "Write output to memory",
    ],
    "sdpa.mma": [
        "Setup",
        "Load Q + initialize accumulators",
        "First KV tile: load and cast K, V",
        "First KV tile: Q * K-transposed (tensor core)",
        "First KV tile: online softmax",
        "First KV tile: cast P + rescale O",
        "First KV tile: P * V (tensor core)",
        "Remaining KV tiles",
        "Final normalize and write output",
    ],
    "sdpa.flash_optimized": [
        "Setup and load Q (vectorized)",
        "Read K tile into shared memory",
        "Read V tile into shared memory",
        "First query: score keys (Q dot K)",
        "First query: softmax max + rescale",
        "First query: accumulate weighted V",
        "Continue remaining queries",
        "Repeat for remaining KV tiles",
        "Write output to memory",
    ],
}


def gpu_clock_hz():
    props = torch.cuda.get_device_properties(0)
    for attr in ("clock_rate", "clockRate"):
        if hasattr(props, attr):
            return getattr(props, attr) * 1000.0
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.max.sm",
         "--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()[0]
    return float(out) * 1e6


def cycles_to_ns(cycles, hz):
    return cycles * 1e9 / hz


_MATMUL_BINDINGS = {
    "matmul.naive": ("default", "profile_sections_matmul_naive"),
    "matmul.tiled": ("tiled", "profile_sections_matmul_tiled"),
    "matmul.shared_memory_coalesced": ("shared_memory_coalesced", "profile_sections_matmul_shared_memory_coalesced"),
    "matmul.shared_memory_cache_blocking": ("shared_memory_cache_blocking", "profile_sections_matmul_shared_memory_cache_blocking"),
    "matmul.block_tiling_1d": ("block_tiling_1d", "profile_sections_matmul_1d_blocktiling"),
    "matmul.block_tiling_2d": ("block_tiling_2d", "profile_sections_matmul_2d_blocktiling"),
}

_SDPA_BINDINGS = {
    "sdpa.naive": ("scaled_dot_product_attention", "profile_sections_sdpa_naive"),
    "sdpa.tiled": ("scaled_dot_product_attention_tiled", "profile_sections_sdpa_tiled"),
    "sdpa.flash": ("scaled_dot_product_attention_flash", "profile_sections_sdpa_flash"),
    "sdpa.mma": ("scaled_dot_product_attention_mma", "profile_sections_sdpa_mma"),
    "sdpa.flash_optimized": ("scaled_dot_product_attention_flash_optimized", "profile_sections_sdpa_flash_optimized"),
}


def make_runner(name):
    """Returns a closure that profiles one launch. Tensor setup + warmup
    happen once; the closure can be called repeatedly to measure."""
    if name in _MATMUL_BINDINGS:
        method, binding = _MATMUL_BINDINGS[name]
        a = ts.Tensor(np.random.randn(1024, 1024).astype(np.float32), device="cuda")
        b = ts.Tensor(np.random.randn(1024, 1024).astype(np.float32), device="cuda")
        for _ in range(3):
            a.matmul(b, method=method)
        torch.cuda.synchronize()
        fn = getattr(_C, binding)
        return lambda: fn(a._c_tensor, b._c_tensor)
    if name in _SDPA_BINDINGS:
        import tensorax.functional as F
        method, binding = _SDPA_BINDINGS[name]
        q = ts.Tensor(np.random.randn(4, 8, 256, 512).astype(np.float32), device="cuda")
        k = ts.Tensor(np.random.randn(4, 8, 256, 512).astype(np.float32), device="cuda")
        v = ts.Tensor(np.random.randn(4, 8, 256, 512).astype(np.float32), device="cuda")
        fn_warm = getattr(F, method)
        for _ in range(3):
            fn_warm(q, k, v)
        torch.cuda.synchronize()
        fn = getattr(_C, binding)
        return lambda: fn(q._c_tensor, k._c_tensor, v._c_tensor)
    raise ValueError(name)


def fmt_ns(ns):
    if ns >= 1e6: return f"{ns/1e6:.2f}ms"
    if ns >= 1e3: return f"{ns/1e3:.2f}us"
    return f"{ns:.0f}ns"


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def print_table(name, labels, stats, total_avg, n):
    rows = sorted(zip(labels, stats), key=lambda r: -r[1]["avg"])
    name_w = max(4, max(len(r[0]) for r in rows))
    headers = ["Name", "Self %", "Avg", "Min", "Max", "Std", "Runs"]
    widths = [name_w, 7, 10, 10, 10, 10, 5]
    sep = "  ".join("-" * w for w in widths)
    print(f"\n=== {name}   avg total {fmt_ns(total_avg)} over {n} runs ===")
    print(sep)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for label, s in rows:
        pct = 100.0 * s["avg"] / total_avg if total_avg else 0
        cells = [
            label.ljust(name_w),
            f"{pct:.2f}%".ljust(7),
            fmt_ns(s["avg"]).ljust(10),
            fmt_ns(s["min"]).ljust(10),
            fmt_ns(s["max"]).ljust(10),
            fmt_ns(s["std"]).ljust(10),
            str(n).ljust(5),
        ]
        print("  ".join(cells))
    print(sep)


def write_outputs(name, labels, stats, total_avg, n, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avgs = [s["avg"] for s in stats]
    stds = [s["std"] for s in stats]

    width = max(10, 0.9 * len(labels) + 4)
    fig, ax = plt.subplots(figsize=(width, 6))
    bars = ax.bar(range(len(labels)), avgs, yerr=stds, color="#4c72b0",
                  capsize=4, error_kw={"ecolor": "#333"})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("ns (block 0,0,0, mean across runs)")
    ax.set_title(f"{name}  -  avg total {fmt_ns(total_avg)} over {n} runs")
    for bar, d in zip(bars, avgs):
        pct = 100.0 * d / total_avg if total_avg else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    png = out_dir / f"{name.replace('.', '_')}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    csv_path = out_dir / f"{name.replace('.', '_')}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "avg_ns", "min_ns", "max_ns", "std_ns", "pct"])
        for label, s in zip(labels, stats):
            pct = 100.0 * s["avg"] / total_avg if total_avg else 0
            w.writerow([label, f"{s['avg']:.1f}", f"{s['min']:.1f}",
                        f"{s['max']:.1f}", f"{s['std']:.1f}", f"{pct:.2f}"])
    return png, csv_path


def append_log(name, labels, stats, n, out_dir, sha):
    log_path = out_dir / f"{name.replace('.', '_')}_log.csv"
    new_file = not log_path.exists()
    ts_now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "git_sha", "runs", "section",
                        "avg_ns", "min_ns", "max_ns", "std_ns"])
        for label, s in zip(labels, stats):
            w.writerow([ts_now, sha, n, label,
                        f"{s['avg']:.1f}", f"{s['min']:.1f}",
                        f"{s['max']:.1f}", f"{s['std']:.1f}"])
    return log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="profile only these kernels (e.g. sdpa.mma)")
    ap.add_argument("--out", default="benchmarks/profiling/sections")
    ap.add_argument("--iterations", "-n", type=int, default=30,
                    help="number of measured runs per kernel (default: 30)")
    ap.add_argument("--no-log", action="store_true",
                    help="skip appending to per-kernel log CSV")
    args = ap.parse_args()

    if not hasattr(_C, "profile_sections_sdpa_mma"):
        sys.exit("Build was not compiled with TENSORAX_PROFILE=1.\n"
                 "Rebuild: TENSORAX_PROFILE=1 bash build.sh")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hz = gpu_clock_hz()
    sha = git_rev()
    print(f"GPU clock: {hz/1e9:.2f} GHz   git: {sha}   runs/kernel: {args.iterations}")

    targets = list(SECTIONS) if not args.only else args.only
    for name in targets:
        if name not in SECTIONS:
            print(f"!! unknown kernel '{name}'", file=sys.stderr); continue
        labels = SECTIONS[name]
        n = args.iterations
        runner = make_runner(name)

        all_deltas = np.zeros((n, len(labels)), dtype=np.float64)
        for i in range(n):
            cycles = runner()
            ts_ns = [cycles_to_ns(c, hz) for c in cycles]
            for j in range(len(labels)):
                all_deltas[i, j] = ts_ns[j + 1] - ts_ns[j]

        stats = []
        for j in range(len(labels)):
            col = all_deltas[:, j]
            stats.append({
                "avg": float(col.mean()),
                "min": float(col.min()),
                "max": float(col.max()),
                "std": float(col.std()),
            })
        total_avg = sum(s["avg"] for s in stats)

        print_table(name, labels, stats, total_avg, n)
        png, csv_path = write_outputs(name, labels, stats, total_avg, n, out)
        print(f"  png: {png}")
        print(f"  csv: {csv_path}")
        if not args.no_log:
            log_path = append_log(name, labels, stats, n, out, sha)
            print(f"  log: {log_path}")


if __name__ == "__main__":
    main()
