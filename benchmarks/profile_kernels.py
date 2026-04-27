"""Drive Nsight Systems (nsys) profiling for every CUDA kernel in tensorax.

For each (op, method) pair, spawns nsys against profile_workload.py. nsys
captures a timeline of every CUDA kernel launch, memcpy, and CUDA API call
inside the cudaProfilerStart/Stop window in the workload, then exports a
per-kernel summary CSV.

Outputs to benchmarks/profiling/:
    <op>_<method>.nsys-rep   timeline report (open in Nsight Systems UI)
    <op>_<method>.csv        per-kernel time/count summary

Usage (activate .venv first):
    python benchmarks/profile_kernels.py
    python benchmarks/profile_kernels.py --only matmul.tiled sdpa.flash
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


WORKLOADS = [
    *(("matmul", m) for m in [
        "default",
        "shared_memory_coalesced",
        "tiled",
        "shared_memory_cache_blocking",
        "block_tiling_1d",
        "block_tiling_2d",
    ]),
    *(("sdpa", m) for m in [
        "naive",
        "tiled",
        "flash",
        "mma",
        "flash_optimized",
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="OP.METHOD",
        help="Profile only these workloads, e.g. matmul.tiled sdpa.flash",
    )
    ap.add_argument("--out", default="benchmarks/profiling")
    args = ap.parse_args()

    if shutil.which("nsys") is None:
        sys.exit("nsys not found in PATH. Install Nsight Systems or add /usr/local/cuda/bin to PATH.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # nsys writes scratch state to $TMPDIR/nvidia/nsight_systems. The default
    # /tmp/nvidia is often owned by root after a sudo'd run, so point at a
    # user-writable dir under the output folder.
    tmpdir = (out / ".nsys_tmp").resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)

    selected = WORKLOADS
    if args.only:
        wanted = set(args.only)
        selected = [(o, m) for (o, m) in WORKLOADS if f"{o}.{m}" in wanted]
        unknown = wanted - {f"{o}.{m}" for (o, m) in selected}
        if unknown:
            sys.exit(f"unknown workloads: {sorted(unknown)}")

    workload_script = Path(__file__).parent / "profile_workload.py"

    for op, method in selected:
        tag = f"{op}_{method}"
        report = out / tag
        csv_path = out / f"{tag}.csv"
        print(f"\n=== Profiling {tag} ===")

        nsys_cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx,osrt",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o", str(report),
            sys.executable, str(workload_script), op,
            "--method", method,
        ]
        rc = subprocess.call(nsys_cmd, env=env)
        if rc != 0:
            print(f"!! nsys failed for {tag} (rc={rc})", file=sys.stderr)
            continue

        # Per-kernel summary CSV (sorted by total GPU time)
        with open(csv_path, "w") as f:
            subprocess.call(
                [
                    "nsys", "stats",
                    "--report", "cuda_gpu_kern_sum",
                    "--format", "csv",
                    "--force-export", "true",
                    f"{report}.nsys-rep",
                ],
                stdout=f,
                env=env,
            )
        print(f"  report: {report}.nsys-rep")
        print(f"  csv:    {csv_path}")


if __name__ == "__main__":
    main()
