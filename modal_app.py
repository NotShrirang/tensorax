"""Modal app for the tensora Hopper SDPA work.

Three entrypoints, invoked from the Makefile:
  - build:     compile the C extension on a CPU container, stash _C.so on a Volume
  - benchmark: run benchmarks/attn_benchmark.py on an H100
  - profile:   run ncu over a single kernel variant on an H100

The build step runs CPU-only because nvcc does not need a GPU. H100 is
reserved for actual GPU work. Source is mounted (not baked into the image)
so iterating on kernel code does not invalidate any image layer. The CUTLASS
clone, uv cache, and built .so live on Volumes so they survive across runs.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

CUTLASS_TAG = "v4.4.2"
CUTLASS_SPARSE_REV = "ex88-v1"
CUDA_ARCH = "9.0"

app = modal.App("tensora-hopper")

cutlass_vol = modal.Volume.from_name("tensora-cutlass", create_if_missing=True)
build_vol = modal.Volume.from_name("tensora-build", create_if_missing=True)

VOLUMES = {
    "/cutlass_vol": cutlass_vol,
    "/build_vol": build_vol,
}

_CUTLASS_MARKER = f"{CUTLASS_TAG}+{CUTLASS_SPARSE_REV}"
_CUTLASS_SETUP = (
    "set -e; "
    "mkdir -p /cutlass_vol; "
    f"if [ -f /cutlass_vol/.tag ] && [ \"$(cat /cutlass_vol/.tag)\" = \"{_CUTLASS_MARKER}\" ]; then "
    f"  echo 'CUTLASS already at {_CUTLASS_MARKER}'; "
    "else "
    f"  echo 'Re-cloning CUTLASS at {_CUTLASS_MARKER}'; "
    "  rm -rf /cutlass_vol/.git /cutlass_vol/.tag /cutlass_vol/include /cutlass_vol/tools /cutlass_vol/examples; "
    "  cd /cutlass_vol; "
    "  git init -q .; "
    "  git remote add origin https://github.com/NVIDIA/cutlass.git; "
    "  git config core.sparseCheckout true; "
    "  printf 'include/\\ntools/util/include/\\nexamples/88_hopper_fmha/\\n' > .git/info/sparse-checkout; "
    f"  git fetch --depth 1 origin refs/tags/{CUTLASS_TAG}:refs/tags/{CUTLASS_TAG}; "
    f"  git checkout {CUTLASS_TAG}; "
    f"  echo {_CUTLASS_MARKER} > .tag; "
    "fi"
)

_SKIP_PARTS = {"build", "htmlcov", ".venv", "__pycache__", ".git"}


def _ignore_local(path) -> bool:
    parts = Path(path).parts
    if any(part in _SKIP_PARTS for part in parts):
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if any(part.endswith(".so") for part in parts):
        return True
    name = parts[-1] if parts else ""
    if name.startswith("ncu_logs."):
        return True
    return False


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git", "ca-certificates", "build-essential",
        "gcc-12", "g++-12", "clang", "cmake", "ninja-build", "curl",
    )
    .run_commands(
        "ln -sf $(find /opt/nvidia/nsight-compute -name ncu -type f 2>/dev/null | head -1) /usr/local/bin/ncu || true",
    )
    .pip_install("uv", "pybind11>=2.6.0", "numpy", "wheel", "setuptools>=61")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:${PATH}",
        "TENSORAX_ARCH": "hopper",
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
        "CUTLASS_HOME": "/cutlass_vol",
        "CC": "gcc-12",
        "CXX": "g++-12",
    })
    .add_local_dir(".", remote_path="/workspace", ignore=_ignore_local)
)


def _ensure_cutlass():
    subprocess.run(_CUTLASS_SETUP, shell=True, check=True)


def _build_extension():
    _ensure_cutlass()
    subprocess.run(
        ["uv", "pip", "install", "--system", "--no-build-isolation", "-e", "."],
        cwd="/workspace", check=True,
    )
    subprocess.run(
        "strip --strip-unneeded /workspace/tensorax/_C*.so || true",
        shell=True,
    )
    subprocess.run("cp -v /workspace/tensorax/_C*.so /build_vol/", shell=True, check=True)
    build_vol.commit()


def _install_prebuilt():
    subprocess.run("cp -v /build_vol/_C*.so /workspace/tensorax/", shell=True, check=True)
    subprocess.run(
        ["uv", "pip", "install", "--system", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd="/workspace", check=True,
    )


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=900,
)
def build():
    """Compile the Hopper extension on a CPU container."""
    _build_extension()
    print("[build] OK — _C.so written to tensora-build volume")


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=600,
    scaledown_window=30,
)
def benchmark(quick: bool = False):
    """Run benchmarks/attn_benchmark.py on H100."""
    _install_prebuilt()
    cmd = ["python", "benchmarks/attn_benchmark.py"]
    if quick:
        cmd += [
            "--seq_len", "128", "--d_k", "64", "--d_v", "64",
            "--batch", "1", "--heads", "4", "--times", "5",
        ]
    subprocess.run(cmd, cwd="/workspace", check=True)


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=600,
    scaledown_window=30,
)
def bench_cute(times: int = 30):
    """Run the cute_fp16 v0 shape (S=2048, D=128) against the fp16 SDPA baselines."""
    _install_prebuilt()
    cmd = [
        "python", "benchmarks/attn_benchmark.py",
        "--batch", "4", "--heads", "8",
        "--seq_len", "2048", "--d_k", "128", "--d_v", "128",
        "--times", str(times),
        "--only", "cute_fp16", "cute_fp16_pp", "cute_fp16_pp_q3",
                  "pytorch_fp16", "pytorch_compile_fp16",
    ]
    subprocess.run(cmd, cwd="/workspace", check=True)


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=600,
    scaledown_window=30,
)
def bench_mm_cute(times: int = 30):
    """Run the matmul cute_fp16 v0 shape (B=1, M=N=K=4096) against cuBLAS fp16 baseline."""
    _install_prebuilt()
    cmd = [
        "python", "benchmarks/matmul_benchmark.py",
        "--batch", "1",
        "--M", "4096", "--K", "4096", "--N", "4096",
        "--times", str(times),
        "--only", "cute_fp16", "cute_fp16_c4", "cute_fp16_pp", "cute_fp16_t256",
                  "pytorch_fp16", "pytorch_compile_fp16",
    ]
    subprocess.run(cmd, cwd="/workspace", check=True)


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=1800,
    scaledown_window=30,
)
def profile(kernel: str = "mma_fp16"):
    """Run ncu over a single kernel variant at the default bench shape (B=4 H=8 S=256 D=512)."""
    _install_prebuilt()
    out = "/build_vol/ncu_report.ncu-rep"
    cmd = (
        f"ncu --set full --target-processes all -f -o {shlex.quote(out)} "
        f"python benchmarks/attn_benchmark.py --only {shlex.quote(kernel)} "
        f"--times 3 --quiet"
    )
    subprocess.run(cmd, shell=True, cwd="/workspace", check=True)
    build_vol.commit()
    print("[profile] report saved to tensora-build volume — "
          "download with `modal volume get tensora-build ncu_report.ncu-rep`")


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=1800,
    scaledown_window=30,
)
def profile_cute(kernel: str = "cute_fp16_pp"):
    """Run ncu over a cute_* variant at the cute bench shape (B=4 H=8 S=2048 D=128)."""
    _install_prebuilt()
    out = f"/build_vol/ncu_report_{kernel}.ncu-rep"
    cmd = (
        f"ncu --set full --target-processes all -f -o {shlex.quote(out)} "
        f"python benchmarks/attn_benchmark.py "
        f"--batch 4 --heads 8 --seq_len 2048 --d_k 128 --d_v 128 "
        f"--only {shlex.quote(kernel)} --times 3 --quiet"
    )
    subprocess.run(cmd, shell=True, cwd="/workspace", check=True)
    build_vol.commit()
    print(f"[profile_cute] report saved to tensora-build volume as ncu_report_{kernel}.ncu-rep — "
          f"download with `modal volume get tensora-build ncu_report_{kernel}.ncu-rep`")


@app.local_entrypoint()
def main(action: str = "bench", quick: bool = False, kernel: str = "mma_fp16", times: int = 30):
    if action == "build":
        build.remote()
    elif action == "bench":
        benchmark.remote(quick=quick)
    elif action == "bench-cute":
        bench_cute.remote(times=times)
    elif action == "bench-mm-cute":
        bench_mm_cute.remote(times=times)
    elif action == "profile":
        profile.remote(kernel=kernel)
    elif action == "profile-cute":
        profile_cute.remote(kernel=kernel if kernel != "mma_fp16" else "cute_fp16_pp")
    else:
        sys.exit(f"unknown action: {action!r}")
