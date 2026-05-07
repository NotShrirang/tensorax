.PHONY: bench bench-quick bench-cute bench-mm-cute profile profile-cute build shell logs help

bench:        ## Run SDPA benchmark on H100 (default shape: B=4 H=8 S=256 D=512)
	modal run modal_app.py::benchmark

bench-quick:  ## Smaller shapes, fast iteration
	modal run modal_app.py::benchmark --quick

bench-cute:   ## v0 cute_fp16 shape (B=4 H=8 S=2048 D=128) vs pytorch fp16
	modal run modal_app.py::bench_cute

bench-mm-cute: ## v0 matmul cute_fp16 shape (B=1, M=N=K=4096) vs pytorch fp16 (cuBLAS)
	modal run modal_app.py::bench_mm_cute

profile:      ## ncu profile of a kernel variant at default shape (default: mma_fp16; override with KERNEL=...)
	modal run modal_app.py::profile $(if $(KERNEL),--kernel $(KERNEL),)

profile-cute: ## ncu profile of a cute_* variant at S=2048,D=128 (default: cute_fp16_pp; override with KERNEL=...)
	modal run modal_app.py::profile_cute $(if $(KERNEL),--kernel $(KERNEL),)

build:        ## Build-only sanity check (CPU container, no H100 spin-up)
	modal run modal_app.py::build

shell:        ## Interactive shell on an H100 container
	modal shell modal_app.py::benchmark

logs:         ## Tail logs from the most recent run
	modal app logs tensora-hopper

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'
