.PHONY: bench bench-quick profile build shell logs help

bench:        ## Run SDPA benchmark on H100
	modal run modal_app.py::benchmark

bench-quick:  ## Smaller shapes, fast iteration
	modal run modal_app.py::benchmark --quick

profile:      ## ncu profile of a kernel variant (default: mma_fp16; override with KERNEL=...)
	modal run modal_app.py::profile $(if $(KERNEL),--kernel $(KERNEL),)

build:        ## Build-only sanity check (CPU container, no H100 spin-up)
	modal run modal_app.py::build

shell:        ## Interactive shell on an H100 container
	modal shell modal_app.py::benchmark

logs:         ## Tail logs from the most recent run
	modal app logs tensora-hopper

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'
