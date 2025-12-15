#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t status = call;                                         \
        if (status != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(status));                 \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

namespace tensorax {
    namespace cuda {
        // Common CUDA configurations
        constexpr int BLOCK_SIZE = 256;
        constexpr int TILE_SIZE = 16;
        constexpr int WARP_SIZE = 32;

        // Get optimal grid dimensions
        inline dim3 get_grid_size(int64_t n, int block_size = BLOCK_SIZE) {
            return dim3((n + block_size - 1) / block_size);
        }
    } // namespace cuda
} // namespace tensoraxx
