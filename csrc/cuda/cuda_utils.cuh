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

        constexpr size_t SMEM_LIMIT = 49152;

        inline int compute_tiled_tile_k(int64_t d_k, int64_t d_v, size_t smem_limit = SMEM_LIMIT) {
            int tile_k = 32;
            while (tile_k > 1) {
                size_t need = (size_t)tile_k * (d_k + d_v) * sizeof(float);
                if (need <= smem_limit) return tile_k;
                tile_k /= 2;
            }
            return 1;
        }

        inline void compute_flash_tiles(int64_t d_k, int64_t d_v, int &br, int &bc, size_t smem_limit = SMEM_LIMIT) {
            br = 32;
            bc = 32;
            while (br > 1 || bc > 1) {
                size_t need = ((size_t)br * d_k
                             + (size_t)bc * d_k
                             + (size_t)bc * d_v
                             + (size_t)br * bc
                             + (size_t)br * d_v
                             + (size_t)br
                             + (size_t)br) * sizeof(float);
                if (need <= smem_limit) return;
                if (br >= bc) br /= 2;
                else bc /= 2;
            }
        }
    } // namespace cuda
} // namespace tensoraxx
