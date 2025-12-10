#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"

#include <cmath>

#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

namespace tensora {
    // Simple matrix multiplication kernel (naive implementation)
    __global__ void matmul_kernel_naive(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < m && col < n) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }

    // Tiled matrix multiplication kernel (optimized)
    __global__ void matmul_kernel_tiled(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k) {
        constexpr int TILE_SIZE = cuda::TILE_SIZE;
        __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
        
        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        
        float sum = 0.0f;
        
        // Loop over tiles
        for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            // Load tile from A
            if (row < m && t * TILE_SIZE + threadIdx.x < k) {
                tile_a[threadIdx.y][threadIdx.x] = a[row * k + t * TILE_SIZE + threadIdx.x];
            } else {
                tile_a[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load tile from B
            if (col < n && t * TILE_SIZE + threadIdx.y < k) {
                tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
            } else {
                tile_b[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial product
            for (int i = 0; i < TILE_SIZE; ++i) {
                sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        // Write result
        if (row < m && col < n) {
            c[row * n + col] = sum;
        }
    }


    __global__ void matmul_kernel_shared_memory_coalesced(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
        const int x = blockIdx.x * cuda::WARP_SIZE + (threadIdx.x / cuda::WARP_SIZE);
        const int y = blockIdx.y * cuda::WARP_SIZE + (threadIdx.x % cuda::WARP_SIZE);

        if (x < m && y < n) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += a[x * k + i] * b[i * n + y];
            }
            c[x * n + y] = alpha * sum + beta * c[x * n + y];
        }
    }

    void matmul_cuda(const float* a, const float* b, float* c,
                    int64_t batch_size, int64_t m, int64_t n, int64_t k) {
        // Use naive kernel for simplicity
        dim3 block(cuda::WARP_SIZE, cuda::WARP_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::WARP_SIZE), CEIL_DIV(m, cuda::WARP_SIZE));
        
        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_c = m * n;
        
        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a + batch * matrix_size_a;
            const float* b_batch = b + batch * matrix_size_b;
            float* c_batch = c + batch * matrix_size_c;
            
            matmul_kernel_naive<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Host-side wrapper for batched matrix multiplication
    void matmul_tiled_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k) {
        // Use tiled kernel for better performance
        dim3 block(cuda::TILE_SIZE, cuda::TILE_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::TILE_SIZE), CEIL_DIV(m, cuda::TILE_SIZE));
        
        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_c = m * n;
        
        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a + batch * matrix_size_a;
            const float* b_batch = b + batch * matrix_size_b;
            float* c_batch = c + batch * matrix_size_c;
            
            matmul_kernel_tiled<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k);
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_cuda_shared_memory_coalesced_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
        dim3 block(cuda::WARP_SIZE * cuda::WARP_SIZE);
        dim3 grid(CEIL_DIV(m, cuda::WARP_SIZE), CEIL_DIV(n, cuda::WARP_SIZE));
        
        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_c = m * n;
        
        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a + batch * matrix_size_a;
            const float* b_batch = b + batch * matrix_size_b;
            float* c_batch = c + batch * matrix_size_c;
            matmul_kernel_shared_memory_coalesced<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, alpha, beta);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
} // namespace tensora
