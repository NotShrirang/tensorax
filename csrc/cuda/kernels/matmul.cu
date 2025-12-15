#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"

#include <cmath>

#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

namespace tensorax {
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
    __global__ void matmul_kernel_tiled(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n, int k) {
        constexpr int TILE_SIZE = cuda::WARP_SIZE;
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
            
            #pragma unroll
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
        const int x = blockIdx.y * cuda::WARP_SIZE + (threadIdx.x / cuda::WARP_SIZE);
        const int y = blockIdx.x * cuda::WARP_SIZE + (threadIdx.x % cuda::WARP_SIZE);

        if (x < m && y < n) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += a[x * k + i] * b[i * n + y];
            }
            c[x * n + y] = alpha * sum + beta * c[x * n + y];
        }
    }

    __global__ void matmul_kernel_shared_memory_cache_blocking(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
        const int BLOCKSIZE = cuda::WARP_SIZE;
        __shared__ float shared_a[BLOCKSIZE * BLOCKSIZE];
        __shared__ float shared_b[BLOCKSIZE * BLOCKSIZE];

        const int cRow = blockIdx.y;
        const int cCol = blockIdx.x;

        const int threadRow = threadIdx.y;
        const int threadCol = threadIdx.x;

        const int globalRow = cRow * BLOCKSIZE + threadRow;
        const int globalCol = cCol * BLOCKSIZE + threadCol;

        const float* A = a + cRow * BLOCKSIZE * k;
        const float* B = b + cCol * BLOCKSIZE;
        float* C = c + cRow * BLOCKSIZE * n + cCol * BLOCKSIZE;
        
        float tmp = 0.0f;

        for (int bkIdx = 0; bkIdx < k; bkIdx += BLOCKSIZE) {
            if (globalRow < m && bkIdx + threadCol < k) {
                shared_a[threadRow * BLOCKSIZE + threadCol] = A[threadRow * k + threadCol];
            } else {
                shared_a[threadRow * BLOCKSIZE + threadCol] = 0.0f;
            }
            
            if (globalCol < n && bkIdx + threadRow < k) {
                shared_b[threadRow * BLOCKSIZE + threadCol] = B[threadRow * n + threadCol];
            } else {
                shared_b[threadRow * BLOCKSIZE + threadCol] = 0.0f;
            }

            __syncthreads();

            A += BLOCKSIZE;
            B += BLOCKSIZE * n;

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
                tmp += shared_a[threadRow * BLOCKSIZE + dotIdx] * 
                       shared_b[dotIdx * BLOCKSIZE + threadCol];
            }

            __syncthreads();
        }

        if (globalRow < m && globalCol < n) {
            C[threadRow * n + threadCol] = alpha * tmp + beta * C[threadRow * n + threadCol];
        }
    }

    void matmul_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k) {
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
            
            matmul_kernel_tiled<<<grid, block>>>(a_batch, b_batch, c_batch, (int)m, (int)n, (int)k);
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_cuda_shared_memory_coalesced_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
        dim3 block(cuda::WARP_SIZE * cuda::WARP_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::WARP_SIZE), CEIL_DIV(m, cuda::WARP_SIZE));
        
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

    void matmul_cuda_shared_memory_cache_blocking_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
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
            matmul_kernel_shared_memory_cache_blocking<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, alpha, beta);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
} // namespace tensoraxx
