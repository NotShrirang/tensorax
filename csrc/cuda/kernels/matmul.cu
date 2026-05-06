#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"
#include "profiling.cuh"

#include <cassert>
#include <cmath>

#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

namespace tensorax {
    // Simple matrix multiplication kernel (naive implementation)
    __global__ void matmul_kernel_naive(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, long long* prof_buf) {
        TX_TICK(prof_buf, 0);
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            int64_t q1 = k / 4, q2 = k / 2, q3 = (3 * k) / 4;
            TX_TICK(prof_buf, 1);
            for (int64_t i = 0; i < k; ++i) {
                sum += a[row * k + i] * b[i * n + col];
                if (i + 1 == q1) TX_TICK(prof_buf, 2);
                else if (i + 1 == q2) TX_TICK(prof_buf, 3);
                else if (i + 1 == q3) TX_TICK(prof_buf, 4);
            }
            TX_TICK(prof_buf, 5);
            c[row * n + col] = sum;
            TX_TICK(prof_buf, 6);
        }
    }

    // Tiled matrix multiplication kernel (optimized)
    __global__ void matmul_kernel_tiled(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n, int k, long long* prof_buf) {
        TX_TICK(prof_buf, 0);
        constexpr int TILE_SIZE = cuda::WARP_SIZE;
        __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        float sum = 0.0f;
        int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
        TX_TICK(prof_buf, 1);

        for (int t = 0; t < numTiles; ++t) {
            if (row < m && t * TILE_SIZE + threadIdx.x < k) {
                tile_a[threadIdx.y][threadIdx.x] = a[row * k + t * TILE_SIZE + threadIdx.x];
            } else {
                tile_a[threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (t == 0) TX_TICK(prof_buf, 2);

            if (col < n && t * TILE_SIZE + threadIdx.y < k) {
                tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
            } else {
                tile_b[threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (t == 0) TX_TICK(prof_buf, 3);

            __syncthreads();

            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i) {
                sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
            }
            if (t == 0) TX_TICK(prof_buf, 4);

            __syncthreads();
        }
        TX_TICK(prof_buf, 5);

        if (row < m && col < n) {
            c[row * n + col] = sum;
        }
        TX_TICK(prof_buf, 6);
    }


    __global__ void matmul_kernel_shared_memory_coalesced(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, float alpha, float beta, long long* prof_buf) {
        TX_TICK(prof_buf, 0);
        const int x = blockIdx.y * cuda::WARP_SIZE + (threadIdx.x / cuda::WARP_SIZE);
        const int y = blockIdx.x * cuda::WARP_SIZE + (threadIdx.x % cuda::WARP_SIZE);

        if (x < m && y < n) {
            float sum = 0.0f;
            int64_t q1 = k / 4, q2 = k / 2, q3 = (3 * k) / 4;
            TX_TICK(prof_buf, 1);
            for (int64_t i = 0; i < k; ++i) {
                sum += a[x * k + i] * b[i * n + y];
                if (i + 1 == q1) TX_TICK(prof_buf, 2);
                else if (i + 1 == q2) TX_TICK(prof_buf, 3);
                else if (i + 1 == q3) TX_TICK(prof_buf, 4);
            }
            TX_TICK(prof_buf, 5);
            c[x * n + y] = alpha * sum + beta * c[x * n + y];
            TX_TICK(prof_buf, 6);
        }
    }

    __global__ void matmul_kernel_shared_memory_cache_blocking(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, float alpha, float beta, long long* prof_buf) {
        const int BLOCKSIZE = cuda::WARP_SIZE;
        __shared__ float shared_a[BLOCKSIZE * BLOCKSIZE];
        __shared__ float shared_b[BLOCKSIZE * BLOCKSIZE];

        TX_TICK(prof_buf, 0);
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
        bool first_iter = true;
        TX_TICK(prof_buf, 1);

        for (int bkIdx = 0; bkIdx < k; bkIdx += BLOCKSIZE) {
            if (globalRow < m && bkIdx + threadCol < k) {
                shared_a[threadRow * BLOCKSIZE + threadCol] = A[threadRow * k + threadCol];
            } else {
                shared_a[threadRow * BLOCKSIZE + threadCol] = 0.0f;
            }
            if (first_iter) TX_TICK(prof_buf, 2);

            if (globalCol < n && bkIdx + threadRow < k) {
                shared_b[threadRow * BLOCKSIZE + threadCol] = B[threadRow * n + threadCol];
            } else {
                shared_b[threadRow * BLOCKSIZE + threadCol] = 0.0f;
            }
            if (first_iter) TX_TICK(prof_buf, 3);

            __syncthreads();

            A += BLOCKSIZE;
            B += BLOCKSIZE * n;

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
                tmp += shared_a[threadRow * BLOCKSIZE + dotIdx] *
                       shared_b[dotIdx * BLOCKSIZE + threadCol];
            }
            if (first_iter) TX_TICK(prof_buf, 4);
            first_iter = false;

            __syncthreads();
        }
        TX_TICK(prof_buf, 5);

        if (globalRow < m && globalCol < n) {
            C[threadRow * n + threadCol] = alpha * tmp + beta * C[threadRow * n + threadCol];
        }
        TX_TICK(prof_buf, 6);
    }

    template<const int BM, const int BN, const int BK, const int TM>
    __global__ void matmul_kernel_1d_blocktiling(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, float alpha, float beta, long long* prof_buf) {
        TX_TICK(prof_buf, 0);
        const uint cRow = blockIdx.y;
        const uint cCol = blockIdx.x;

        const int threadCol = threadIdx.x % BN;
        const int threadRow = threadIdx.x / BN;

        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        a += cRow * BM * k;
        b += cCol * BN;
        c += cRow * BM * n + cCol * BN;

        const uint innerColA = threadIdx.x % BK;
        const uint innerRowA = threadIdx.x / BK;
        const uint innerColB = threadIdx.x % BN;
        const uint innerRowB = threadIdx.x / BN;

        float threadResults[TM] = {0.0f};
        bool first_iter = true;
        TX_TICK(prof_buf, 1);

        for (uint bkIdx = 0; bkIdx < k; bkIdx += BK) {
            shared_a[innerRowA * BK + innerColA] = a[innerRowA * k + innerColA];
            if (first_iter) TX_TICK(prof_buf, 2);
            shared_b[innerRowB * BN + innerColB] = b[innerRowB * n + innerColB];
            if (first_iter) TX_TICK(prof_buf, 3);

            __syncthreads();

            a += BK;
            b += BK * n;

            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                float tmpB = shared_b[dotIdx * BN + threadCol];
                #pragma unroll
                for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                    threadResults[resIdx] += shared_a[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
                }
            }
            if (first_iter) TX_TICK(prof_buf, 4);
            first_iter = false;
            __syncthreads();
        }
        TX_TICK(prof_buf, 5);

        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            c[(threadRow * TM + resIdx) * n + threadCol] = alpha * threadResults[resIdx] + beta * c[(threadRow * TM + resIdx) * n + threadCol];
        }
        TX_TICK(prof_buf, 6);
    }

    template<const int BM, const int BN, const int BK, const int TM, const int TN>
    // 2D blocktiling
    __global__ void matmul_kernel_2d_blocktiling(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k, float alpha, float beta, long long* prof_buf) {
        TX_TICK(prof_buf, 0);
        const uint cRow = blockIdx.y;
        const uint cCol = blockIdx.x;

        const uint totalResultsBlocktile = BM * BN;
        const uint  numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

        assert (numThreadsBlocktile == blockDim.x);

        const int threadCol = threadIdx.x % (BN / TN);
        const int threadRow = threadIdx.x / (BN / TN);

        __shared__ float shared_a[BM * BK];
        __shared__ float shared_b[BK * BN];

        a += cRow * BM * k;
        b += cCol * BN;
        c += cRow * BM * n + cCol * BN;

        const uint innerRowA = threadIdx.x / BK;
        const uint innerColA = threadIdx.x % BK;

        const uint strideA = numThreadsBlocktile / BK;
        const uint innerRowB = threadIdx.x / BN;
        const uint innerColB = threadIdx.x % BN;

        const uint strideB = numThreadsBlocktile / BN;

        float threadResults[TM][TN] = {0.0f};

        float registerA[TM] = {0.0f};
        float registerB[TN] = {0.0f};

        TX_TICK(prof_buf, 1);
        bool first_iter = true;
        for (uint bkIdx = 0; bkIdx < k; bkIdx += BK) {
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                shared_a[(innerRowA + loadOffset) * BK + innerColA] =
                    a[(innerRowA + loadOffset) * k + innerColA];
            }
            if (first_iter) TX_TICK(prof_buf, 2);
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                shared_b[(innerRowB + loadOffset) * BN + innerColB] =
                    b[(innerRowB + loadOffset) * n + innerColB];
            }
            if (first_iter) TX_TICK(prof_buf, 3);

            __syncthreads();

            a += BK;
            b += BK * n;

            #pragma unroll
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                #pragma unroll
                for (uint i = 0; i < TM; ++i) {
                    registerA[i] = shared_a[(threadRow * TM + i) * BK + dotIdx];
                }
                #pragma unroll
                for (uint i = 0; i < TN; ++i) {
                    registerB[i] = shared_b[dotIdx * BN + threadCol * TN + i];
                }
                #pragma unroll
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    #pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resIdxM][resIdxN] += registerA[resIdxM] * registerB[resIdxN];
                    }
                }
            }
            if (first_iter) TX_TICK(prof_buf, 4);
            first_iter = false;
            __syncthreads();
        }
        TX_TICK(prof_buf, 5);

        for (uint tmIdx = 0; tmIdx < TM; ++tmIdx) {
            for (uint tnIdx = 0; tnIdx < TN; ++tnIdx) {
                c[(threadRow * TM + tmIdx) * n + threadCol * TN + tnIdx] =
                    alpha * threadResults[tmIdx][tnIdx] +
                    beta * c[(threadRow * TM + tmIdx) * n + threadCol * TN + tnIdx];
            }
        }
        TX_TICK(prof_buf, 6);
    }

    __device__ __forceinline__ void mma_m16n8k8_tf32(
        float d[4], const float a[4], const float b[2], const float c[4]) {
        // tf32 multiplicands are passed via .b32 (uint32_t) operands — the
        // hardware truncates the f32 mantissa to 10 bits during the op.
        uint32_t a0 = __float_as_uint(a[0]);
        uint32_t a1 = __float_as_uint(a[1]);
        uint32_t a2 = __float_as_uint(a[2]);
        uint32_t a3 = __float_as_uint(a[3]);
        uint32_t b0 = __float_as_uint(b[0]);
        uint32_t b1 = __float_as_uint(b[1]);
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
        );
    }

    __global__ void matmul_kernel_mma_tf32(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K
    ) {
        constexpr int BM = 128, BN = 128, BK = 16;
        constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;
        constexpr int WARP_M = 64, WARP_N = 64;
        constexpr int M_TILES = WARP_M / MMA_M;   // 4
        constexpr int N_TILES = WARP_N / MMA_N;   // 8
        constexpr int SUB_K   = BK / MMA_K;        // 2

        int block_m = blockIdx.y * BM;
        int block_n = blockIdx.x * BN;
        int tid = threadIdx.x;
        int warp_id = tid >> 5;             // 0..3
        int lane    = tid & 31;
        int warp_m_off = (warp_id / 2) * WARP_M;
        int warp_n_off = (warp_id % 2) * WARP_N;

        __shared__ float s_a[BM * BK];
        __shared__ float s_b[BK * BN];

        float acc[M_TILES][N_TILES][4] = {{{0.0f}}};

        int K_chunks = K / BK;
        for (int k_chunk = 0; k_chunk < K_chunks; k_chunk++) {
            int k_base = k_chunk * BK;

            // Load A tile [BM × BK]: 128*16/4 = 512 float4 / 128 threads = 4 each
            #pragma unroll
            for (int i = tid; i < BM * BK / 4; i += 128) {
                int idx4 = i * 4;
                int row = idx4 / BK;
                int col = idx4 % BK;
                *reinterpret_cast<float4*>(&s_a[row * BK + col]) =
                    *reinterpret_cast<const float4*>(&A[(block_m + row) * K + k_base + col]);
            }
            // Load B tile [BK × BN]: 16*128/4 = 512 float4 / 128 threads = 4 each
            #pragma unroll
            for (int i = tid; i < BK * BN / 4; i += 128) {
                int idx4 = i * 4;
                int row = idx4 / BN;
                int col = idx4 % BN;
                *reinterpret_cast<float4*>(&s_b[row * BN + col]) =
                    *reinterpret_cast<const float4*>(&B[(k_base + row) * N + block_n + col]);
            }
            __syncthreads();

            #pragma unroll
            for (int sub_k = 0; sub_k < SUB_K; sub_k++) {
                int k_off = sub_k * MMA_K;

                float a_frag[M_TILES][4];
                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    int m_base = warp_m_off + m * MMA_M;
                    a_frag[m][0] = s_a[(m_base + lane / 4)     * BK + k_off + (lane % 4)];
                    a_frag[m][1] = s_a[(m_base + lane / 4 + 8) * BK + k_off + (lane % 4)];
                    a_frag[m][2] = s_a[(m_base + lane / 4)     * BK + k_off + (lane % 4) + 4];
                    a_frag[m][3] = s_a[(m_base + lane / 4 + 8) * BK + k_off + (lane % 4) + 4];
                }

                float b_frag[N_TILES][2];
                #pragma unroll
                for (int n = 0; n < N_TILES; n++) {
                    int n_base = warp_n_off + n * MMA_N;
                    b_frag[n][0] = s_b[(k_off + (lane % 4))     * BN + n_base + (lane / 4)];
                    b_frag[n][1] = s_b[(k_off + (lane % 4) + 4) * BN + n_base + (lane / 4)];
                }

                #pragma unroll
                for (int m = 0; m < M_TILES; m++) {
                    #pragma unroll
                    for (int n = 0; n < N_TILES; n++) {
                        mma_m16n8k8_tf32(acc[m][n], a_frag[m], b_frag[n], acc[m][n]);
                    }
                }
            }
            __syncthreads();
        }

        // Write output. Each thread holds 8 cells per (m_tile, n_tile): 4 for two
        // (row, col) pairs at row r0 = lane/4 and r0+8.
        int row = lane / 4;
        int col = (lane % 4) * 2;
        #pragma unroll
        for (int m = 0; m < M_TILES; m++) {
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                int m_base = block_m + warp_m_off + m * MMA_M;
                int n_base = block_n + warp_n_off + n * MMA_N;
                C[(m_base + row)     * N + n_base + col]     = acc[m][n][0];
                C[(m_base + row)     * N + n_base + col + 1] = acc[m][n][1];
                C[(m_base + row + 8) * N + n_base + col]     = acc[m][n][2];
                C[(m_base + row + 8) * N + n_base + col + 1] = acc[m][n][3];
            }
        }
    }

    void matmul_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k) {
        TENSORAX_NVTX_RANGE("matmul.default");
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
            
            matmul_kernel_naive<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, nullptr);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Host-side wrapper for batched matrix multiplication
    void matmul_tiled_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k) {
        TENSORAX_NVTX_RANGE("matmul.tiled");
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
            
            matmul_kernel_tiled<<<grid, block>>>(a_batch, b_batch, c_batch, (int)m, (int)n, (int)k, nullptr);
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_shared_memory_coalesced_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
        TENSORAX_NVTX_RANGE("matmul.shared_memory_coalesced");
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
            matmul_kernel_shared_memory_coalesced<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, alpha, beta, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_shared_memory_cache_blocking_cuda(const float* a, const float* b, float* c, int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha, float beta) {
        TENSORAX_NVTX_RANGE("matmul.shared_memory_cache_blocking");
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
            matmul_kernel_shared_memory_cache_blocking<<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, alpha, beta, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_1d_blocktiling_cuda(
        const float* a, const float* b, float* c,
        int64_t batch_size, int64_t m, int64_t n, int64_t k,
        float alpha, float beta
    ) {
        TENSORAX_NVTX_RANGE("matmul.block_tiling_1d");
        const uint BM = 64; // Block size for m dimension
        const uint BN = 64; // Block size for cache blocking
        const uint BK = 8;  // Block size for k dimension
        const uint TM = 8;  // Tile size for n dimension within cache block

        dim3 block((BM * BN) / TM);
        dim3 grid(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        
        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_c = m * n;
        
        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a + batch * matrix_size_a;
            const float* b_batch = b + batch * matrix_size_b;
            float* c_batch = c + batch * matrix_size_c;
            matmul_kernel_1d_blocktiling<BM, BN, BK, TM>
                <<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, alpha, beta, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_2d_blocktiling_cuda(
        const float* a, const float* b, float* c,
        int64_t batch_size, int64_t m, int64_t n, int64_t k,
        float alpha, float beta
    ) {
        TENSORAX_NVTX_RANGE("matmul.block_tiling_2d");
        const int BM = 32; // Block size for m dimension
        const int BN = 32; // Block size for n dimension
        const int BK = 8;  // Block size for k dimension
        const int TM = 4;  // Tile size for m dimension within block
        const int TN = 4;  // Tile size for n dimension within block

        dim3 block((BM * BN) / (TM * TN));
        dim3 grid(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_c = m * n;

        // Process each batch
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a + batch * matrix_size_a;
            const float* b_batch = b + batch * matrix_size_b;
            float* c_batch = c + batch * matrix_size_c;
            matmul_kernel_2d_blocktiling<BM, BN, BK, TM, TN>
                <<<grid, block>>>(a_batch, b_batch, c_batch, m, n, k, alpha, beta, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void matmul_mma_tf32_cuda(
        const float* a, const float* b, float* c,
        int64_t batch_size, int64_t m, int64_t n, int64_t k
    ) {
        TENSORAX_NVTX_RANGE("matmul.mma_tf32");
        constexpr int BM = 128, BN = 128, BK = 16;
        if (m % BM != 0 || n % BN != 0 || k % BK != 0) {
            // Fall back to 2d blocktiling with alpha=1, beta=0 for shape mismatch.
            matmul_2d_blocktiling_cuda(a, b, c, batch_size, m, n, k, 1.0f, 0.0f);
            return;
        }
        dim3 block(128);
        dim3 grid(n / BN, m / BM);

        int64_t matrix_size_a = m * k;
        int64_t matrix_size_b = k * n;
        int64_t matrix_size_c = m * n;

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            matmul_kernel_mma_tf32<<<grid, block>>>(
                a + batch * matrix_size_a,
                b + batch * matrix_size_b,
                c + batch * matrix_size_c,
                (int)m, (int)n, (int)k);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<long long> matmul_2d_blocktiling_profile_sections_cuda(
        const float* a, const float* b, float* c,
        int64_t m, int64_t n, int64_t k,
        float alpha, float beta
    ) {
        TENSORAX_NVTX_RANGE("matmul.block_tiling_2d.profile");
        const int BM = 32, BN = 32, BK = 8, TM = 4, TN = 4;
        dim3 block((BM * BN) / (TM * TN));
        dim3 grid(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        long long* d_buf = prof::alloc_buf();
        matmul_kernel_2d_blocktiling<BM, BN, BK, TM, TN>
            <<<grid, block>>>(a, b, c, m, n, k, alpha, beta, d_buf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return prof::read_buf(d_buf);
    };

    std::vector<long long> matmul_naive_profile_sections_cuda(
        const float* a, const float* b, float* c,
        int64_t m, int64_t n, int64_t k
    ) {
        TENSORAX_NVTX_RANGE("matmul.default.profile");
        dim3 block(cuda::WARP_SIZE, cuda::WARP_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::WARP_SIZE), CEIL_DIV(m, cuda::WARP_SIZE));
        long long* d_buf = prof::alloc_buf();
        matmul_kernel_naive<<<grid, block>>>(a, b, c, m, n, k, d_buf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return prof::read_buf(d_buf);
    }

    std::vector<long long> matmul_tiled_profile_sections_cuda(
        const float* a, const float* b, float* c,
        int64_t m, int64_t n, int64_t k
    ) {
        TENSORAX_NVTX_RANGE("matmul.tiled.profile");
        dim3 block(cuda::TILE_SIZE, cuda::TILE_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::TILE_SIZE), CEIL_DIV(m, cuda::TILE_SIZE));
        long long* d_buf = prof::alloc_buf();
        matmul_kernel_tiled<<<grid, block>>>(a, b, c, (int)m, (int)n, (int)k, d_buf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return prof::read_buf(d_buf);
    }

    std::vector<long long> matmul_shared_memory_coalesced_profile_sections_cuda(
        const float* a, const float* b, float* c,
        int64_t m, int64_t n, int64_t k, float alpha, float beta
    ) {
        TENSORAX_NVTX_RANGE("matmul.shared_memory_coalesced.profile");
        dim3 block(cuda::WARP_SIZE * cuda::WARP_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::WARP_SIZE), CEIL_DIV(m, cuda::WARP_SIZE));
        long long* d_buf = prof::alloc_buf();
        matmul_kernel_shared_memory_coalesced<<<grid, block>>>(a, b, c, m, n, k, alpha, beta, d_buf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return prof::read_buf(d_buf);
    }

    std::vector<long long> matmul_shared_memory_cache_blocking_profile_sections_cuda(
        const float* a, const float* b, float* c,
        int64_t m, int64_t n, int64_t k, float alpha, float beta
    ) {
        TENSORAX_NVTX_RANGE("matmul.shared_memory_cache_blocking.profile");
        dim3 block(cuda::WARP_SIZE, cuda::WARP_SIZE);
        dim3 grid(CEIL_DIV(n, cuda::WARP_SIZE), CEIL_DIV(m, cuda::WARP_SIZE));
        long long* d_buf = prof::alloc_buf();
        matmul_kernel_shared_memory_cache_blocking<<<grid, block>>>(a, b, c, m, n, k, alpha, beta, d_buf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return prof::read_buf(d_buf);
    }

    std::vector<long long> matmul_1d_blocktiling_profile_sections_cuda(
        const float* a, const float* b, float* c,
        int64_t m, int64_t n, int64_t k, float alpha, float beta
    ) {
        TENSORAX_NVTX_RANGE("matmul.block_tiling_1d.profile");
        const uint BM = 64, BN = 64, BK = 8, TM = 8;
        dim3 block((BM * BN) / TM);
        dim3 grid(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
        long long* d_buf = prof::alloc_buf();
        matmul_kernel_1d_blocktiling<BM, BN, BK, TM><<<grid, block>>>(a, b, c, m, n, k, alpha, beta, d_buf);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return prof::read_buf(d_buf);
    }
} // namespace tensoraxx
