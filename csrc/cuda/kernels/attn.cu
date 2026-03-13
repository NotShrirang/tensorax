#include "../cuda_utils.cuh"
#include "../../tensor_ops.h"

#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstdint>

#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

namespace tensorax {

namespace {

constexpr unsigned FULL_WARP_MASK = 0xffffffffu;
constexpr float SOFTMAX_MAX_UPDATE_THRESHOLD = 0.5f;

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_down_sync(FULL_WARP_MASK, v, 16);
    v += __shfl_down_sync(FULL_WARP_MASK, v, 8);
    v += __shfl_down_sync(FULL_WARP_MASK, v, 4);
    v += __shfl_down_sync(FULL_WARP_MASK, v, 2);
    v += __shfl_down_sync(FULL_WARP_MASK, v, 1);
    return v;
}

__device__ __forceinline__ void block_load_f32_vec4(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int count,
    int tid,
    int num_threads
) {
    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
    bool vec_aligned = ((src_addr | dst_addr) & 0xF) == 0;

    if (vec_aligned) {
        int n4 = count / 4;
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (int i = tid; i < n4; i += num_threads) {
            dst4[i] = src4[i];
        }
        for (int i = n4 * 4 + tid; i < count; i += num_threads) {
            dst[i] = src[i];
        }
    } else {
        for (int i = tid; i < count; i += num_threads) {
            dst[i] = src[i];
        }
    }
}

} // namespace

__global__ void sdpa_kernel_naive(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int d_k,
    int d_v,
    float scale
) {
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= num_heads || q_idx >= seq_len_q || v_idx >= d_v)
        return;

    int qkv_base = b * num_heads + h;
    const float* q_row = Q + qkv_base * seq_len_q * d_k + q_idx * d_k;
    const float* k_base = K + qkv_base * seq_len_k * d_k;
    const float* v_base = V + qkv_base * seq_len_k * d_v;
    const float* mask_row = mask ? mask + qkv_base * seq_len_q * seq_len_k + q_idx * seq_len_k : nullptr;

    float max_score = -FLT_MAX;
    for (int j = 0; j < seq_len_k; j++) {
        float score = 0.0f;
        for (int d = 0; d < d_k; d++) {
            score += q_row[d] * k_base[j * d_k + d];
        }
        score *= scale;
        if (mask_row) score += mask_row[j];
        if (score > max_score) max_score = score;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len_k; j++) {
        float score = 0.0f;
        for (int d = 0; d < d_k; d++) {
            score += q_row[d] * k_base[j * d_k + d];
        }
        score *= scale;
        if (mask_row) score += mask_row[j];
        sum_exp += expf(score - max_score);
    }

    float result = 0.0f;
    for (int j = 0; j < seq_len_k; j++) {
        float score = 0.0f;
        for (int d = 0; d < d_k; d++) {
            score += q_row[d] * k_base[j * d_k + d];
        }
        score *= scale;
        if (mask_row) score += mask_row[j];
        float attn_weight = expf(score - max_score) / sum_exp;
        result += attn_weight * v_base[j * d_v + v_idx];
    }

    out[qkv_base * seq_len_q * d_v + q_idx * d_v + v_idx] = result;
}

__global__ void sdpa_kernel_tiled(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int d_k,
    int d_v,
    float scale,
    int tile_k
) {
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || h >= num_heads || q_idx >= seq_len_q)
        return;

    int qkv_base = b * num_heads + h;
    const float* q_row = Q + qkv_base * seq_len_q * d_k + q_idx * d_k;
    const float* k_base = K + qkv_base * seq_len_k * d_k;
    const float* v_base = V + qkv_base * seq_len_k * d_v;
    const float* mask_row = mask ? mask + qkv_base * seq_len_q * seq_len_k + q_idx * seq_len_k : nullptr;

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = s_k + tile_k * d_k;

    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc = 0.0f;

    for (int tile_start = 0; tile_start < seq_len_k; tile_start += tile_k) {
        int tile_end = min(tile_start + tile_k, seq_len_k);
        int tile_len = tile_end - tile_start;

        for (int i = threadIdx.y; i < tile_len; i += blockDim.y) {
            for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
                s_k[i * d_k + d] = k_base[(tile_start + i) * d_k + d];
            }
            if (v_idx < d_v) {
                s_v[i * d_v + v_idx] = v_base[(tile_start + i) * d_v + v_idx];
            }
        }
        __syncthreads();

        if (v_idx < d_v) {
            for (int j = 0; j < tile_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    score += q_row[d] * s_k[j * d_k + d];
                }
                score *= scale;
                if (mask_row) score += mask_row[tile_start + j];

                float new_max = fmaxf(max_score, score);
                float exp_old = expf(max_score - new_max);
                float exp_new = expf(score - new_max);

                acc = acc * exp_old + exp_new * s_v[j * d_v + v_idx];
                sum_exp = sum_exp * exp_old + exp_new;
                max_score = new_max;
            }
        }
        __syncthreads();
    }

    if (v_idx < d_v) {
        out[qkv_base * seq_len_q * d_v + q_idx * d_v + v_idx] = acc / sum_exp;
    }
}

__global__ void sdpa_kernel_flash(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int d_k,
    int d_v,
    float scale,
    int br,
    int bc
) {
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_block = blockIdx.y;
    int q_start = q_block * br;

    if (b >= batch_size || h >= num_heads)
        return;

    int tid = threadIdx.x;
    int qkv_base = b * num_heads + h;

    const float* q_base = Q + qkv_base * seq_len_q * d_k;
    const float* k_ptr = K + qkv_base * seq_len_k * d_k;
    const float* v_ptr = V + qkv_base * seq_len_k * d_v;
    const float* mask_base = mask ? mask + qkv_base * seq_len_q * seq_len_k : nullptr;
    float* out_base = out + qkv_base * seq_len_q * d_v;

    extern __shared__ float flash_smem[];
    float* s_q = flash_smem;
    float* s_k = s_q + br * d_k;
    float* s_v = s_k + bc * d_k;
    float* s_scores = s_v + bc * d_v;
    float* s_o = s_scores + br * bc;
    float* s_m = s_o + br * d_v;
    float* s_l = s_m + br;

    int q_end = min(q_start + br, seq_len_q);
    int q_len = q_end - q_start;

    for (int i = tid; i < q_len * d_k; i += blockDim.x) {
        int qi = i / d_k;
        int d = i % d_k;
        s_q[qi * d_k + d] = q_base[(q_start + qi) * d_k + d];
    }
    for (int i = tid; i < q_len * d_v; i += blockDim.x) {
        s_o[i] = 0.0f;
    }
    for (int i = tid; i < q_len; i += blockDim.x) {
        s_m[i] = -FLT_MAX;
        s_l[i] = 0.0f;
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < seq_len_k; kv_start += bc) {
        int kv_end = min(kv_start + bc, seq_len_k);
        int kv_len = kv_end - kv_start;

        for (int i = tid; i < kv_len * d_k; i += blockDim.x) {
            int ki = i / d_k;
            int d = i % d_k;
            s_k[ki * d_k + d] = k_ptr[(kv_start + ki) * d_k + d];
        }
        for (int i = tid; i < kv_len * d_v; i += blockDim.x) {
            int vi = i / d_v;
            int d = i % d_v;
            s_v[vi * d_v + d] = v_ptr[(kv_start + vi) * d_v + d];
        }
        __syncthreads();

        for (int qi = tid; qi < q_len; qi += blockDim.x) {
            float row_max = s_m[qi];
            float row_sum = s_l[qi];

            float new_max = row_max;
            for (int kj = 0; kj < kv_len; kj++) {
                float score = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    score += s_q[qi * d_k + d] * s_k[kj * d_k + d];
                }
                score *= scale;
                if (mask_base) score += mask_base[(q_start + qi) * seq_len_k + kv_start + kj];
                s_scores[qi * bc + kj] = score;
                if (score > new_max) new_max = score;
            }

            float correction = expf(row_max - new_max);
            float new_sum = row_sum * correction;

            for (int d = 0; d < d_v; d++) {
                s_o[qi * d_v + d] *= correction;
            }

            for (int kj = 0; kj < kv_len; kj++) {
                float p = expf(s_scores[qi * bc + kj] - new_max);
                new_sum += p;
                for (int d = 0; d < d_v; d++) {
                    s_o[qi * d_v + d] += p * s_v[kj * d_v + d];
                }
            }

            s_m[qi] = new_max;
            s_l[qi] = new_sum;
        }
        __syncthreads();
    }

    for (int i = tid; i < q_len * d_v; i += blockDim.x) {
        int qi = i / d_v;
        int d = i % d_v;
        out_base[(q_start + qi) * d_v + d] = s_o[qi * d_v + d] / s_l[qi];
    }
}

__global__ void sdpa_kernel_flash_optimized(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int d_k,
    int d_v,
    float scale,
    int br,
    int bc
) {
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_block = blockIdx.y;
    int q_start = q_block * br;

    if (b >= batch_size || h >= num_heads)
        return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = max(1, blockDim.x >> 5);
    int qkv_base = b * num_heads + h;

    const float* q_base = Q + qkv_base * seq_len_q * d_k;
    const float* k_ptr = K + qkv_base * seq_len_k * d_k;
    const float* v_ptr = V + qkv_base * seq_len_k * d_v;
    const float* mask_base = mask ? mask + qkv_base * seq_len_q * seq_len_k : nullptr;
    float* out_base = out + qkv_base * seq_len_q * d_v;

    extern __shared__ float flash_smem[];
    float* s_q = flash_smem;
    float* s_k = s_q + br * d_k;
    float* s_v = s_k + bc * d_k;
    float* s_scores = s_v + bc * d_v;
    float* s_o = s_scores + br * bc;
    float* s_m = s_o + br * d_v;
    float* s_l = s_m + br;

    int q_end = min(q_start + br, seq_len_q);
    int q_len = q_end - q_start;

    block_load_f32_vec4(
        q_base + q_start * d_k,
        s_q,
        q_len * d_k,
        tid,
        blockDim.x
    );
    for (int i = tid; i < q_len * d_v; i += blockDim.x) {
        s_o[i] = 0.0f;
    }
    for (int i = tid; i < q_len; i += blockDim.x) {
        s_m[i] = -FLT_MAX;
        s_l[i] = 0.0f;
    }
    __syncthreads();

    for (int kv_start = 0; kv_start < seq_len_k; kv_start += bc) {
        int kv_end = min(kv_start + bc, seq_len_k);
        int kv_len = kv_end - kv_start;

        block_load_f32_vec4(
            k_ptr + kv_start * d_k,
            s_k,
            kv_len * d_k,
            tid,
            blockDim.x
        );
        block_load_f32_vec4(
            v_ptr + kv_start * d_v,
            s_v,
            kv_len * d_v,
            tid,
            blockDim.x
        );
        __syncthreads();

        for (int qi = warp_id; qi < q_len; qi += num_warps) {
            float row_max = s_m[qi];
            float row_sum = s_l[qi];

            float tile_max = row_max;
            for (int kj = 0; kj < kv_len; kj++) {
                float acc = 0.0f;
                const float* q_row = &s_q[qi * d_k];
                const float* k_row = &s_k[kj * d_k];
                for (int d = lane; d < d_k; d += 32) {
                    acc += q_row[d] * k_row[d];
                }
                float score = warp_sum(acc);
                if (lane == 0) {
                    score *= scale;
                    if (mask_base) {
                        score += mask_base[(q_start + qi) * seq_len_k + kv_start + kj];
                    }
                    s_scores[qi * bc + kj] = score;
                    tile_max = fmaxf(tile_max, score);
                }
            }

            __syncwarp();

            float used_max = row_max;
            float correction = 1.0f;
            int do_rescale = 0;
            if (lane == 0) {
                float delta = tile_max - row_max;
                if (delta > SOFTMAX_MAX_UPDATE_THRESHOLD) {
                    used_max = tile_max;
                    correction = __expf(row_max - used_max);
                    do_rescale = 1;
                }
            }

            used_max = __shfl_sync(FULL_WARP_MASK, used_max, 0);
            correction = __shfl_sync(FULL_WARP_MASK, correction, 0);
            do_rescale = __shfl_sync(FULL_WARP_MASK, do_rescale, 0);

            if (do_rescale) {
                for (int d = lane; d < d_v; d += 32) {
                    s_o[qi * d_v + d] *= correction;
                }
            }

            float new_sum = row_sum * correction;
            for (int kj = 0; kj < kv_len; kj++) {
                float p = __expf(s_scores[qi * bc + kj] - used_max);
                if (lane == 0) {
                    new_sum += p;
                }
                for (int d = lane; d < d_v; d += 32) {
                    s_o[qi * d_v + d] += p * s_v[kj * d_v + d];
                }
            }

            if (lane == 0) {
                s_m[qi] = used_max;
                s_l[qi] = new_sum;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < q_len * d_v; i += blockDim.x) {
        int qi = i / d_v;
        int d = i % d_v;
        out_base[(q_start + qi) * d_v + d] = s_o[qi * d_v + d] / s_l[qi];
    }
}

void sdpa_naive_cuda(
    const float* Q,
    const float* K,
    const float* V,
    const float* mask,
    float* out,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t d_k,
    int64_t d_v
) {
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));

    dim3 block(16, 16);
    dim3 grid(
        CEIL_DIV(d_v, block.x),
        CEIL_DIV(seq_len_q, block.y),
        batch_size * num_heads
    );

    sdpa_kernel_naive<<<grid, block>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sdpa_tiled_cuda(
    const float* Q,
    const float* K,
    const float* V,
    const float* mask,
    float* out,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t d_k,
    int64_t d_v
) {
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    int tile_k = cuda::compute_tiled_tile_k(d_k, d_v);

    dim3 block(16, 16);
    dim3 grid(
        CEIL_DIV(d_v, block.x),
        CEIL_DIV(seq_len_q, block.y),
        batch_size * num_heads
    );

    size_t smem_size = ((size_t)tile_k * d_k + (size_t)tile_k * d_v) * sizeof(float);

    sdpa_kernel_tiled<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, tile_k
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sdpa_flash_cuda(
    const float* Q,
    const float* K,
    const float* V,
    const float* mask,
    float* out,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t d_k,
    int64_t d_v
) {
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    int br, bc;
    cuda::compute_flash_tiles(d_k, d_v, br, bc);

    int threads = 128;
    dim3 block(threads);
    dim3 grid(
        1,
        CEIL_DIV(seq_len_q, br),
        batch_size * num_heads
    );

    size_t smem_size = ((size_t)br * d_k
                      + (size_t)bc * d_k
                      + (size_t)bc * d_v
                      + (size_t)br * bc
                      + (size_t)br * d_v
                      + (size_t)br
                      + (size_t)br) * sizeof(float);

    sdpa_kernel_flash<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, br, bc
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sdpa_optimized_flash_cuda(
    const float* Q,
    const float* K,
    const float* V,
    const float* mask,
    float* out,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t d_k,
    int64_t d_v
) {
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    int br, bc;
    cuda::compute_flash_tiles(d_k, d_v, br, bc);

    int threads = 128;
    dim3 block(threads);
    dim3 grid(
        1,
        CEIL_DIV(seq_len_q, br),
        batch_size * num_heads
    );

    size_t smem_size = ((size_t)br * d_k
                      + (size_t)bc * d_k
                      + (size_t)bc * d_v
                      + (size_t)br * bc
                      + (size_t)br * d_v
                      + (size_t)br
                      + (size_t)br) * sizeof(float);

    sdpa_kernel_flash_optimized<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, br, bc
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


} // namespace tensorax
