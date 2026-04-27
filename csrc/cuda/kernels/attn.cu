#include "../cuda_utils.cuh"
#include "../../tensor_ops.h"

#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>

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

__device__ __forceinline__ void ldmatrix_m16n8_x4_b16(uint32_t regs[4], const void* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_m16n8_x2_b16(uint32_t regs[2], const void* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16_fp32_fp16_fp16_fp32(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// PTX Intrinsic for Fast Approximate Exponential (Base 2)
// This is much faster than the standard math.h expf()
__device__ __forceinline__ float exp2_approx(float x) {
    float res;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

// Wrapper for base-e exponential using the base-2 PTX intrinsic
__device__ __forceinline__ float exp_approx(float x) {
    // exp(x) = exp2(x * log2(e))
    // log2(e) ≈ 1.4426950408889634
    return exp2_approx(x * 1.4426950408889634f);
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

__global__ void sdpa_kernel_mma(
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
    int q_start = blockIdx.y * 16;

    if (b >= batch_size || h >= num_heads || q_start >= seq_len_q) return;

    int tid = threadIdx.x;
    int qkv_base = b * num_heads + h;

    const float* q_base = Q + qkv_base * seq_len_q * d_k;
    const float* k_ptr  = K + qkv_base * seq_len_k * d_k;
    const float* v_ptr  = V + qkv_base * seq_len_k * d_v;
    float* out_base     = out + qkv_base * seq_len_q * d_v;

    extern __shared__ __half mma_smem[];
    __half* s_q = mma_smem;
    __half* s_k = s_q + 16 * 16;
    __half* s_v = s_k + 16 * 16;
    float* s_scores = reinterpret_cast<float*>(s_v + 16 * 16);

    int row = tid / 2;
    int col_chunk = (tid % 2) * 8;

    // Fused fp32 -> fp16 on the way into shared memory: read 8 floats per
    // thread (two float4), pack into four __half2, store as 8 halfs.
    const float4* gq = reinterpret_cast<const float4*>(q_base + (q_start + row) * d_k + col_chunk);
    float4 fq0 = gq[0];
    float4 fq1 = gq[1];
    __half2* sq = reinterpret_cast<__half2*>(s_q + row * 16 + col_chunk);
    sq[0] = __floats2half2_rn(fq0.x, fq0.y);
    sq[1] = __floats2half2_rn(fq0.z, fq0.w);
    sq[2] = __floats2half2_rn(fq1.x, fq1.y);
    sq[3] = __floats2half2_rn(fq1.z, fq1.w);

    const float4* gk = reinterpret_cast<const float4*>(k_ptr + row * d_k + col_chunk);
    float4 fk0 = gk[0];
    float4 fk1 = gk[1];
    __half2* sk = reinterpret_cast<__half2*>(s_k + row * 16 + col_chunk);
    sk[0] = __floats2half2_rn(fk0.x, fk0.y);
    sk[1] = __floats2half2_rn(fk0.z, fk0.w);
    sk[2] = __floats2half2_rn(fk1.x, fk1.y);
    sk[3] = __floats2half2_rn(fk1.z, fk1.w);

    __syncthreads();

    uint32_t reg_q[4]; 
    uint32_t reg_k0[2];
    uint32_t reg_k1[2];
    
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    int smem_row = tid % 16;
    int smem_col = (tid / 16) * 8;

    ldmatrix_m16n8_x4_b16(reg_q, s_q + smem_row * 16 + smem_col);

    ldmatrix_m16n8_x2_b16(reg_k0, s_k + smem_row * 16 + 0);

    ldmatrix_m16n8_x2_b16(reg_k1, s_k + smem_row * 16 + 8);

    mma_m16n8k16_fp32_fp16_fp16_fp32(acc0, reg_q, reg_k0, acc0);

    mma_m16n8k16_fp32_fp16_fp16_fp32(acc1, reg_q, reg_k1, acc1);

    int r0 = tid / 4;
    int r1 = tid / 4 + 8;
    int c0 = (tid % 4) * 2;

    s_scores[r0 * 16 + c0 + 0] = acc0[0];
    s_scores[r0 * 16 + c0 + 1] = acc0[1];
    s_scores[r1 * 16 + c0 + 0] = acc0[2];
    s_scores[r1 * 16 + c0 + 1] = acc0[3];

    s_scores[r0 * 16 + c0 + 8 + 0] = acc1[0];
    s_scores[r0 * 16 + c0 + 8 + 1] = acc1[1];
    s_scores[r1 * 16 + c0 + 8 + 0] = acc1[2];
    s_scores[r1 * 16 + c0 + 8 + 1] = acc1[3];
    
    __syncthreads();

    if (tid < 16) {
        float row_max = -FLT_MAX;

        for (int i = 0; i < 16; i++) {
            s_scores[tid * 16 + i] *= scale;
            row_max = fmaxf(row_max, s_scores[tid * 16 + i]);
        }
        
        float row_sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            float e = exp_approx(s_scores[tid * 16 + i] - row_max);
            s_scores[tid * 16 + i] = e;
            row_sum += e;
        }

        for (int i = 0; i < 16; i++) {
            s_k[tid * 16 + i] = __float2half(s_scores[tid * 16 + i] / row_sum); 
        }
    }
    
    __syncthreads();

    // Fused fp32 -> fp16 V load (transposed into s_v): one float4 pair per
    // thread; the transpose forces scalar stores into shared.
    int v_row = tid / 2;
    int v_col_chunk = (tid % 2) * 8;
    const float4* gv = reinterpret_cast<const float4*>(v_ptr + v_row * d_v + v_col_chunk);
    float4 fv0 = gv[0];
    float4 fv1 = gv[1];
    s_v[(v_col_chunk + 0) * 16 + v_row] = __float2half(fv0.x);
    s_v[(v_col_chunk + 1) * 16 + v_row] = __float2half(fv0.y);
    s_v[(v_col_chunk + 2) * 16 + v_row] = __float2half(fv0.z);
    s_v[(v_col_chunk + 3) * 16 + v_row] = __float2half(fv0.w);
    s_v[(v_col_chunk + 4) * 16 + v_row] = __float2half(fv1.x);
    s_v[(v_col_chunk + 5) * 16 + v_row] = __float2half(fv1.y);
    s_v[(v_col_chunk + 6) * 16 + v_row] = __float2half(fv1.z);
    s_v[(v_col_chunk + 7) * 16 + v_row] = __float2half(fv1.w);
    
    __syncthreads();

    ldmatrix_m16n8_x4_b16(reg_q, s_k + smem_row * 16 + smem_col);
    
    uint32_t reg_v0[2];
    uint32_t reg_v1[2];
    ldmatrix_m16n8_x2_b16(reg_v0, s_v + smem_row * 16 + 0);
    ldmatrix_m16n8_x2_b16(reg_v1, s_v + smem_row * 16 + 8);

    float out_acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    mma_m16n8k16_fp32_fp16_fp16_fp32(out_acc0, reg_q, reg_v0, out_acc0);
    mma_m16n8k16_fp32_fp16_fp16_fp32(out_acc1, reg_q, reg_v1, out_acc1);

    // Output is fp32 directly (no fp16 round-trip).
    out_base[r0 * d_v + c0 + 0] = out_acc0[0];
    out_base[r0 * d_v + c0 + 1] = out_acc0[1];
    out_base[r1 * d_v + c0 + 0] = out_acc0[2];
    out_base[r1 * d_v + c0 + 1] = out_acc0[3];

    out_base[r0 * d_v + c0 + 8 + 0] = out_acc1[0];
    out_base[r0 * d_v + c0 + 8 + 1] = out_acc1[1];
    out_base[r1 * d_v + c0 + 8 + 0] = out_acc1[2];
    out_base[r1 * d_v + c0 + 8 + 1] = out_acc1[3];
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


__global__ void cast_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 v = *reinterpret_cast<const float4*>(in + idx);
        __half2 lo = __floats2half2_rn(v.x, v.y);
        __half2 hi = __floats2half2_rn(v.z, v.w);
        *reinterpret_cast<__half2*>(out + idx)     = lo;
        *reinterpret_cast<__half2*>(out + idx + 2) = hi;
    } else {
        for (int i = 0; idx + i < size; ++i) {
            out[idx + i] = __float2half(in[idx + i]);
        }
    }
}

__global__ void cast_f16_to_f32(const __half* __restrict__ in, float* __restrict__ out, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        __half2 lo = *reinterpret_cast<const __half2*>(in + idx);
        __half2 hi = *reinterpret_cast<const __half2*>(in + idx + 2);
        float4 v;
        v.x = __low2float(lo);
        v.y = __high2float(lo);
        v.z = __low2float(hi);
        v.w = __high2float(hi);
        *reinterpret_cast<float4*>(out + idx) = v;
    } else {
        for (int i = 0; idx + i < size; ++i) {
            out[idx + i] = __half2float(in[idx + i]);
        }
    }
}

void sdpa_mma_cuda(
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

    dim3 block(32);
    dim3 grid(
        1,
        CEIL_DIV(seq_len_q, 16),
        batch_size * num_heads
    );

    size_t smem_size = (16 * 16 * 3) * sizeof(__half) + (16 * 16) * sizeof(float);

    sdpa_kernel_mma<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


} // namespace tensorax
