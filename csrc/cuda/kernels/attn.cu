#include "../cuda_utils.cuh"
#include "../../tensor_ops.h"
#include "../profiling.cuh"

#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
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

// cp.async helpers (sm_80+). Stages 16 bytes (one float4 / eight halves) per
// thread directly from global to shared memory, bypassing the register file.
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
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
    float scale,
    long long* prof_buf
) {
    TX_TICK(prof_buf, 0);
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
    TX_TICK(prof_buf, 1);

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
    TX_TICK(prof_buf, 2);

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
    TX_TICK(prof_buf, 3);

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
    TX_TICK(prof_buf, 4);

    out[qkv_base * seq_len_q * d_v + q_idx * d_v + v_idx] = result;
    TX_TICK(prof_buf, 5);
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
    int tile_k,
    long long* prof_buf
) {
    TX_TICK(prof_buf, 0);
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
    bool first_tile = true;
    TX_TICK(prof_buf, 1);

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
        if (first_tile) TX_TICK(prof_buf, 2);
        __syncthreads();

        if (v_idx < d_v) {
            for (int j = 0; j < tile_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    score += q_row[d] * s_k[j * d_k + d];
                }
                score *= scale;
                if (mask_row) score += mask_row[tile_start + j];
                if (first_tile && j == 0) TX_TICK(prof_buf, 3);

                float new_max = fmaxf(max_score, score);
                float exp_old = expf(max_score - new_max);
                float exp_new = expf(score - new_max);
                if (first_tile && j == 0) TX_TICK(prof_buf, 4);

                acc = acc * exp_old + exp_new * s_v[j * d_v + v_idx];
                sum_exp = sum_exp * exp_old + exp_new;
                max_score = new_max;
                if (first_tile && j == 0) TX_TICK(prof_buf, 5);
            }
        }
        if (first_tile) TX_TICK(prof_buf, 6);
        first_tile = false;
        __syncthreads();
    }
    TX_TICK(prof_buf, 7);

    if (v_idx < d_v) {
        out[qkv_base * seq_len_q * d_v + q_idx * d_v + v_idx] = acc / sum_exp;
    }
    TX_TICK(prof_buf, 8);
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
    float scale,
    long long* prof_buf
) {
    constexpr int Q_ROWS  = 16;
    constexpr int M_TILES = Q_ROWS / 16;

    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_start = blockIdx.y * Q_ROWS;
    if (b >= batch_size || h >= num_heads || q_start >= seq_len_q) return;

    int tid       = threadIdx.x;
    int warp_id   = tid >> 5;
    int lane      = tid & 31;
    int qkv_base  = b * num_heads + h;
    const float* q_base = Q + qkv_base * seq_len_q * d_k;
    const float* k_ptr  = K + qkv_base * seq_len_k * d_k;
    const float* v_ptr  = V + qkv_base * seq_len_k * d_v;
    float* out_base     = out + qkv_base * seq_len_q * d_v;

    extern __shared__ __half mma_smem[];
    __half* s_q          = mma_smem;
    __half* s_p          = s_q + Q_ROWS * d_k;
    float*  s_kstage_f32 = reinterpret_cast<float*>(s_p + Q_ROWS * 16);
    __half* s_kchunk     = reinterpret_cast<__half*>(s_kstage_f32 + 2 * 16 * 16);
    float*  s_vstage_f32 = reinterpret_cast<float*>(s_kchunk + 2 * 16 * 16);
    __half* s_vchunk     = reinterpret_cast<__half*>(s_vstage_f32 + 4 * 2 * 16 * 16);
    float*  s_scores     = reinterpret_cast<float*>(s_vchunk + 4 * 2 * 16 * 16);
    float*  s_m          = s_scores + Q_ROWS * 16;
    float*  s_l          = s_m + Q_ROWS;
    float*  s_corr       = s_l + Q_ROWS;

    int r0         = lane / 4;
    int r1         = r0 + 8;
    int c0         = (lane % 4) * 2;
    int smem_row_a = lane % 16;
    int smem_col_a = (lane / 16) * 8;
    int b_col      = lane & 7;
    int b_khalf    = ((lane >> 3) & 1) * 8;

#ifdef TENSORAX_PROFILE
    bool log = (prof_buf != nullptr &&
                threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
                blockIdx.x  == 0 && blockIdx.y  == 0 && blockIdx.z  == 0);
    long long t_entry      = log ? clock64() : 0;
    long long t_setup_end  = 0, t_qload_end = 0, t_loop_end = 0;
    if (log) t_setup_end = clock64();
#endif

    int total_q4 = (Q_ROWS * d_k) / 4;
    for (int i = tid; i < total_q4; i += 128) {
        int idx = i * 4;
        int row = idx / d_k;
        int col = idx % d_k;
        bool in_range = (q_start + row) < seq_len_q;
        float4 v = in_range
            ? *reinterpret_cast<const float4*>(q_base + (q_start + row) * d_k + col)
            : float4{0.0f, 0.0f, 0.0f, 0.0f};
        __half2* s = reinterpret_cast<__half2*>(s_q + row * d_k + col);
        s[0] = __floats2half2_rn(v.x, v.y);
        s[1] = __floats2half2_rn(v.z, v.w);
    }

    if (tid < Q_ROWS) { s_m[tid] = -FLT_MAX; s_l[tid] = 0.0f; }

    constexpr int MAX_CHUNKS = 16;
    float oacc0[M_TILES][MAX_CHUNKS][4];
    float oacc1[M_TILES][MAX_CHUNKS][4];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; mt++) {
        #pragma unroll
        for (int i = 0; i < MAX_CHUNKS; i++) {
            oacc0[mt][i][0] = oacc0[mt][i][1] = oacc0[mt][i][2] = oacc0[mt][i][3] = 0.0f;
            oacc1[mt][i][0] = oacc1[mt][i][1] = oacc1[mt][i][2] = oacc1[mt][i][3] = 0.0f;
        }
    }
    __syncthreads();

#ifdef TENSORAX_PROFILE
    if (log) t_qload_end = clock64();
#endif

    int dk_chunks = d_k / 16;
    int dv_chunks = d_v / 16;

    auto issue_k_chunk = [&] (int kv_start, int dk, int buf) {
        int row = lane / 2;
        int col = (lane % 2) * 8;
        float* dst0 = s_kstage_f32 + buf * 256 + row * 16 + col;
        float* dst1 = dst0 + 4;
        const float* src0 = k_ptr + (kv_start + row) * d_k + dk + col;
        const float* src1 = src0 + 4;
        cp_async_16B(dst0, src0);
        cp_async_16B(dst1, src1);
    };
    auto cast_k_chunk = [&] (int buf) {
        int base = lane * 8;
        const float* src = s_kstage_f32 + buf * 256 + base;
        __half2* dst = reinterpret_cast<__half2*>(s_kchunk + buf * 256 + base);
        dst[0] = __floats2half2_rn(src[0], src[1]);
        dst[1] = __floats2half2_rn(src[2], src[3]);
        dst[2] = __floats2half2_rn(src[4], src[5]);
        dst[3] = __floats2half2_rn(src[6], src[7]);
    };

    auto issue_v_chunk_warp = [&] (int kv_start, int dv, int buf) {
        int k = lane / 2;
        int n = (lane % 2) * 8;
        int slot = (warp_id * 2 + buf) * 256;
        float* dst0 = s_vstage_f32 + slot + k * 16 + n;
        float* dst1 = dst0 + 4;
        const float* src0 = v_ptr + (kv_start + k) * d_v + dv + n;
        const float* src1 = src0 + 4;
        cp_async_16B(dst0, src0);
        cp_async_16B(dst1, src1);
    };
    auto cast_v_chunk_warp = [&] (int buf) {
        int k = lane / 2;
        int n0 = (lane % 2) * 8;
        int slot = (warp_id * 2 + buf) * 256;
        const float* src = s_vstage_f32 + slot + k * 16 + n0;
        __half* dst = s_vchunk + slot;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            dst[(n0 + i) * 16 + k] = __float2half(src[i]);
        }
    };

    if (warp_id == 0) {
        issue_k_chunk(0, 0, 0);
        cp_async_commit();
    }

    for (int kv_start = 0; kv_start < seq_len_k; kv_start += 16) {
        if (warp_id == 0) {
            float acc[M_TILES][2][4] = {{{0.0f}}};

            for (int dk_chunk = 0; dk_chunk < dk_chunks; dk_chunk++) {
                int dk = dk_chunk * 16;
                int cur_buf = dk_chunk & 1;

                if (dk_chunk + 1 < dk_chunks) {
                    int next_buf = (dk_chunk + 1) & 1;
                    issue_k_chunk(kv_start, dk + 16, next_buf);
                    cp_async_commit();
                    cp_async_wait_group<1>();
                } else {
                    cp_async_wait_all();
                }
                __syncwarp();

                cast_k_chunk(cur_buf);
                __syncwarp();

                uint32_t reg_q[M_TILES][4];
                uint32_t reg_k0[2];
                uint32_t reg_k1[2];
                #pragma unroll
                for (int mt = 0; mt < M_TILES; mt++) {
                    ldmatrix_m16n8_x4_b16(reg_q[mt],
                        s_q + (mt * 16 + smem_row_a) * d_k + dk + smem_col_a);
                }
                ldmatrix_m16n8_x2_b16(reg_k0, s_kchunk + cur_buf * 256 +  b_col      * 16 + b_khalf);
                ldmatrix_m16n8_x2_b16(reg_k1, s_kchunk + cur_buf * 256 + (b_col + 8) * 16 + b_khalf);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; mt++) {
                    mma_m16n8k16_fp32_fp16_fp16_fp32(acc[mt][0], reg_q[mt], reg_k0, acc[mt][0]);
                    mma_m16n8k16_fp32_fp16_fp16_fp32(acc[mt][1], reg_q[mt], reg_k1, acc[mt][1]);
                }
            }

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                int row_off = mt * 16;
                s_scores[(row_off + r0) * 16 + c0]         = acc[mt][0][0] * scale;
                s_scores[(row_off + r0) * 16 + c0 + 1]     = acc[mt][0][1] * scale;
                s_scores[(row_off + r1) * 16 + c0]         = acc[mt][0][2] * scale;
                s_scores[(row_off + r1) * 16 + c0 + 1]     = acc[mt][0][3] * scale;
                s_scores[(row_off + r0) * 16 + c0 + 8]     = acc[mt][1][0] * scale;
                s_scores[(row_off + r0) * 16 + c0 + 8 + 1] = acc[mt][1][1] * scale;
                s_scores[(row_off + r1) * 16 + c0 + 8]     = acc[mt][1][2] * scale;
                s_scores[(row_off + r1) * 16 + c0 + 8 + 1] = acc[mt][1][3] * scale;
            }

            __syncwarp();

            if (lane < Q_ROWS) {
                float old_max = s_m[lane];
                float tile_max = -FLT_MAX;
                for (int j = 0; j < 16; j++) {
                    tile_max = fmaxf(tile_max, s_scores[lane * 16 + j]);
                }
                float new_max = fmaxf(old_max, tile_max);
                float corr = exp_approx(old_max - new_max);
                float tile_sum = 0.0f;
                for (int j = 0; j < 16; j++) {
                    float p = exp_approx(s_scores[lane * 16 + j] - new_max);
                    s_scores[lane * 16 + j] = p;
                    tile_sum += p;
                }
                s_m[lane]    = new_max;
                s_l[lane]    = s_l[lane] * corr + tile_sum;
                s_corr[lane] = corr;
            }
        }
        __syncthreads();

        for (int i = tid; i < (Q_ROWS * 16) / 2; i += 128) {
            int idx = i * 2;
            *reinterpret_cast<__half2*>(s_p + idx) =
                __floats2half2_rn(s_scores[idx], s_scores[idx + 1]);
        }
        {
            float corr[M_TILES][2];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                corr[mt][0] = s_corr[mt * 16 + r0];
                corr[mt][1] = s_corr[mt * 16 + r1];
            }
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                #pragma unroll
                for (int i = 0; i < MAX_CHUNKS; i++) {
                    oacc0[mt][i][0] *= corr[mt][0];
                    oacc0[mt][i][1] *= corr[mt][0];
                    oacc0[mt][i][2] *= corr[mt][1];
                    oacc0[mt][i][3] *= corr[mt][1];
                    oacc1[mt][i][0] *= corr[mt][0];
                    oacc1[mt][i][1] *= corr[mt][0];
                    oacc1[mt][i][2] *= corr[mt][1];
                    oacc1[mt][i][3] *= corr[mt][1];
                }
            }
        }
        __syncthreads();

        int dv_per_warp;
        int dv_start;
        int dv_chunks_per_warp;
        if (d_v >= 64) {
            dv_per_warp        = d_v / 4;
            dv_start           = warp_id * dv_per_warp;
            dv_chunks_per_warp = dv_per_warp / 16;
        } else {
            dv_per_warp        = d_v;
            dv_start           = 0;
            dv_chunks_per_warp = (warp_id == 0) ? (d_v / 16) : 0;
        }

        if (dv_chunks_per_warp > 0) {
        issue_v_chunk_warp(kv_start, dv_start, 0);
        cp_async_commit();

        for (int dvc = 0; dvc < dv_chunks_per_warp; dvc++) {
            int dv = dv_start + dvc * 16;
            int cur_buf = dvc & 1;

            if (dvc + 1 < dv_chunks_per_warp) {
                int next_buf = (dvc + 1) & 1;
                issue_v_chunk_warp(kv_start, dv + 16, next_buf);
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_all();
            }
            __syncwarp();

            cast_v_chunk_warp(cur_buf);
            __syncwarp();

            uint32_t reg_p[M_TILES][4];
            uint32_t reg_v0[2];
            uint32_t reg_v1[2];
            int v_slot = (warp_id * 2 + cur_buf) * 256;
            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                ldmatrix_m16n8_x4_b16(reg_p[mt],
                    s_p + (mt * 16 + smem_row_a) * 16 + smem_col_a);
            }
            ldmatrix_m16n8_x2_b16(reg_v0, s_vchunk + v_slot +  b_col      * 16 + b_khalf);
            ldmatrix_m16n8_x2_b16(reg_v1, s_vchunk + v_slot + (b_col + 8) * 16 + b_khalf);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                mma_m16n8k16_fp32_fp16_fp16_fp32(oacc0[mt][dvc], reg_p[mt], reg_v0, oacc0[mt][dvc]);
                mma_m16n8k16_fp32_fp16_fp16_fp32(oacc1[mt][dvc], reg_p[mt], reg_v1, oacc1[mt][dvc]);
            }
        }
        }

        if (warp_id == 0 && kv_start + 16 < seq_len_k) {
            issue_k_chunk(kv_start + 16, 0, 0);
            cp_async_commit();
        }
        __syncthreads();
    }

#ifdef TENSORAX_PROFILE
    if (log) t_loop_end = clock64();
#endif
    {
        int dv_per_warp_o, dv_start_o, dv_chunks_per_warp_o;
        if (d_v >= 64) {
            dv_per_warp_o        = d_v / 4;
            dv_start_o           = warp_id * dv_per_warp_o;
            dv_chunks_per_warp_o = dv_per_warp_o / 16;
        } else {
            dv_per_warp_o        = d_v;
            dv_start_o           = 0;
            dv_chunks_per_warp_o = (warp_id == 0) ? (d_v / 16) : 0;
        }
        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++) {
            int row_off = mt * 16;
            int q0 = q_start + row_off + r0;
            int q1 = q_start + row_off + r1;
            bool ok0 = (q0 < seq_len_q);
            bool ok1 = (q1 < seq_len_q);
            float inv_l0 = ok0 ? (1.0f / s_l[row_off + r0]) : 0.0f;
            float inv_l1 = ok1 ? (1.0f / s_l[row_off + r1]) : 0.0f;
            for (int dvc = 0; dvc < dv_chunks_per_warp_o; dvc++) {
                int dv = dv_start_o + dvc * 16;
                if (ok0) {
                    out_base[q0 * d_v + dv + c0]         = oacc0[mt][dvc][0] * inv_l0;
                    out_base[q0 * d_v + dv + c0 + 1]     = oacc0[mt][dvc][1] * inv_l0;
                    out_base[q0 * d_v + dv + c0 + 8]     = oacc1[mt][dvc][0] * inv_l0;
                    out_base[q0 * d_v + dv + c0 + 8 + 1] = oacc1[mt][dvc][1] * inv_l0;
                }
                if (ok1) {
                    out_base[q1 * d_v + dv + c0]         = oacc0[mt][dvc][2] * inv_l1;
                    out_base[q1 * d_v + dv + c0 + 1]     = oacc0[mt][dvc][3] * inv_l1;
                    out_base[q1 * d_v + dv + c0 + 8]     = oacc1[mt][dvc][2] * inv_l1;
                    out_base[q1 * d_v + dv + c0 + 8 + 1] = oacc1[mt][dvc][3] * inv_l1;
                }
            }
        }
    }

#ifdef TENSORAX_PROFILE
    if (log) {
        long long t_end = clock64();
        prof_buf[0] = 0;
        prof_buf[1] = t_setup_end - t_entry;
        prof_buf[2] = prof_buf[1] + (t_qload_end - t_setup_end);
        prof_buf[3] = prof_buf[2] + (t_loop_end - t_qload_end);
        prof_buf[4] = prof_buf[3] + (t_end - t_loop_end);
    }
#endif
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
    int bc,
    long long* prof_buf
) {
    TX_TICK(prof_buf, 0);
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
    TX_TICK(prof_buf, 1);
    __syncthreads();

    bool first_kv = true;
    for (int kv_start = 0; kv_start < seq_len_k; kv_start += bc) {
        int kv_end = min(kv_start + bc, seq_len_k);
        int kv_len = kv_end - kv_start;

        for (int i = tid; i < kv_len * d_k; i += blockDim.x) {
            int ki = i / d_k;
            int d = i % d_k;
            s_k[ki * d_k + d] = k_ptr[(kv_start + ki) * d_k + d];
        }
        if (first_kv) TX_TICK(prof_buf, 2);
        for (int i = tid; i < kv_len * d_v; i += blockDim.x) {
            int vi = i / d_v;
            int d = i % d_v;
            s_v[vi * d_v + d] = v_ptr[(kv_start + vi) * d_v + d];
        }
        if (first_kv) TX_TICK(prof_buf, 3);
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
            if (first_kv && qi == 0) TX_TICK(prof_buf, 4);

            float correction = expf(row_max - new_max);
            float new_sum = row_sum * correction;

            for (int d = 0; d < d_v; d++) {
                s_o[qi * d_v + d] *= correction;
            }
            if (first_kv && qi == 0) TX_TICK(prof_buf, 5);

            for (int kj = 0; kj < kv_len; kj++) {
                float p = expf(s_scores[qi * bc + kj] - new_max);
                new_sum += p;
                for (int d = 0; d < d_v; d++) {
                    s_o[qi * d_v + d] += p * s_v[kj * d_v + d];
                }
            }
            if (first_kv && qi == 0) TX_TICK(prof_buf, 6);

            s_m[qi] = new_max;
            s_l[qi] = new_sum;
        }
        if (first_kv) TX_TICK(prof_buf, 7);
        first_kv = false;
        __syncthreads();
    }
    TX_TICK(prof_buf, 8);

    for (int i = tid; i < q_len * d_v; i += blockDim.x) {
        int qi = i / d_v;
        int d = i % d_v;
        out_base[(q_start + qi) * d_v + d] = s_o[qi * d_v + d] / s_l[qi];
    }
    TX_TICK(prof_buf, 9);
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
    int bc,
    long long* prof_buf
) {
    TX_TICK(prof_buf, 0);
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
    TX_TICK(prof_buf, 1);
    __syncthreads();

    bool first_kv = true;
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
        if (first_kv) TX_TICK(prof_buf, 2);
        block_load_f32_vec4(
            v_ptr + kv_start * d_v,
            s_v,
            kv_len * d_v,
            tid,
            blockDim.x
        );
        if (first_kv) TX_TICK(prof_buf, 3);
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
            if (first_kv && qi == 0 && lane == 0) TX_TICK(prof_buf, 4);

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
            if (first_kv && qi == 0 && lane == 0) TX_TICK(prof_buf, 5);

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
            if (first_kv && qi == 0 && lane == 0) TX_TICK(prof_buf, 6);

            if (lane == 0) {
                s_m[qi] = used_max;
                s_l[qi] = new_sum;
            }
        }
        if (first_kv) TX_TICK(prof_buf, 7);
        first_kv = false;
        __syncthreads();
    }
    TX_TICK(prof_buf, 8);

    for (int i = tid; i < q_len * d_v; i += blockDim.x) {
        int qi = i / d_v;
        int d = i % d_v;
        out_base[(q_start + qi) * d_v + d] = s_o[qi * d_v + d] / s_l[qi];
    }
    TX_TICK(prof_buf, 9);
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
    TENSORAX_NVTX_RANGE("sdpa.naive");
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));

    dim3 block(16, 16);
    dim3 grid(
        CEIL_DIV(d_v, block.x),
        CEIL_DIV(seq_len_q, block.y),
        batch_size * num_heads
    );

    sdpa_kernel_naive<<<grid, block>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, nullptr
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
    TENSORAX_NVTX_RANGE("sdpa.tiled");
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
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, tile_k, nullptr
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
    TENSORAX_NVTX_RANGE("sdpa.flash");
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
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, br, bc, nullptr
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
    TENSORAX_NVTX_RANGE("sdpa.flash_optimized");
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
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, br, bc, nullptr
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<long long> sdpa_optimized_flash_profile_sections_cuda(
    const float* Q, const float* K, const float* V,
    const float* mask, float* out,
    int64_t batch_size, int64_t num_heads,
    int64_t seq_len_q, int64_t seq_len_k,
    int64_t d_k, int64_t d_v
) {
    TENSORAX_NVTX_RANGE("sdpa.flash_optimized.profile");
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    int br, bc;
    cuda::compute_flash_tiles(d_k, d_v, br, bc);

    int threads = 128;
    dim3 block(threads);
    dim3 grid(1, CEIL_DIV(seq_len_q, br), batch_size * num_heads);
    size_t smem_size = ((size_t)br * d_k + (size_t)bc * d_k + (size_t)bc * d_v
                      + (size_t)br * bc + (size_t)br * d_v
                      + (size_t)br + (size_t)br) * sizeof(float);

    long long* d_buf = prof::alloc_buf();
    sdpa_kernel_flash_optimized<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, br, bc, d_buf
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return prof::read_buf(d_buf);
}


__global__ void sdpa_kernel_mma_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int d_k,
    int d_v,
    float scale
) {
    constexpr int NUM_WARPS     = 4;
    constexpr int ROWS_PER_WARP = 16;
    constexpr int Q_ROWS        = NUM_WARPS * ROWS_PER_WARP;

    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_start = blockIdx.y * Q_ROWS;
    if (b >= batch_size || h >= num_heads || q_start >= seq_len_q) return;

    int tid       = threadIdx.x;
    int warp_id   = tid >> 5;
    int lane      = tid & 31;
    int q_row_off = warp_id * ROWS_PER_WARP;
    int qkv_base  = b * num_heads + h;
    const __half* q_base = Q + qkv_base * seq_len_q * d_k;
    const __half* k_ptr  = K + qkv_base * seq_len_k * d_k;
    const __half* v_ptr  = V + qkv_base * seq_len_k * d_v;
    float* out_base      = out + qkv_base * seq_len_q * d_v;

    extern __shared__ __half mma_smem[];
    __half* s_q       = mma_smem;
    __half* s_kchunk  = s_q + Q_ROWS * d_k;
    __half* s_vstage  = s_kchunk + 2 * 16 * 16;
    __half* s_vchunk  = s_vstage + 2 * 16 * 16;

    int r0         = lane / 4;
    int r1         = r0 + 8;
    int c0         = (lane % 4) * 2;
    int smem_row_a = lane % 16;
    int smem_col_a = (lane / 16) * 8;
    int b_col      = lane & 7;
    int b_khalf    = ((lane >> 3) & 1) * 8;

    int total_q_ops = (Q_ROWS * d_k) / 8;
    for (int i = tid; i < total_q_ops; i += 128) {
        int idx = i * 8;
        int row = idx / d_k;
        int col = idx % d_k;
        bool in_range = (q_start + row) < seq_len_q;
        float4* dst = reinterpret_cast<float4*>(s_q + row * d_k + col);
        if (in_range) {
            *dst = *reinterpret_cast<const float4*>(q_base + (q_start + row) * d_k + col);
        } else {
            *dst = float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    constexpr int MAX_CHUNKS = 32;
    float oacc[MAX_CHUNKS][2][4];
    #pragma unroll
    for (int i = 0; i < MAX_CHUNKS; i++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            oacc[i][n][0] = oacc[i][n][1] = oacc[i][n][2] = oacc[i][n][3] = 0.0f;
        }
    }
    __syncthreads();

    int dk_chunks = d_k / 16;
    int dv_chunks = d_v / 16;

    auto issue_k_chunk = [&] (int kv_start, int dk, int buf) {
        int row = lane / 2;
        int col = (lane % 2) * 8;
        cp_async_16B(s_kchunk + buf * 256 + row * 16 + col,
                     k_ptr + (kv_start + row) * d_k + dk + col);
    };
    auto issue_v_chunk = [&] (int kv_start, int dv, int buf) {
        int k = lane / 2;
        int n = (lane % 2) * 8;
        cp_async_16B(s_vstage + buf * 256 + k * 16 + n,
                     v_ptr + (kv_start + k) * d_v + dv + n);
    };
    auto transpose_v_chunk = [&] (int buf) {
        int k = lane / 2;
        int n0 = (lane % 2) * 8;
        const __half* src = s_vstage + buf * 256 + k * 16 + n0;
        __half* dst = s_vchunk + buf * 256;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            dst[(n0 + i) * 16 + k] = src[i];
        }
    };

    if (warp_id == 0) {
        issue_k_chunk(0, 0, 0);
        cp_async_commit();
    }

    float m_r0 = -FLT_MAX, m_r1 = -FLT_MAX;
    float l_r0 = 0.0f,     l_r1 = 0.0f;

    for (int kv_start = 0; kv_start < seq_len_k; kv_start += 16) {
        float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};

        for (int dk_chunk = 0; dk_chunk < dk_chunks; dk_chunk++) {
            int dk = dk_chunk * 16;
            int cur_buf = dk_chunk & 1;

            if (warp_id == 0) {
                if (dk_chunk + 1 < dk_chunks) {
                    int next_buf = (dk_chunk + 1) & 1;
                    issue_k_chunk(kv_start, dk + 16, next_buf);
                    cp_async_commit();
                    cp_async_wait_group<1>();
                } else {
                    cp_async_wait_all();
                }
            }
            __syncthreads();

            uint32_t reg_q[4];
            uint32_t reg_k0[2];
            uint32_t reg_k1[2];
            ldmatrix_m16n8_x4_b16(reg_q,
                s_q + (q_row_off + smem_row_a) * d_k + dk + smem_col_a);
            ldmatrix_m16n8_x2_b16(reg_k0, s_kchunk + cur_buf * 256 +  b_col      * 16 + b_khalf);
            ldmatrix_m16n8_x2_b16(reg_k1, s_kchunk + cur_buf * 256 + (b_col + 8) * 16 + b_khalf);
            mma_m16n8k16_fp32_fp16_fp16_fp32(acc[0], reg_q, reg_k0, acc[0]);
            mma_m16n8k16_fp32_fp16_fp16_fp32(acc[1], reg_q, reg_k1, acc[1]);
        }

        #pragma unroll
        for (int n = 0; n < 2; n++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[n][i] *= scale;
        }

        float lane_max_r0 = fmaxf(fmaxf(acc[0][0], acc[0][1]),
                                  fmaxf(acc[1][0], acc[1][1]));
        float lane_max_r1 = fmaxf(fmaxf(acc[0][2], acc[0][3]),
                                  fmaxf(acc[1][2], acc[1][3]));
        lane_max_r0 = fmaxf(lane_max_r0, __shfl_xor_sync(FULL_WARP_MASK, lane_max_r0, 1));
        lane_max_r0 = fmaxf(lane_max_r0, __shfl_xor_sync(FULL_WARP_MASK, lane_max_r0, 2));
        lane_max_r1 = fmaxf(lane_max_r1, __shfl_xor_sync(FULL_WARP_MASK, lane_max_r1, 1));
        lane_max_r1 = fmaxf(lane_max_r1, __shfl_xor_sync(FULL_WARP_MASK, lane_max_r1, 2));

        bool update_r0 = (lane_max_r0 - m_r0) > SOFTMAX_MAX_UPDATE_THRESHOLD;
        bool update_r1 = (lane_max_r1 - m_r1) > SOFTMAX_MAX_UPDATE_THRESHOLD;
        float new_m_r0 = update_r0 ? lane_max_r0 : m_r0;
        float new_m_r1 = update_r1 ? lane_max_r1 : m_r1;
        float corr_r0  = update_r0 ? exp_approx(m_r0 - new_m_r0) : 1.0f;
        float corr_r1  = update_r1 ? exp_approx(m_r1 - new_m_r1) : 1.0f;

        acc[0][0] = exp_approx(acc[0][0] - new_m_r0);
        acc[0][1] = exp_approx(acc[0][1] - new_m_r0);
        acc[1][0] = exp_approx(acc[1][0] - new_m_r0);
        acc[1][1] = exp_approx(acc[1][1] - new_m_r0);
        acc[0][2] = exp_approx(acc[0][2] - new_m_r1);
        acc[0][3] = exp_approx(acc[0][3] - new_m_r1);
        acc[1][2] = exp_approx(acc[1][2] - new_m_r1);
        acc[1][3] = exp_approx(acc[1][3] - new_m_r1);

        float sum_r0 = acc[0][0] + acc[0][1] + acc[1][0] + acc[1][1];
        float sum_r1 = acc[0][2] + acc[0][3] + acc[1][2] + acc[1][3];
        sum_r0 += __shfl_xor_sync(FULL_WARP_MASK, sum_r0, 1);
        sum_r0 += __shfl_xor_sync(FULL_WARP_MASK, sum_r0, 2);
        sum_r1 += __shfl_xor_sync(FULL_WARP_MASK, sum_r1, 1);
        sum_r1 += __shfl_xor_sync(FULL_WARP_MASK, sum_r1, 2);

        m_r0 = new_m_r0;
        m_r1 = new_m_r1;
        l_r0 = l_r0 * corr_r0 + sum_r0;
        l_r1 = l_r1 * corr_r1 + sum_r1;

        if (update_r0) {
            #pragma unroll
            for (int i = 0; i < MAX_CHUNKS; i++) {
                oacc[i][0][0] *= corr_r0;
                oacc[i][0][1] *= corr_r0;
                oacc[i][1][0] *= corr_r0;
                oacc[i][1][1] *= corr_r0;
            }
        }
        if (update_r1) {
            #pragma unroll
            for (int i = 0; i < MAX_CHUNKS; i++) {
                oacc[i][0][2] *= corr_r1;
                oacc[i][0][3] *= corr_r1;
                oacc[i][1][2] *= corr_r1;
                oacc[i][1][3] *= corr_r1;
            }
        }

        uint32_t reg_p[4];
        {
            __half2 h0 = __floats2half2_rn(acc[0][0], acc[0][1]);
            __half2 h1 = __floats2half2_rn(acc[0][2], acc[0][3]);
            __half2 h2 = __floats2half2_rn(acc[1][0], acc[1][1]);
            __half2 h3 = __floats2half2_rn(acc[1][2], acc[1][3]);
            reg_p[0] = *reinterpret_cast<uint32_t*>(&h0);
            reg_p[1] = *reinterpret_cast<uint32_t*>(&h1);
            reg_p[2] = *reinterpret_cast<uint32_t*>(&h2);
            reg_p[3] = *reinterpret_cast<uint32_t*>(&h3);
        }

        __syncthreads();

        if (warp_id == 0) {
            issue_v_chunk(kv_start, 0, 0);
            cp_async_commit();
        }

        for (int dvc = 0; dvc < dv_chunks; dvc++) {
            int dv = dvc * 16;
            int cur_buf = dvc & 1;

            if (warp_id == 0) {
                if (dvc + 1 < dv_chunks) {
                    int next_buf = (dvc + 1) & 1;
                    issue_v_chunk(kv_start, dv + 16, next_buf);
                    cp_async_commit();
                    cp_async_wait_group<1>();
                } else {
                    cp_async_wait_all();
                }
            }
            __syncthreads();

            if (warp_id == 0) {
                transpose_v_chunk(cur_buf);
            }
            __syncthreads();

            uint32_t reg_v0[2];
            uint32_t reg_v1[2];
            ldmatrix_m16n8_x2_b16(reg_v0, s_vchunk + cur_buf * 256 +  b_col      * 16 + b_khalf);
            ldmatrix_m16n8_x2_b16(reg_v1, s_vchunk + cur_buf * 256 + (b_col + 8) * 16 + b_khalf);

            mma_m16n8k16_fp32_fp16_fp16_fp32(oacc[dvc][0], reg_p, reg_v0, oacc[dvc][0]);
            mma_m16n8k16_fp32_fp16_fp16_fp32(oacc[dvc][1], reg_p, reg_v1, oacc[dvc][1]);
        }

        if (warp_id == 0 && kv_start + 16 < seq_len_k) {
            issue_k_chunk(kv_start + 16, 0, 0);
            cp_async_commit();
        }
        __syncthreads();
    }

    {
        int q0 = q_start + q_row_off + r0;
        int q1 = q_start + q_row_off + r1;
        bool ok0 = (q0 < seq_len_q);
        bool ok1 = (q1 < seq_len_q);
        float inv_l0 = ok0 ? (1.0f / l_r0) : 0.0f;
        float inv_l1 = ok1 ? (1.0f / l_r1) : 0.0f;
        for (int dvc = 0; dvc < dv_chunks; dvc++) {
            int dv = dvc * 16;
            if (ok0) {
                out_base[q0 * d_v + dv + c0]         = oacc[dvc][0][0] * inv_l0;
                out_base[q0 * d_v + dv + c0 + 1]     = oacc[dvc][0][1] * inv_l0;
                out_base[q0 * d_v + dv + c0 + 8]     = oacc[dvc][1][0] * inv_l0;
                out_base[q0 * d_v + dv + c0 + 8 + 1] = oacc[dvc][1][1] * inv_l0;
            }
            if (ok1) {
                out_base[q1 * d_v + dv + c0]         = oacc[dvc][0][2] * inv_l1;
                out_base[q1 * d_v + dv + c0 + 1]     = oacc[dvc][0][3] * inv_l1;
                out_base[q1 * d_v + dv + c0 + 8]     = oacc[dvc][1][2] * inv_l1;
                out_base[q1 * d_v + dv + c0 + 8 + 1] = oacc[dvc][1][3] * inv_l1;
            }
        }
    }
}

static size_t mma_fp16_smem_bytes(int64_t d_k, int64_t d_v) {
    constexpr int Q_ROWS = 64;
    (void)d_v;
    return (size_t)Q_ROWS * (size_t)d_k * sizeof(__half)
         + (size_t)2 * 16 * 16 * sizeof(__half)
         + (size_t)2 * 16 * 16 * sizeof(__half)
         + (size_t)2 * 16 * 16 * sizeof(__half);
}

void sdpa_mma_fp16_cuda(
    const void* Q_h,
    const void* K_h,
    const void* V_h,
    float* out,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t d_k,
    int64_t d_v
) {
    TENSORAX_NVTX_RANGE("sdpa.mma.fp16");
    if (d_k % 16 != 0 || d_v % 16 != 0 || seq_len_q % 16 != 0 || seq_len_k % 16 != 0) {
        throw std::runtime_error("sdpa_mma_fp16: d_k, d_v, seq_len_q, seq_len_k must be multiples of 16");
    }

    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    dim3 block(128);
    dim3 grid(1, CEIL_DIV(seq_len_q, 64), batch_size * num_heads);

    size_t smem_size = mma_fp16_smem_bytes(d_k, d_v);
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(sdpa_kernel_mma_fp16,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_size);
    }

    sdpa_kernel_mma_fp16<<<grid, block, smem_size>>>(
        static_cast<const __half*>(Q_h),
        static_cast<const __half*>(K_h),
        static_cast<const __half*>(V_h),
        out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale
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

void cast_f32_to_f16_cuda(const float* in, void* out, int64_t size) {
    int threads = 256;
    int total4 = (int)((size + 3) / 4);
    int blocks = (total4 + threads - 1) / threads;
    cast_f32_to_f16<<<blocks, threads>>>(in, static_cast<__half*>(out), (int)size);
    CUDA_CHECK(cudaGetLastError());
}

static size_t mma_smem_bytes(int64_t d_k, int64_t d_v) {
    constexpr int Q_ROWS = 32;
    (void)d_v;
    return (size_t)Q_ROWS * (size_t)d_k * sizeof(__half)
         + (size_t)Q_ROWS * 16 * sizeof(__half)
         + (size_t)2 * 16 * 16 * sizeof(float)
         + (size_t)2 * 16 * 16 * sizeof(__half)
         + (size_t)4 * 2 * 16 * 16 * sizeof(float)
         + (size_t)4 * 2 * 16 * 16 * sizeof(__half)
         + (size_t)Q_ROWS * 16 * sizeof(float)
         + (size_t)3 * Q_ROWS * sizeof(float);
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
    TENSORAX_NVTX_RANGE("sdpa.mma");
    if (mask || d_k % 16 != 0 || d_v % 16 != 0 || seq_len_q % 16 != 0 || seq_len_k % 16 != 0) {
        sdpa_tiled_cuda(Q, K, V, mask, out,
                        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v);
        return;
    }

    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    dim3 block(128);
    dim3 grid(1, CEIL_DIV(seq_len_q, 16), batch_size * num_heads);

    size_t smem_size = mma_smem_bytes(d_k, d_v);
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(sdpa_kernel_mma,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_size);
    }

    sdpa_kernel_mma<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, nullptr
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<long long> sdpa_mma_profile_sections_cuda(
    const float* Q, const float* K, const float* V,
    const float* mask, float* out,
    int64_t batch_size, int64_t num_heads,
    int64_t seq_len_q, int64_t seq_len_k,
    int64_t d_k, int64_t d_v
) {
    TENSORAX_NVTX_RANGE("sdpa.mma.profile");
    if (mask || d_k % 16 != 0 || d_v % 16 != 0 || seq_len_q % 16 != 0 || seq_len_k % 16 != 0) {
        return sdpa_tiled_profile_sections_cuda(Q, K, V, mask, out,
                                               batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    dim3 block(128);
    dim3 grid(1, CEIL_DIV(seq_len_q, 16), batch_size * num_heads);
    size_t smem_size = mma_smem_bytes(d_k, d_v);
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(sdpa_kernel_mma,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_size);
    }

    long long* d_buf = prof::alloc_buf();
    sdpa_kernel_mma<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, d_buf
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return prof::read_buf(d_buf);
}

std::vector<long long> sdpa_naive_profile_sections_cuda(
    const float* Q, const float* K, const float* V,
    const float* mask, float* out,
    int64_t batch_size, int64_t num_heads,
    int64_t seq_len_q, int64_t seq_len_k,
    int64_t d_k, int64_t d_v
) {
    TENSORAX_NVTX_RANGE("sdpa.naive.profile");
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(d_v, block.x), CEIL_DIV(seq_len_q, block.y), batch_size * num_heads);
    long long* d_buf = prof::alloc_buf();
    sdpa_kernel_naive<<<grid, block>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, d_buf
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return prof::read_buf(d_buf);
}

std::vector<long long> sdpa_tiled_profile_sections_cuda(
    const float* Q, const float* K, const float* V,
    const float* mask, float* out,
    int64_t batch_size, int64_t num_heads,
    int64_t seq_len_q, int64_t seq_len_k,
    int64_t d_k, int64_t d_v
) {
    TENSORAX_NVTX_RANGE("sdpa.tiled.profile");
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    int tile_k = cuda::compute_tiled_tile_k(d_k, d_v);
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(d_v, block.x), CEIL_DIV(seq_len_q, block.y), batch_size * num_heads);
    size_t smem_size = ((size_t)tile_k * d_k + (size_t)tile_k * d_v) * sizeof(float);
    long long* d_buf = prof::alloc_buf();
    sdpa_kernel_tiled<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, tile_k, d_buf
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return prof::read_buf(d_buf);
}

std::vector<long long> sdpa_flash_profile_sections_cuda(
    const float* Q, const float* K, const float* V,
    const float* mask, float* out,
    int64_t batch_size, int64_t num_heads,
    int64_t seq_len_q, int64_t seq_len_k,
    int64_t d_k, int64_t d_v
) {
    TENSORAX_NVTX_RANGE("sdpa.flash.profile");
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    int br, bc;
    cuda::compute_flash_tiles(d_k, d_v, br, bc);
    int threads = 128;
    dim3 block(threads);
    dim3 grid(1, CEIL_DIV(seq_len_q, br), batch_size * num_heads);
    size_t smem_size = ((size_t)br * d_k + (size_t)bc * d_k + (size_t)bc * d_v
                      + (size_t)br * bc + (size_t)br * d_v
                      + (size_t)br + (size_t)br) * sizeof(float);
    long long* d_buf = prof::alloc_buf();
    sdpa_kernel_flash<<<grid, block, smem_size>>>(
        Q, K, V, mask, out,
        batch_size, num_heads, seq_len_q, seq_len_k, d_k, d_v, scale, br, bc, d_buf
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return prof::read_buf(d_buf);
}


} // namespace tensorax
