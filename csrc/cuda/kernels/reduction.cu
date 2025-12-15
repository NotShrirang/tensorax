#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"
#include "reduction.cuh"

namespace tensorax {

__global__ void reduce_sum_kernel(const float* input, float* output, int64_t output_size, int64_t reduce_size, 
                                  int64_t inner_size, int64_t stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        int64_t outer_idx = idx / inner_size;
        int64_t inner_idx = idx % inner_size;
        
        float sum = 0.0f;
        for (int64_t i = 0; i < reduce_size; ++i) {
            int64_t input_idx = outer_idx * stride * reduce_size + i * stride + inner_idx;
            sum += input[input_idx];
        }
        output[idx] = sum;
    }
}

__global__ void reduce_mean_kernel(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                                   int64_t inner_size, int64_t stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        int64_t outer_idx = idx / inner_size;
        int64_t inner_idx = idx % inner_size;
        
        float sum = 0.0f;
        for (int64_t i = 0; i < reduce_size; ++i) {
            int64_t input_idx = outer_idx * stride * reduce_size + i * stride + inner_idx;
            sum += input[input_idx];
        }
        output[idx] = sum / static_cast<float>(reduce_size);
    }
}

__global__ void reduce_max_kernel(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                                  int64_t inner_size, int64_t stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        int64_t outer_idx = idx / inner_size;
        int64_t inner_idx = idx % inner_size;
        
        float max_val = -INFINITY;
        for (int64_t i = 0; i < reduce_size; ++i) {
            int64_t input_idx = outer_idx * stride * reduce_size + i * stride + inner_idx;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        output[idx] = max_val;
    }
}

void reduce_sum_cuda(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                     int64_t inner_size, int64_t stride) {
    dim3 grid = cuda::get_grid_size(output_size);
    dim3 block(cuda::BLOCK_SIZE);
    reduce_sum_kernel<<<grid, block>>>(input, output, output_size, reduce_size, inner_size, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void reduce_mean_cuda(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                      int64_t inner_size, int64_t stride) {
    dim3 grid = cuda::get_grid_size(output_size);
    dim3 block(cuda::BLOCK_SIZE);
    reduce_mean_kernel<<<grid, block>>>(input, output, output_size, reduce_size, inner_size, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void reduce_max_cuda(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                     int64_t inner_size, int64_t stride) {
    dim3 grid = cuda::get_grid_size(output_size);
    dim3 block(cuda::BLOCK_SIZE);
    reduce_max_kernel<<<grid, block>>>(input, output, output_size, reduce_size, inner_size, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace tensoraxx
