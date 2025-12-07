#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"

namespace tensora {

__global__ void reduce_sum_kernel(const float* input, float* output, 
                                   int64_t size, int64_t reduce_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float sum = 0.0f;
        int64_t base = idx * reduce_size;
        for (int64_t i = 0; i < reduce_size; ++i) {
            sum += input[base + i];
        }
        output[idx] = sum;
    }
}

__global__ void reduce_max_kernel(const float* input, float* output,
                                   int64_t size, int64_t reduce_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float max_val = -INFINITY;
        int64_t base = idx * reduce_size;
        for (int64_t i = 0; i < reduce_size; ++i) {
            max_val = fmaxf(max_val, input[base + i]);
        }
        output[idx] = max_val;
    }
}

} // namespace tensora
