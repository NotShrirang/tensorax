#pragma once

#include <cstdint>

namespace tensorax {

// Kernel declarations
__global__ void reduce_sum_kernel(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                                  int64_t inner_size, int64_t stride);
__global__ void divide_with_type_casting_kernel(float* data, int64_t size, int64_t divisor);
__global__ void reduce_max_kernel(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                                  int64_t inner_size, int64_t stride);

// Wrapper function declarations
void reduce_sum_cuda(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                     int64_t inner_size, int64_t stride);
void reduce_mean_cuda(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                      int64_t inner_size, int64_t stride);
void reduce_max_cuda(const float* input, float* output, int64_t output_size, int64_t reduce_size,
                     int64_t inner_size, int64_t stride);

} // namespace tensoraxx
