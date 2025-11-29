#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"

namespace tensora {
namespace cuda {

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void broadcasting_add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* out, int64_t size_a, int64_t size_b) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;  // row index (0 to size_b-1)
    int64_t j = blockIdx.y * blockDim.y + threadIdx.y;  // col index (0 to size_a-1)
    
    if (i < size_b && j < size_a) {
        out[i * size_a + j] = a[j] + b[i];
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void relu_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void sqrt_kernel(const float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx]);
    }
}

} // namespace cuda

// Host-side wrappers
void add_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::add_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void broadcasting_add_cuda(const float* a, const float* b, float* out, int64_t size_a, int64_t size_b) {
    dim3 block(16, 16);  // 2D block for 2D operation
    dim3 grid((size_b + block.x - 1) / block.x, (size_a + block.y - 1) / block.y);
    cuda::broadcasting_add_kernel<<<grid, block>>>(a, b, out, size_a, size_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sub_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::sub_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mul_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::mul_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void div_cuda(const float* a, const float* b, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::div_kernel<<<grid, block>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void relu_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::relu_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sigmoid_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::sigmoid_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tanh_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::tanh_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sqrt_cuda(const float* in, float* out, int64_t size) {
    dim3 grid = cuda::get_grid_size(size);
    dim3 block(cuda::BLOCK_SIZE);
    cuda::sqrt_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace tensora
