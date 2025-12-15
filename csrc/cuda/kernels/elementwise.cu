#include "../cuda/cuda_utils.cuh"
#include "../tensor_ops.h"
#include "reduction.cuh"
#include <vector>

namespace tensorax {
    __global__ void add_kernel(const float* a, const float* b, float* out, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            out[idx] = a[idx] + b[idx];
        }
    }

    __global__ void broadcasting_add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* out,
                                            const int64_t* stride_a, const int64_t* stride_b, const int64_t* stride_out,
                                            int64_t ndim, int64_t size_out) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size_out) return;

        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t remaining = idx;

        for (int64_t i = 0; i < ndim; ++i) {
            int64_t coord = remaining / stride_out[i];
            remaining = remaining % stride_out[i];
            
            idx_a += coord * stride_a[i];
            idx_b += coord * stride_b[i];
        }
        
        out[idx] = a[idx_a] + b[idx_b];
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

    __global__ void pow_kernel(const float* input, float* output, float power, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = powf(input[idx], power);
        }

    }

    __global__ void log_kernel(const float* input, float* output, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = logf(input[idx]);
        }
    }

    __global__ void exp_kernel(const float* input, float* output, int64_t size) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = expf(input[idx]);
        }
    }

    __global__ void transpose_kernel(const float* in, float* out, int64_t batch_size, int64_t rows, int64_t cols) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_size = batch_size * rows * cols;
        
        if (idx < total_size) {
            int64_t matrix_size = rows * cols;
            int64_t batch = idx / matrix_size;
            int64_t local_idx = idx % matrix_size;
            int64_t i = local_idx / cols;
            int64_t j = local_idx % cols;
            
            out[batch * matrix_size + j * rows + i] = in[idx];
        }
    }

    void add_cuda(const float* a, const float* b, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        add_kernel<<<grid, block>>>(a, b, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void broadcasting_add_cuda(const float* a, const float* b, float* out, 
                            const std::vector<int64_t> &shape_a, 
                            const std::vector<int64_t> &shape_b,
                            const std::vector<int64_t> &shape_out) {
        int64_t ndim_out = shape_out.size();
        int64_t ndim_a = shape_a.size();
        int64_t ndim_b = shape_b.size();

        std::vector<int64_t> stride_a_orig(ndim_a, 1);
        for (int64_t i = ndim_a - 2; i >= 0; --i) {
            stride_a_orig[i] = stride_a_orig[i + 1] * shape_a[i + 1];
        }
        
        std::vector<int64_t> stride_b_orig(ndim_b, 1);
        for (int64_t i = ndim_b - 2; i >= 0; --i) {
            stride_b_orig[i] = stride_b_orig[i + 1] * shape_b[i + 1];
        }

        std::vector<int64_t> stride_out(ndim_out, 1);
        for (int64_t i = ndim_out - 2; i >= 0; --i) {
            stride_out[i] = stride_out[i + 1] * shape_out[i + 1];
        }

        std::vector<int64_t> stride_a(ndim_out, 0);
        for (int64_t i = 0; i < ndim_out; ++i) {
            int64_t idx_a = i - (ndim_out - ndim_a);
            if (idx_a >= 0 && idx_a < ndim_a) {
                if (shape_a[idx_a] == shape_out[i]) {
                    stride_a[i] = stride_a_orig[idx_a];
                } else if (shape_a[idx_a] == 1) {
                    stride_a[i] = 0;
                }
            }
        }
        
        std::vector<int64_t> stride_b(ndim_out, 0);
        for (int64_t i = 0; i < ndim_out; ++i) {
            int64_t idx_b = i - (ndim_out - ndim_b);
            if (idx_b >= 0 && idx_b < ndim_b) {
                if (shape_b[idx_b] == shape_out[i]) {
                    stride_b[i] = stride_b_orig[idx_b];
                } else if (shape_b[idx_b] == 1) {
                    stride_b[i] = 0;
                }
            }
        }

        int64_t *d_stride_a, *d_stride_b, *d_stride_out;
        CUDA_CHECK(cudaMalloc(&d_stride_a, ndim_out * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_stride_b, ndim_out * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_stride_out, ndim_out * sizeof(int64_t)));
        
        CUDA_CHECK(cudaMemcpy(d_stride_a, stride_a.data(), ndim_out * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_stride_b, stride_b.data(), ndim_out * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_stride_out, stride_out.data(), ndim_out * sizeof(int64_t), cudaMemcpyHostToDevice));

        int64_t size_out = 1;
        for (auto dim : shape_out) {
            size_out *= dim;
        }

        dim3 grid = cuda::get_grid_size(size_out);
        dim3 block(cuda::BLOCK_SIZE);
        broadcasting_add_kernel<<<grid, block>>>(a, b, out, d_stride_a, d_stride_b, d_stride_out, ndim_out, size_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_stride_a));
        CUDA_CHECK(cudaFree(d_stride_b));
        CUDA_CHECK(cudaFree(d_stride_out));
    }

    void sub_cuda(const float* a, const float* b, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        sub_kernel<<<grid, block>>>(a, b, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void mul_cuda(const float* a, const float* b, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        mul_kernel<<<grid, block>>>(a, b, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void div_cuda(const float* a, const float* b, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        div_kernel<<<grid, block>>>(a, b, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void relu_cuda(const float* in, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        relu_kernel<<<grid, block>>>(in, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void sigmoid_cuda(const float* in, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        sigmoid_kernel<<<grid, block>>>(in, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void tanh_cuda(const float* in, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        tanh_kernel<<<grid, block>>>(in, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void sqrt_cuda(const float* in, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        sqrt_kernel<<<grid, block>>>(in, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void pow_cuda(const float* in, float* out, float power, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        pow_kernel<<<grid, block>>>(in, out, power, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void sum_cuda(const float* in, float* out, const std::vector<int64_t> &shape, int64_t dim) {
        if (dim == -1) {
            // Sum all elements
            int64_t total_size = 1;
            for (auto s : shape) {
                total_size *= s;
            }
            // Use reduction kernel: treat as 1 output element, reducing over all input
            reduce_sum_cuda(in, out, 1, total_size, 1, 1);
            return;
        }
        
        int64_t reduce_size = shape[dim];
        int64_t output_size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                output_size *= shape[i];
            }
        }
        
        // Calculate inner_size: product of dimensions after dim
        int64_t inner_size = 1;
        for (size_t i = dim + 1; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }
        
        // Stride is the size of elements to skip between consecutive elements along dim
        int64_t stride = inner_size;
        
        reduce_sum_cuda(in, out, output_size, reduce_size, inner_size, stride);
    }

    void mean_cuda(const float* in, float* out, const std::vector<int64_t> &shape, int64_t dim) {
        if (dim == -1) {
            // Mean of all elements
            int64_t total_size = 1;
            for (auto s : shape) {
                total_size *= s;
            }
            // Use reduction kernel: treat as 1 output element, reducing over all input
            reduce_mean_cuda(in, out, 1, total_size, 1, 1);
            return;
        }
        
        int64_t reduce_size = shape[dim];
        int64_t output_size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                output_size *= shape[i];
            }
        }
        
        // Calculate inner_size: product of dimensions after dim
        int64_t inner_size = 1;
        for (size_t i = dim + 1; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }
        
        // Stride is the size of elements to skip between consecutive elements along dim
        int64_t stride = inner_size;
        
        reduce_mean_cuda(in, out, output_size, reduce_size, inner_size, stride);
    }

    void log_cuda(const float* in, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        log_kernel<<<grid, block>>>(in, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void exp_cuda(const float* in, float* out, int64_t size) {
        dim3 grid = cuda::get_grid_size(size);
        dim3 block(cuda::BLOCK_SIZE);
        exp_kernel<<<grid, block>>>(in, out, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void transpose_cuda(const float* in, float* out, int64_t batch_size, int64_t rows, int64_t cols) {
        int64_t total_size = batch_size * rows * cols;
        dim3 grid = cuda::get_grid_size(total_size);
        dim3 block(cuda::BLOCK_SIZE);
        transpose_kernel<<<grid, block>>>(in, out, batch_size, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}// namespace tensorax