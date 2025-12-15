#include "cuda_utils.cuh"
#include "../tensor_ops.h"
#include <cuda_runtime.h>

namespace tensorax {

bool cuda_is_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

} // namespace tensoraxx
