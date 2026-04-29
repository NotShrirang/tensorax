#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#define TENSORAX_NVTX_RANGE(name) ::nvtx3::scoped_range _tx_nvtx_range_{(name)}

#ifdef TENSORAX_PROFILE
  #define TX_TICK(buf, idx)                                                   \
      do {                                                                    \
          if ((buf) != nullptr &&                                             \
              threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&     \
              blockIdx.x  == 0 && blockIdx.y  == 0 && blockIdx.z  == 0) {     \
              (buf)[(idx)] = clock64();                                       \
          }                                                                   \
      } while (0)
#else
  #define TX_TICK(buf, idx) ((void)0)
#endif

namespace tensorax {
namespace prof {

constexpr int MAX_TICKS = 32;

inline long long* alloc_buf() {
    long long* d_buf = nullptr;
    cudaMalloc(&d_buf, MAX_TICKS * sizeof(long long));
    cudaMemset(d_buf, 0, MAX_TICKS * sizeof(long long));
    return d_buf;
}

inline std::vector<long long> read_buf(long long* d_buf) {
    std::vector<long long> h(MAX_TICKS, 0);
    cudaMemcpy(h.data(), d_buf, MAX_TICKS * sizeof(long long),
               cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    return h;
}

}
}
