#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef TENSORAX_HOPPER

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "collective/fmha_fusion.hpp"
#include "device/device_universal.hpp"
#include "kernel/fmha_kernel_builder.hpp"
#include "kernel/fmha_options.hpp"

namespace tensorax {
namespace hopper {

using namespace cute;

// Cached workspace + LSE buffer — sized up on demand, never shrunk. Avoids per-call
// cudaMalloc against PyTorch's caching allocator, which causes large variance in
// benchmarks. Single-threaded usage assumed (Python GIL serializes kernel launches).
static void*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;
static float* g_lse_buffer     = nullptr;
static size_t g_lse_size       = 0;

static void* ensure_workspace(size_t bytes)
{
    if (bytes == 0) return nullptr;
    if (bytes > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaError_t e = cudaMalloc(&g_workspace, bytes);
        if (e != cudaSuccess) {
            g_workspace = nullptr;
            g_workspace_size = 0;
            throw std::runtime_error(
                std::string("sdpa_cute_fp16: workspace cudaMalloc failed: ") + cudaGetErrorString(e));
        }
        g_workspace_size = bytes;
    }
    return g_workspace;
}

static float* ensure_lse(size_t count)
{
    size_t bytes = count * sizeof(float);
    if (bytes > g_lse_size) {
        if (g_lse_buffer) cudaFree(g_lse_buffer);
        cudaError_t e = cudaMalloc(&g_lse_buffer, bytes);
        if (e != cudaSuccess) {
            g_lse_buffer = nullptr;
            g_lse_size = 0;
            throw std::runtime_error(
                std::string("sdpa_cute_fp16: cudaMalloc(LSE) failed: ") + cudaGetErrorString(e));
        }
        g_lse_size = bytes;
    }
    return g_lse_buffer;
}

template <int HEAD_DIM, class DispatchPolicy, class... ExtraOptions>
struct CuteSdpaFwd {
    using Element = cutlass::half_t;
    using ElementAccQK = float;
    using ElementAccPV = float;

    using TileShape = Shape<_128, _128, Int<HEAD_DIM>>;

    using StrideQ   = cute::tuple<int, _1, cute::tuple<int, int>>;
    using StrideK   = cute::tuple<int, _1, cute::tuple<int, int>>;
    using StrideV   = cute::tuple<int, _1, cute::tuple<int, int>>;
    using StrideO   = cute::tuple<int, _1, cute::tuple<int, int>>;
    using StrideLSE = cute::tuple<_1, cute::tuple<int, int>>;

    using ActiveFusion = cutlass::fmha::collective::DefaultFusion;

    using Operation = cutlass::device::Universal<
        typename cutlass::fmha::kernel::FmhaBuilder<
            Element, ElementAccQK, ElementAccPV,
            TileShape, StrideQ, StrideK, StrideV,
            ActiveFusion, DispatchPolicy,
            ExtraOptions...
        >::Kernel>;
};

template <class DispatchPolicy, class... ExtraOptions>
static void launch_d128(
    const void* Q, const void* K, const void* V,
    void* O, float* lse,
    int B, int H, int Sq, int Sk, int D,
    cudaStream_t stream)
{
    using R  = CuteSdpaFwd<128, DispatchPolicy, ExtraOptions...>;
    using Op = typename R::Operation;

    typename R::StrideQ   sQ   = make_stride(D, _1{}, make_stride(H * Sq * D, Sq * D));
    typename R::StrideK   sK   = make_stride(D, _1{}, make_stride(H * Sk * D, Sk * D));
    typename R::StrideV   sV   = make_stride(D, _1{}, make_stride(H * Sk * D, Sk * D));
    typename R::StrideO   sO   = make_stride(D, _1{}, make_stride(H * Sq * D, Sq * D));
    typename R::StrideLSE sLSE = make_stride(_1{}, make_stride(H * Sq, Sq));

    cutlass::KernelHardwareInfo hw_info{};
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    auto* Qe = reinterpret_cast<typename R::Element*>(const_cast<void*>(Q));
    auto* Ke = reinterpret_cast<typename R::Element*>(const_cast<void*>(K));
    auto* Ve = reinterpret_cast<typename R::Element*>(const_cast<void*>(V));
    auto* Oe = reinterpret_cast<typename R::Element*>(O);

    typename Op::Arguments args{
        cute::make_tuple(B, H, Sq, Sk, D),
        { Qe, sQ, Ke, sK, Ve, sV },
        { Oe, sO, lse, sLSE },
        hw_info
    };

    Op op;

    if (op.can_implement(args) != cutlass::Status::kSuccess) {
        throw std::runtime_error("sdpa_cute_fp16: can_implement returned non-success");
    }

    void* ws = ensure_workspace(Op::get_workspace_size(args));

    cutlass::Status st = op.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        throw std::runtime_error("sdpa_cute_fp16: initialize failed");
    }

    st = op.run(stream);
    if (st != cutlass::Status::kSuccess) {
        throw std::runtime_error("sdpa_cute_fp16: launch failed");
    }
}

}  // namespace hopper

namespace {

template <class DispatchPolicy, class... ExtraOptions>
void run_with_policy(
    const void* Q, const void* K, const void* V, void* O,
    int64_t B, int64_t H, int64_t Sq, int64_t Sk, int64_t D)
{
    if (D != 128) {
        throw std::runtime_error("sdpa_cute_fp16: only D=128 supported in v0");
    }
    if (Sq % 128 != 0 || Sk % 128 != 0) {
        throw std::runtime_error("sdpa_cute_fp16: Sq and Sk must be multiples of 128 (TileShape Q/KV = 128)");
    }

    float* lse = hopper::ensure_lse((size_t)(B * H * Sq));

    hopper::launch_d128<DispatchPolicy, ExtraOptions...>(
        Q, K, V, O, lse,
        (int)B, (int)H, (int)Sq, (int)Sk, (int)D,
        /*stream=*/0);
    // No internal cudaDeviceSynchronize() — caller (Python bench / next op) syncs.
}

}  // anonymous namespace

void sdpa_cute_fp16_cuda(
    const void* Q, const void* K, const void* V,
    void* O,
    int64_t B, int64_t H, int64_t Sq, int64_t Sk, int64_t D)
{
    run_with_policy<cutlass::gemm::KernelTmaWarpSpecializedCooperative>(
        Q, K, V, O, B, H, Sq, Sk, D);
}

void sdpa_cute_fp16_pingpong_cuda(
    const void* Q, const void* K, const void* V,
    void* O,
    int64_t B, int64_t H, int64_t Sq, int64_t Sk, int64_t D)
{
    run_with_policy<cutlass::gemm::KernelTmaWarpSpecializedPingpong>(
        Q, K, V, O, B, H, Sq, Sk, D);
}

void sdpa_cute_fp16_pp_q3_cuda(
    const void* Q, const void* K, const void* V,
    void* O,
    int64_t B, int64_t H, int64_t Sq, int64_t Sk, int64_t D)
{
    using Tag = cutlass::fmha::kernel::Tag;
    using StagesQ3 = cutlass::fmha::kernel::Option<Tag::kStagesQ, cute::Int<3>>;
    run_with_policy<cutlass::gemm::KernelTmaWarpSpecializedPingpong, StagesQ3>(
        Q, K, V, O, B, H, Sq, Sk, D);
}

// kNumMmaWarpGroups=3 is incompatible with BlockQO=128 (128/3=42 non-integer
// per-warpgroup tile, breaks epilogue TMA-store smem layout). Would need a
// different TileShape (M=96 or M=192) to enable.

}  // namespace tensorax

#else  // !TENSORAX_HOPPER

namespace tensorax {
void sdpa_cute_fp16_cuda(const void*, const void*, const void*, void*,
                         int64_t, int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("sdpa_cute_fp16: extension not built with TENSORAX_HOPPER");
}
void sdpa_cute_fp16_pingpong_cuda(const void*, const void*, const void*, void*,
                                  int64_t, int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("sdpa_cute_fp16_pingpong: extension not built with TENSORAX_HOPPER");
}
void sdpa_cute_fp16_pp_q3_cuda(const void*, const void*, const void*, void*,
                               int64_t, int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("sdpa_cute_fp16_pp_q3: extension not built with TENSORAX_HOPPER");
}
}  // namespace tensorax

#endif
