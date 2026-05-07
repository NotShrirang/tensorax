#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef TENSORAX_HOPPER

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace tensorax {
namespace hopper {

using namespace cute;

// Cached workspace — sized up on demand, never shrunk. Avoids per-call cudaMalloc
// against PyTorch's caching allocator, which causes large variance in benchmarks.
// Single-threaded usage assumed (Python GIL serializes kernel launches).
static void*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;

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
                std::string("matmul_cute_fp16: workspace cudaMalloc failed: ") + cudaGetErrorString(e));
        }
        g_workspace_size = bytes;
    }
    return g_workspace;
}

template <class TileShape_, class ClusterShape_, class KernelSched_, class EpilogueSched_>
struct CuteGemmFp16 {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;

    // PyTorch tensors are contiguous (M,K) and (K,N) and (M,N) — row-major.
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 16 / sizeof(ElementA);
    static constexpr int AlignmentB = 16 / sizeof(ElementB);
    static constexpr int AlignmentC = 16 / sizeof(ElementC);
    static constexpr int AlignmentD = 16 / sizeof(ElementD);

    using TileShape    = TileShape_;
    using ClusterShape = ClusterShape_;

    using KernelSchedule   = KernelSched_;
    using EpilogueSchedule = EpilogueSched_;
    using TileScheduler    = cutlass::gemm::PersistentScheduler;

    using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementScalar,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueSchedule,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileScheduler>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <class TileShape, class ClusterShape, class KernelSched, class EpilogueSched>
static void launch_gemm_fp16(
    const void* A, const void* B, void* C,
    int B_size, int M, int N, int K,
    cudaStream_t stream)
{
    using G = CuteGemmFp16<TileShape, ClusterShape, KernelSched, EpilogueSched>;
    using Gemm = typename G::Gemm;

    // Row-major batched strides.
    // A: (B, M, K) row-major → stride (K, 1, M*K)
    // B: (B, K, N) row-major → stride (N, 1, K*N)
    // C/D: (B, M, N) row-major → stride (N, 1, M*N)
    typename G::StrideA sA = cutlass::make_cute_packed_stride(
        typename G::StrideA{}, cute::make_shape(M, K, B_size));
    typename G::StrideB sB = cutlass::make_cute_packed_stride(
        typename G::StrideB{}, cute::make_shape(N, K, B_size));
    typename G::StrideC sC = cutlass::make_cute_packed_stride(
        typename G::StrideC{}, cute::make_shape(M, N, B_size));
    typename G::StrideD sD = cutlass::make_cute_packed_stride(
        typename G::StrideD{}, cute::make_shape(M, N, B_size));

    auto* Ae = reinterpret_cast<typename G::ElementA const*>(A);
    auto* Be = reinterpret_cast<typename G::ElementB const*>(B);
    auto* De = reinterpret_cast<typename G::ElementD*>(C);

    cutlass::KernelHardwareInfo hw_info{};
    hw_info.device_id = 0;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, B_size},
        {Ae, sA, Be, sB},
        {
            { /*alpha*/ typename G::ElementScalar(1.0f), /*beta*/ typename G::ElementScalar(0.0f) },
            /*ptr_C=*/ nullptr, sC,
            /*ptr_D=*/ De, sD
        },
        hw_info
    };

    Gemm op;

    if (op.can_implement(args) != cutlass::Status::kSuccess) {
        throw std::runtime_error("matmul_cute_fp16: can_implement returned non-success");
    }

    void* ws = ensure_workspace(Gemm::get_workspace_size(args));

    cutlass::Status st = op.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        throw std::runtime_error("matmul_cute_fp16: initialize failed");
    }

    st = op.run(stream);
    if (st != cutlass::Status::kSuccess) {
        throw std::runtime_error("matmul_cute_fp16: launch failed");
    }
}

}  // namespace hopper

namespace {

template <class TileShape, class ClusterShape, class KernelSched, class EpilogueSched>
void run_gemm(const void* A, const void* B, void* C,
              int64_t batch, int64_t M, int64_t N, int64_t K)
{
    if (M % 8 != 0 || N % 8 != 0 || K % 8 != 0) {
        throw std::runtime_error("matmul_cute_fp16: M, N, K must be multiples of 8 (TMA alignment 16B / 2B fp16)");
    }

    hopper::launch_gemm_fp16<TileShape, ClusterShape, KernelSched, EpilogueSched>(
        A, B, C, (int)batch, (int)M, (int)N, (int)K, /*stream=*/0);
    // No internal cudaDeviceSynchronize() — caller (Python bench / next op) syncs.
}

}  // anonymous namespace

void matmul_cute_fp16_cuda(
    const void* A, const void* B, void* C,
    int64_t batch, int64_t M, int64_t N, int64_t K)
{
    using namespace cute;
    run_gemm<
        Shape<_128, _128, _64>,
        Shape<_2, _1, _1>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cutlass::epilogue::TmaWarpSpecializedCooperative
    >(A, B, C, batch, M, N, K);
}

void matmul_cute_fp16_c4_cuda(
    const void* A, const void* B, void* C,
    int64_t batch, int64_t M, int64_t N, int64_t K)
{
    using namespace cute;
    run_gemm<
        Shape<_128, _128, _64>,
        Shape<_4, _1, _1>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cutlass::epilogue::TmaWarpSpecializedCooperative
    >(A, B, C, batch, M, N, K);
}

void matmul_cute_fp16_pp_cuda(
    const void* A, const void* B, void* C,
    int64_t batch, int64_t M, int64_t N, int64_t K)
{
    using namespace cute;
    run_gemm<
        Shape<_128, _128, _64>,
        Shape<_2, _1, _1>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong,
        cutlass::epilogue::TmaWarpSpecialized
    >(A, B, C, batch, M, N, K);
}

void matmul_cute_fp16_t256_cuda(
    const void* A, const void* B, void* C,
    int64_t batch, int64_t M, int64_t N, int64_t K)
{
    using namespace cute;
    run_gemm<
        Shape<_128, _256, _64>,
        Shape<_2, _1, _1>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cutlass::epilogue::TmaWarpSpecializedCooperative
    >(A, B, C, batch, M, N, K);
}

}  // namespace tensorax

#else  // !TENSORAX_HOPPER

namespace tensorax {
void matmul_cute_fp16_cuda(const void*, const void*, void*,
                           int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("matmul_cute_fp16: extension not built with TENSORAX_HOPPER");
}
void matmul_cute_fp16_c4_cuda(const void*, const void*, void*,
                              int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("matmul_cute_fp16_c4: extension not built with TENSORAX_HOPPER");
}
void matmul_cute_fp16_pp_cuda(const void*, const void*, void*,
                              int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("matmul_cute_fp16_pp: extension not built with TENSORAX_HOPPER");
}
void matmul_cute_fp16_t256_cuda(const void*, const void*, void*,
                                int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error("matmul_cute_fp16_t256: extension not built with TENSORAX_HOPPER");
}
}  // namespace tensorax

#endif
