#ifdef TENSORAX_HOPPER

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace tensorax {
namespace hopper {

using namespace cute;

extern "C" int tensorax_hopper_compiled() {
    using BlockShape = Shape<_128, _64, _32>;
    using BlockLayout = Layout<BlockShape>;
    return static_cast<int>(size(BlockLayout{}));
}

}
}

#endif
