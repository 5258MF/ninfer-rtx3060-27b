// w8 splitk/small-T GEMM entry points for non-sm_120a builds.
//
// The w8_small_t_mma and w8_rowsplit_medium_t_splitk kernel families allocate more than 48 KiB
// of static shared memory, which ptxas only accepts for SM120 (Blackwell) targets. They are
// excluded from non-120 builds (see src/CMakeLists.txt); these stubs keep the dispatch layer
// linkable and turn any use (e.g. a w8-quantized checkpoint routed through the splitk/small-T
// paths) into a clear runtime error on this architecture.
#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void throw_w8_sm120_only(const char* feature) {
    throw std::runtime_error(std::string(feature) +
                             ": w8 splitk/small-T GEMM kernels use more than 48 KiB of static "
                             "shared memory (SM120 only) and are not compiled for this "
                             "architecture");
}

} // namespace

// w8 GEMM (ops/linear/w8/w8_launch.h).
void launch_w8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 small-T GEMM");
}

void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 exact-T splitk GEMM");
}

void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 exact-T composite GEMM");
}

void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 dflash medium GEMM");
}

void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 medium splitk GEMM");
}

// Attention input projection (w8_attn_input_kernels.h).
void w8_attn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 attention input splitk GEMM (q/gate/k/v)");
}

void w8_attn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     cudaStream_t) {
    throw_w8_sm120_only("w8 attention input splitk GEMM (q/k/v)");
}

// GDN input projection (w8_gdn_input_kernels.h).
void w8_gdn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                    cudaStream_t) {
    throw_w8_sm120_only("w8 GDN input splitk GEMM");
}

void w8_gdn_input_splitk_conv_snapshot_launch(const Tensor&, const Weight&, const Tensor&,
                                              Tensor&, const Tensor&, const Tensor&,
                                              const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                              cudaStream_t) {
    throw_w8_sm120_only("w8 GDN input splitk conv snapshot");
}

void w8_gdn_input_splitk_conv_record_launch(const Tensor&, const Weight&, const Tensor&,
                                            const Tensor&, const Tensor&, const Tensor&,
                                            Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                            cudaStream_t) {
    throw_w8_sm120_only("w8 GDN input splitk conv record");
}

// Linear add (w8_linear_add_kernels.h).
void w8_linear_add_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 linear add splitk GEMM");
}

void w8_linear_add_medium_splitk_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 linear add medium splitk GEMM");
}

// Linear pair (w8_pair_kernels.h).
void w8_pair_splitk_exact_t_launch(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&,
                                   cudaStream_t) {
    throw_w8_sm120_only("w8 pair splitk exact-T GEMM");
}

void w8_pair_splitk_medium_launch(W8PairScheduleId, const Tensor&, const Weight&, const Weight&,
                                  Tensor&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 pair splitk medium GEMM");
}

// Linear SwiGLU (w8_linear_swiglu_kernels.h).
void w8_linear_swiglu_splitk_exact_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_w8_sm120_only("w8 linear SwiGLU splitk GEMM");
}

} // namespace ninfer::ops::detail
