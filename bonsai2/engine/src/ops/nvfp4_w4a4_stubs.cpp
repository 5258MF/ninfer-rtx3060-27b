// NVFP4 W4A4 (TMA) kernel entry points for non-sm_120a builds.
//
// The warp-specialized W4A4 kernels use Blackwell TMA instructions (nvfp4_w4a4_tma.cuh)
// and are excluded from non-120 builds (src/CMakeLists.txt). These stubs keep the dispatch
// layer linkable and turn any accidental NVFP4 W4A4 use into a clear runtime error.
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void nvfp4_w4a4_unsupported() {
    throw std::runtime_error(
        "NVFP4 W4A4 execution requires the SM120 (Blackwell) TMA kernels, which are not "
        "compiled for this architecture");
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight,
                                Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x;
    (void)weight;
    (void)workspace;
    (void)stream;
    nvfp4_w4a4_unsupported();
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x;
    (void)weight;
    (void)out;
    (void)workspace;
    (void)stream;
    nvfp4_w4a4_unsupported();
}

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    (void)x;
    (void)weight;
    (void)q;
    (void)gate;
    (void)k;
    (void)v;
    (void)workspace;
    (void)stream;
    nvfp4_w4a4_unsupported();
}

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x;
    (void)weight;
    (void)qkv;
    (void)z;
    (void)workspace;
    (void)stream;
    nvfp4_w4a4_unsupported();
}

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    (void)x;
    (void)weight;
    (void)residual;
    (void)workspace;
    (void)stream;
    nvfp4_w4a4_unsupported();
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    (void)x;
    (void)weight;
    (void)out;
    (void)workspace;
    (void)stream;
    nvfp4_w4a4_unsupported();
}

} // namespace ninfer::ops::detail
