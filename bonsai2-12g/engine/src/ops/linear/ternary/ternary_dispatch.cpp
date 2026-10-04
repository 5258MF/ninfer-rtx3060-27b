#include "ops/linear/ternary/ternary_dispatch.h"

#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy) {
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("ternary linear: unsupported shape or T");
    }
    // The reference kernel decodes whole 128-weight groups, so K has to be a whole number
    // of them. Every width the model uses (5120, 6144, 10240, 17408) satisfies this, and
    // row_split_geometry() pads to the same 128 boundary, so no padding groups exist.
    if (k % PTQ1RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    // Both ternary formats take the same schedule; the atom is chosen from w.qtype inside
    // the launch. Decode (T = 1) and small prefill use different token tiles.
    return t == 1 ? launch_ternary_gemm_t1 : launch_ternary_gemm_t8;
}

TernaryS8Scratch allocate_ternary_s8_scratch(WorkspaceArena& workspace, std::int32_t k,
                                             std::int32_t tokens) {
    TernaryS8Scratch scratch{};
    if (tokens < ternary_s8_min_tokens()) {
        // Below the int8 rung's threshold the rung is not taken, so the scratch would be wasted
        // bytes -- allocate nothing and let the launch fall through to the bf16 rungs.
        return scratch;
    }
    const DeviceSpan codes  = workspace.alloc_bytes(ternary_s8_codes_bytes(k, tokens));
    const DeviceSpan scales = workspace.alloc_bytes(ternary_s8_scales_bytes(tokens));
    scratch.codes           = static_cast<std::int8_t*>(codes.data);
    scratch.scales          = static_cast<float*>(scales.data);
    return scratch;
}

void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream, TernaryS8Scratch scratch) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x_folded.ne[1], policy);
    // No workspace on this entry point: the CALLER already folded the activation, and may hand in
    // the int8 scratch it allocated from its own arena (see allocate_ternary_s8_scratch). With an
    // empty scratch these calls stay on the bf16 rungs, which is what they did before 2026-09-23.
    launch(x_folded, w, out, out_row_stride, stream, scratch);
}

void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream, TernaryS8Scratch scratch) {
    ternary_dispatch_basis_strided(x_folded, w, out, w.n, policy, stream, scratch);
}

void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x.ne[1], policy);

    if (!ternary_rotation_enabled()) {
        // NINFER_TERNARY_HADAMARD=0: run the GEMM against the raw activation so the full forward
        // pass (and therefore the M4 decode speed) is measurable. The result is numerically
        // meaningless -- the activation is in the wrong basis -- which is the point: it separates
        // "the ternary decode is broken" from "the rotation is broken" without a rebuild.
        launch(x, w, out, w.n, stream, TernaryS8Scratch{});
        return;
    }
    if (workspace == nullptr) {
        // Name the shape in the message: a folded ternary weight reaching the workspace-free
        // entry point is a call-site wiring gap, and the shape is what identifies the weight.
        throw std::invalid_argument(
            "ternary linear: the folded basis needs a rotation workspace [N=" +
            std::to_string(w.n) + ", K=" + std::to_string(w.k) + ", T=" +
            std::to_string(x.ne[1]) + ", qtype=" + std::to_string(static_cast<int>(w.qtype)) + "]");
    }

    // Scoped: the scratch is handed back when this op returns, so it does not accumulate across
    // the (many) graph constructions of one load. Sequential reuse on one stream is safe.
    // folded_activation() rejects a ternary weight with no sign block, so an unfolded artifact
    // fails loudly here instead of silently multiplying by unfolded weights.
    auto scope              = workspace->scope();
    const Tensor activation = folded_activation(x, w, *workspace, stream);

    // int8 rung scratch (#14): one int8 code row per token plus one fp32 scale per token. Taken from
    // the same arena the rotation just used, and counted by ternary_rotation_workspace_bytes() so
    // the planner sizes the arena for it -- allocating it lazily inside the launch would be illegal,
    // because this op runs inside captured CUDA graphs.
    TernaryS8Scratch scratch{};
    if (x.ne[1] >= ternary_s8_min_tokens()) {
        const DeviceSpan codes  = workspace->alloc_bytes(ternary_s8_codes_bytes(w.k, x.ne[1]));
        const DeviceSpan scales = workspace->alloc_bytes(ternary_s8_scales_bytes(x.ne[1]));
        scratch.codes           = static_cast<std::int8_t*>(codes.data);
        scratch.scales          = static_cast<float*>(scales.data);
    }
    launch(activation, w, out, w.n, stream, scratch);
}

} // namespace ninfer::ops::detail
