#pragma once

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"
#include "ops/linear/ternary/ternary_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy);

// The int8 rung's activation-quantization scratch, allocated from a caller-provided arena. Callers
// that fold the activation themselves (the split attention and GDN input projections rotate once and
// feed several weights) use this and hand the result to the basis entry points below, so those
// projections can take the int8 rung too. Without it they stay on the bf16 rungs no matter how many
// tokens there are -- measured 2026-09-23 on a 4,044-token prefill: those 7 projections per layer
// (attention q/g/k/v + GDN qk/value/z, 13 of the 24 probe lines) were running the bf16 wide-t rung
// at 56.94 ms (PQ2) / 86.93 ms (PTQ1_0) per forward instead of the s8 rung's 36.90 / 34.58 ms.
// The arena capacity already counts these bytes: ternary_rotation_workspace_bytes() is the capacity
// statement for every ternary linear call (see linear.cpp), and it includes the s8 scratch.
TernaryS8Scratch allocate_ternary_s8_scratch(WorkspaceArena& workspace, std::int32_t k,
                                             std::int32_t tokens);

// Rotate the activation into the folded basis when the weight needs it, then run the GEMM.
void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream);

// Run the GEMM when the activation is ALREADY in the folded basis. Callers that feed several
// folded weights from one activation (the split GDN input projection feeds three) rotate once and
// then call this, instead of paying for -- and diverging on -- a redundant rotation per weight.
// `scratch` is optional: pass allocate_ternary_s8_scratch(...)'s result to let this call take the
// int8 rung, or nothing to stay on the bf16 rungs.
void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream,
                            TernaryS8Scratch scratch = {});

// Same, for a weight that writes a ROW RANGE of a larger fused output. `out` must already point at
// the range start (Tensor::slice over ne[0], the contiguous axis, does that) and out_row_stride is
// the parent's row count -- which is also the token stride under ninfer's ne[0]-contiguous layout.
void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream, TernaryS8Scratch scratch = {});

} // namespace ninfer::ops::detail
