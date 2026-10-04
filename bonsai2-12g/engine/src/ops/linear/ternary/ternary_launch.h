#pragma once

#include "core/tensor.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The token tile is the only schedule knob on the reference path: one token per CTA at
// T = 1 (decode), eight otherwise (a small prefill). Both share one kernel template.
//
// out_row_stride is the row count of the OUT tensor's parent allocation (the token stride), so a
// caller can have several weights write disjoint row ranges of one fused output -- the split GDN
// and attention parents do exactly that. Pass w.n when the output is the whole tensor.
//
// The int8 rung (#14) needs an activation-quantization scratch buffer, which the caller owns
// because this op runs inside captured CUDA graphs (a lazy cudaMalloc at launch time would be
// illegal there). Callers that have no workspace -- the *_basis entry points used by the fused
// GDN/attention parents -- pass a null scratch and therefore stay on the bf16 rungs.
using TernaryLaunch = void (*)(const Tensor&, const Weight&, Tensor&, std::int32_t, cudaStream_t,
                               TernaryS8Scratch);

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);
void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);

} // namespace ninfer::ops::detail
