#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Working-set capture geometry shared by the capture op callers: retrieval blocks are
// 128 tokens. Buffer slots are sized by the Program's configured prefill chunk;
// split chunks can contribute to one block across multiple launches.
inline constexpr std::uint32_t kKvmemCaptureBlockTokens = 128;



/**
 * Accumulates columns [begin, begin + count) of a row-major BF16 matrix into an FP32
 * column-sum vector, adding into the existing sum values (+=, not overwrite).
 *
 * x is contiguous BF16 [rows, tokens]; sums is contiguous FP32 [rows]. count may be
 * zero (a no-op launch is permitted but must still be ordered on the stream). The
 * oracle evaluates the post-state sum as old_sum + sum of the represented BF16 columns
 * in FP64. The Op owns no allocation and leaves x unchanged. The working-set capture
 * path calls it once per attention layer over a chunk-local span of the pre-RoPE
 * query or key tensor.
 */
void span_accumulate(const Tensor& x, std::uint32_t begin, std::uint32_t count, Tensor& sums,
                     cudaStream_t stream);

} // namespace ninfer::ops
