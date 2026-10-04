#pragma once

#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The decode-shaped GEMV that replaces the reference kernel at T == 1 lives in
// ternary_rowsplit_gemv.cuh. It is deliberately NOT included here yet: the CUDA 13.3 cudafe++
// front end dies (0xC0000409) on that kernel in this translation unit, so it stays out of the built
// header until the construct is pinned down.

// Correctness-first reference path for the ternary row-split formats.
//
// One CTA owns one output row and a tile of kTileT tokens. The CTA has exactly kGroupK
// (128) threads, so thread `index` owns weight `index` inside every 128-weight group:
// a group is decoded once by the whole CTA and reused across the whole token tile, which
// is what keeps this from being 128x redundant. This is the "dequantize the block and
// accumulate in F32" shape that upstream's reference kernel uses; a repack/SIMT fast path
// and an MMA path are deliberately left out until the numerics are pinned down.
//
// Activation layout is [K, T] with ne[0] = K CONTIGUOUS, i.e. the ninfer/ggml convention: element
// (column, token) lives at token * k + column. Output is [N, T] the same way: (row, token) at
// token * rows + row. Treating either as row-major ([K, T] with k*T + t) scrambles every prefill
// while T == 1 still works, because the two layouts coincide exactly at one token.
template <class Storage, class Atom, int kTileT>
__global__ __launch_bounds__(Storage::kGroupK)
void ternary_rowsplit_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ high,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t k, std::int32_t t, std::int32_t groups_per_row,
                                  std::int32_t out_row_stride) {
    static_assert(kTileT >= 1, "the ternary GEMM needs a positive token tile");
    constexpr int kThreads = Storage::kGroupK;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) { return; }
    const int t0 = static_cast<int>(blockIdx.y) * kTileT;

    // Planes are laid out row-major over [rows, groups_per_row].
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups_per_row * Storage::kCodeBytesPerGroup;
    const std::uint8_t* high_row =
        high == nullptr ? nullptr
                        : high + static_cast<std::int64_t>(row) * groups_per_row *
                                     Storage::kHighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * groups_per_row * Storage::kScaleBytesPerGroup;

    const int index = static_cast<int>(threadIdx.x);

    float accumulator[kTileT];
#pragma unroll
    for (int i = 0; i < kTileT; ++i) { accumulator[i] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t* group_codes = code_row + group * Storage::kCodeBytesPerGroup;
        const std::uint8_t* group_high =
            high_row == nullptr ? nullptr : high_row + group * Storage::kHighBytesPerGroup;
        const float scale = Atom::load_scale(scale_row + group * Storage::kScaleBytesPerGroup);
        const float weight = Atom::decode_one(group_codes, group_high, scale, index);

        const std::int32_t column = group * Storage::kGroupK + index;
        if (column < k) {
#pragma unroll
            for (int i = 0; i < kTileT; ++i) {
                const std::int32_t token = t0 + i;
                if (token < t) {
                    accumulator[i] += weight *
                        __bfloat162float(x[static_cast<std::int64_t>(token) * k + column]);
                }
            }
        }
    }

    __shared__ float partials[kTileT][kThreads];
#pragma unroll
    for (int i = 0; i < kTileT; ++i) { partials[i][index] = accumulator[i]; }
    __syncthreads();

    // Tree reduction over the CTA; every stage halves the active thread range.
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (index < stride) {
#pragma unroll
            for (int i = 0; i < kTileT; ++i) { partials[i][index] += partials[i][index + stride]; }
        }
        __syncthreads();
    }

    if (index == 0) {
#pragma unroll
        for (int i = 0; i < kTileT; ++i) {
            const std::int32_t token = t0 + i;
            if (token < t) {
                // out_row_stride is the PARENT's row count, so several projections can write
                // disjoint row ranges of one fused output (the GDN split parent, for instance).
                out[static_cast<std::int64_t>(token) * out_row_stride + row] =
                    __float2bfloat16_rn(partials[i][0]);
            }
        }
    }
}

} // namespace ninfer::ops::detail
