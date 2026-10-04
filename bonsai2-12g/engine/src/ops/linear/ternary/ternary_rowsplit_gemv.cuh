#pragma once

// Decode-shaped ternary GEMV (T == 1) for PQ2_0: one warp owns one output row.
//
// Why: the reference kernel in ternary_rowsplit_gemm.cuh gives each output row a full 128-thread
// CTA and seven block-wide barriers. That is fine for correctness but leaves decode far from the
// card's measured bandwidth. Here each lane covers four consecutive weights of every 128-group, so
// a warp reads exactly one 32-byte code span plus one 2-byte scale per group and reduces through
// shuffles with no __syncthreads at all.
//
// PQ2_0 packing makes the mapping exact rather than approximate: a group is 32 bytes holding four
// two-bit codes each, so for lane l the four columns 4l..4l+3 live entirely in byte l. Lane l
// loads one byte, decodes four weights, and consumes four bf16 activations.
//
// Occupancy, not instruction count, is what this kernel is tuned for: a GEMV needs many warps in
// flight to cover DRAM latency. A two-rows-per-warp variant (which reuses the activation loads)
// measured SLOWER (32.8 vs 46.4 t/s) because the extra accumulators and row bases cost registers
// and dropped the resident warp count. So: one row per warp, and only the micro-optimisations that
// remove instructions without adding state -- one 16-bit scale load, paired bf16 activation
// loads, and the per-group scale multiply hoisted out of the four FMAs.
//
// LAYOUT: ninfer/ggml put ne[0] on the contiguous axis, so a [k, 1] activation keeps element
// (column, 0) at column, and a one-token output row is simply out[row].

// ternary_pow3() and the already-verified PTQ1_0 decode live here; PTQ1 gets its weight fetch from
// that single source of truth instead of a second hand-written copy of the trit formula.
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Warps per block for the GEMV below.
inline constexpr int kGemvWarpsPerBlock = 8;

inline constexpr int kGemvCodeBytesPerGroup  = 32;
inline constexpr int kGemvScaleBytesPerGroup = 2;
inline constexpr int kGemvGroupK             = 128;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    // One 16-bit load instead of two byte loads plus a shift/or; the scale plane is 2-byte aligned.
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                             std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant, for the speculative VERIFY pass (T = draft + 1, i.e. 2..4).
//
// A per-token GEMV would re-read every weight for every token, and weights are exactly what the
// decode path is bound by, so a verify round would cost T full decode passes and speculation could
// never pay for itself (measured: 15.7 t/s with MTP N=2 against 62.1 t/s without). This kernel
// keeps the weights single-read: the code byte and scale are loaded once per group and reused for
// all kT tokens, while each token contributes its own four activations.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        // Weights: one byte of codes + one 16-bit scale, reused across every token in the tile.
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // Token-major output: element (row, token) lives at token * out_row_stride + row.
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

// One lane owns four consecutive columns (4l..4l+3). For PTQ1_0 those four always fall inside ONE
// region, and inside a region the trit multiplier is a constant OF THE LANE, not of the column --
// so the region branch and the 3^n chain are hoisted out of the group loop and the inner loop keeps
// only a multiply, a uint8 wrap, a shift and a subtract per weight. That matters here because the
// PTQ1_0 unpack is instruction-bound rather than bandwidth-bound (the PQ2_0 "one lane owns one code
// byte" trick does not transfer, since a base-3 byte carries parts of five different columns):
//   region 0: weights   0..79 : byte codes[j & 15],            trit = j >> 4      (c=16 stage)
//   region 1: weights  80..119: byte codes[16 + ((j-80) & 7)], trit = (j-80) >> 3 (c=8 stage)
//   region 2: weights 120..127: bytes high[0],high[1],high[0],high[1] with trits t,t,t+1,t+1
inline constexpr int kPtq1CodeBytesPerGroup = 24;
inline constexpr int kPtq1HighBytesPerGroup = 2;

struct Ptq1LanePlan {
    const std::uint8_t* row;      // code plane row, or high plane row in region 2
    int bytes_per_group;
    int src_off;                  // byte offset of the lane's first weight inside the group
    int pow_a;                    // 3^trit for columns 0 and 1
    int pow_b;                    // 3^trit for columns 2 and 3 (region 2 only; == pow_a otherwise)
    int idx0, idx1, idx2, idx3;   // byte index of each column inside the lane's span
};

__device__ __forceinline__ Ptq1LanePlan ptq1_lane_plan(int lane,
                                                       const std::uint8_t* code_row,
                                                       const std::uint8_t* high_row) {
    Ptq1LanePlan p{};
    const int col0 = lane * 4;
    if (lane < 20) {
        p.row = code_row;
        p.bytes_per_group = kPtq1CodeBytesPerGroup;
        p.src_off         = col0 & 15;
        p.pow_a = p.pow_b = ternary_pow3(col0 >> 4);
        p.idx0 = 0; p.idx1 = 1; p.idx2 = 2; p.idx3 = 3;
    } else if (lane < 30) {
        const int local = col0 - 80;
        p.row = code_row;
        p.bytes_per_group = kPtq1CodeBytesPerGroup;
        p.src_off         = 16 + (local & 7);
        p.pow_a = p.pow_b = ternary_pow3(local >> 3);
        p.idx0 = 0; p.idx1 = 1; p.idx2 = 2; p.idx3 = 3;
    } else {
        const int local = col0 - 120;
        p.row = high_row;
        p.bytes_per_group = kPtq1HighBytesPerGroup;
        p.src_off         = 0;
        p.pow_a           = ternary_pow3(local >> 1);
        p.pow_b           = ternary_pow3((local >> 1) + 1);
        p.idx0 = 0; p.idx1 = 1; p.idx2 = 0; p.idx3 = 1;
    }
    return p;
}

// ((uint16)(q * 3^n) * 3) >> 8 with the uint8 wrap-around the C dequantizer relies on, minus the
// scale multiply (hoisted, exactly as the PQ2 kernel hoists its own).
__device__ __forceinline__ float ptq1_trit_from_byte(std::uint8_t raw, int pow3) {
    const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3);
    return static_cast<float>(static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8) - 1);
}

__device__ __forceinline__ float ptq1_dot4(const Ptq1LanePlan& p, int group,
                                           const float2& low, const float2& high_act) {
    const std::uint8_t* src =
        p.row + static_cast<std::int64_t>(group) * p.bytes_per_group + p.src_off;
    const float w0 = ptq1_trit_from_byte(src[p.idx0], p.pow_a);
    const float w1 = ptq1_trit_from_byte(src[p.idx1], p.pow_a);
    const float w2 = ptq1_trit_from_byte(src[p.idx2], p.pow_b);
    const float w3 = ptq1_trit_from_byte(src[p.idx3], p.pow_b);
    return fmaf(w0, low.x, fmaf(w1, low.y, fmaf(w2, high_act.x, w3 * high_act.y)));
}

// T == 1 decode: same shape as the PQ2 kernel above (one warp per output row, shuffle reduce, no
// __syncthreads), so it inherits the same occupancy-first reasoning.
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, std::int32_t rows,
                              std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1CodeBytesPerGroup;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1HighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    const Ptq1LanePlan plan = ptq1_lane_plan(lane, code_row, high_row);

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high_act = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const float dot = ptq1_dot4(plan, group, low, high_act);
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant: identical policy to the PQ2 tile kernel above -- the four weight values
// are decoded once per group and reused by every token in the tile, which is what keeps the
// speculative verify pass (T = draft + 1) from costing T full decode passes.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_ptq1_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ high,
                                   const std::uint8_t* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                   std::int32_t groups_per_row, std::int32_t tokens,
                                   std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1CodeBytesPerGroup;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(warp) * groups_per_row * kPtq1HighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    const Ptq1LanePlan plan = ptq1_lane_plan(lane, code_row, high_row);

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t* src =
            plan.row + static_cast<std::int64_t>(group) * plan.bytes_per_group + plan.src_off;
        const float scale   = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = ptq1_trit_from_byte(src[plan.idx0], plan.pow_a);
        const float weight1 = ptq1_trit_from_byte(src[plan.idx1], plan.pow_a);
        const float weight2 = ptq1_trit_from_byte(src[plan.idx2], plan.pow_b);
        const float weight3 = ptq1_trit_from_byte(src[plan.idx3], plan.pow_b);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high_act = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high_act.x, weight3 * high_act.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

} // namespace ninfer::ops::detail
