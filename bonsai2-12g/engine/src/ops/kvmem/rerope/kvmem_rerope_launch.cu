// KVMem re-RoPE consumer, device half. Contract: ops/kvmem/rerope/kvmem_rerope.h.
//
// PORTED FROM (Apache-2.0, attribution kept): kvmem-qw3, src/kernels_cuda.cu:4936-5060
// (rope_block_remap_paged_kernel / _batched_kernel). What we keep from theirs is the DECISION and
// the two-stage ordering for a de-rotate/re-rotate pair:
//
//   de-rotate by the OLD position, then re-rotate by the NEW one, each with its own angles --
//   NOT a single rotation by (new-orig) -- because the range-reduction error of the bake has to
//   cancel against the same sincos the bake used (their comment at kernels_cuda.cu:4930).
//
// Here that reasoning leads somewhere different, and the difference is the whole point of this file:
//
//   THEIR fp8 K is raw e4m3 of the baked K (kernels_cuda.cu:9142), so a two-stage rotation in the
//   code domain is a valid approximation of a re-bake. OURS is not that: our stored fp8 K is
//   quantize(H * baked_K) with H the normalized Sylvester Hadamard
//   (ops/kv_cache/append/kernel.cuh:36 -> :43; the attention side H's the query instead,
//   ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh:137). H does not commute with RoPE's
//   pair rotations, so the two-stage code-domain rotation is NOT an approximation of anything here:
//   it rotates a basis, not a vector. We therefore do not ship it -- and the standalone verifier
//   keeps it as a negative control, so the claim is measured rather than argued.
//
// THE SHIPPED PATH re-bakes from raw K through the engine's own pipeline:
//   raw K (BF16, pre-RoPE) -> RoPE at the new position (the fixed Text 1-D D256/R64 arithmetic,
//   expression shapes copied from ops/kernel/rope.cuh:81-83) -> BF16 narrowing (the KV tensor the
//   append kernel reads) -> normalized Hadamard -> warp absmax -> kv_cache_fp8_quant_params /
//   kv_cache_fp8_quant_code (the engine's own codec) -> codes + scale.
// Row ownership is the append kernel's: one warp per (token, kv head), lane l carrying dims
// l + 32*r, r = 0..7, so the Hadamard and its rounding order are the engine's exactly.

#include "ops/kvmem/rerope/kvmem_rerope.h"

#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/rope.cuh"                 // kTextRopeInvFrequency + the rope convention
#include "ops/kv_cache/append/geometry.cuh"    // KVCacheAppendFullGeometry<KVHeads>
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh" // the engine's fp8 quantizer
#include "ops/kv_cache/int8_g64_codec.cuh"     // the engine's int8 group-64 quantizer
#include "ops/kv_cache/hadamard_d256.cuh"      // normalized_hadamard_d256_inplace

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {
namespace detail {
namespace {

constexpr int kHeadDim    = kKvMemReropeHeadDim;
constexpr int kRowValues  = kHeadDim / 32; // one warp owns one row: lane l carries l + 32*r
constexpr int kRowsPerCTA = 8;             // 256 threads / 32
constexpr unsigned kFullMask = 0xffffffffU;

// ---------------------------------------------------------------------------------------------
// Store policies.
//
// Everything up to this point -- RoPE at the new position, the BF16 narrowing, the normalized
// Hadamard -- is shared, because the Hadamard-on-K invariant holds for every KV encoding in this
// tree (see the contract header). Only the final quantize-and-store differs per codec, so that is
// the only thing a policy owns.
//
// Lane l carries dims l + 32*r for r = 0..7, which is the append kernel's row ownership and the
// reason the rounding order matches the engine's exactly.
// ---------------------------------------------------------------------------------------------

struct ReropeStoreFp8Row256 {
    template <int KVHeads>
    static __device__ __forceinline__ void store(const float (&values)[kRowValues], int lane,
                                                 int page, int kv_head, int page_offset,
                                                 void* __restrict__ codes_v,
                                                 void* __restrict__ scales_v) {
        auto* __restrict__ codes  = static_cast<std::uint8_t*>(codes_v);
        auto* __restrict__ scales = static_cast<__half*>(scales_v);
        float local_absmax = 0.0F;
#pragma unroll
        for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
        const KVCacheFp8QuantParams quant =
            kv_cache_fp8_quant_params(warp_max(local_absmax, kFullMask));
#pragma unroll
        for (int r = 0; r < kRowValues; ++r) {
            codes[kv_cache_fp8_code_index<KVCacheAppendFullGeometry<KVHeads>>(
                page, kv_head, lane + 32 * r, page_offset)] =
                kv_cache_fp8_quant_code(values[r], quant.inverse_scale);
        }
        if (lane == 0) {
            scales[kv_cache_fp8_scale_index<KVCacheAppendFullGeometry<KVHeads>>(
                page, kv_head, page_offset)] = quant.scale;
        }
    }
};

struct ReropeStoreInt8G64 {
    // Int8 keeps FOUR group scales per row instead of one. With head_dim 256, 32 lanes and
    // 8 registers per lane, dim = lane + 32*r falls in group (lane + 32*r) / 64 = r / 2 for every
    // lane, so the group a register belongs to is lane-invariant and the four absmax reductions are
    // plain warp reductions over register pairs {2g, 2g+1}. Every lane runs all four so the warp
    // reduction stays convergent.
    static_assert(kRowValues == 2 * kKVCacheInt8Groups,
                  "int8 group mapping assumes 8 registers over 4 groups of 64 dims");

    template <int KVHeads>
    static __device__ __forceinline__ void store(const float (&values)[kRowValues], int lane,
                                                 int page, int kv_head, int page_offset,
                                                 void* __restrict__ codes_v,
                                                 void* __restrict__ scales_v) {
        auto* __restrict__ codes  = static_cast<std::int8_t*>(codes_v);
        auto* __restrict__ scales = static_cast<__half*>(scales_v);
        KVCacheInt8QuantParams quant[kKVCacheInt8Groups];
#pragma unroll
        for (int g = 0; g < kKVCacheInt8Groups; ++g) {
            const float pair_absmax =
                fmaxf(fabsf(values[2 * g]), fabsf(values[2 * g + 1]));
            quant[g] = kv_cache_int8_quant_params(warp_max(pair_absmax, kFullMask));
        }
#pragma unroll
        for (int r = 0; r < kRowValues; ++r) {
            codes[kv_cache_int8_quant_code_index<KVCacheAppendFullGeometry<KVHeads>>(
                page, kv_head, lane + 32 * r, page_offset)] =
                kv_cache_int8_quant_code(values[r], quant[r / 2].inverse_scale);
        }
        if (lane < kKVCacheInt8Groups) {
            scales[kv_cache_int8_quant_scale_index<KVCacheAppendFullGeometry<KVHeads>>(
                page, kv_head, lane, page_offset)] = quant[lane].scale;
        }
    }
};

// The fixed Text 1-D path uses the table; anything else uses the generic kernel's frequency form.
__device__ __forceinline__ float rerope_inv_frequency(int pair, int rotary_dim, float theta) {
    if (rotary_dim == kKvMemReropeRotaryDim && theta == kKvMemReropeTheta) {
        return kTextRopeInvFrequency[pair];
    }
    return powf(theta, -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim));
}

// ---------------------------------------------------------------------------------------------
// Path B: re-bake from raw K and quantize. One warp owns one (token, kv head) row; CTA = 256
// threads = 8 rows. grid.x covers rows, grid.y unused, grid.z selects the block (batched form).
//
// Pairing note: lane l holds dims l+32r. With rope_dim a multiple of 64 the split-half partner of
// dim d is d +/- rope_dim/2, which lands in another REGISTER of the SAME LANE (r +/- half/32), so
// the rotation needs no shuffles. valid_geometry enforces that.
// ---------------------------------------------------------------------------------------------
template <int KVHeads, typename Store>
__device__ __forceinline__ void rerope_rebake_row(const __nv_bfloat16* __restrict__ raw_k,
                                                  void* __restrict__ codes,
                                                  void* __restrict__ scales,
                                                  const std::int32_t* __restrict__ block_table,
                                                  std::int32_t raw_token_begin,
                                                  std::int32_t position, int kv_head, int token,
                                                  int rope_dim, float theta, int lane) {
    const int half        = rope_dim / 2;
    const int half_regs   = half / 32;
    const std::int32_t rtok = raw_token_begin + token;
    const __nv_bfloat16* row =
        raw_k + static_cast<std::int64_t>(kHeadDim) *
                    (static_cast<std::int64_t>(kv_head) + static_cast<std::int64_t>(KVHeads) * rtok);

    float values[kRowValues];
    float raw_row[kRowValues];
#pragma unroll
    for (int r = 0; r < kRowValues; ++r) {
        values[r]  = __bfloat162float(row[lane + 32 * r]);
        raw_row[r] = values[r];
    }

    // RoPE at the new position, in the engine's expression shapes (ops/kernel/rope.cuh:81-83).
    // The partner values are read from `raw_row`, never from a register already overwritten this
    // pass -- the engine's apply_rope_head loads both halves into locals before storing for exactly
    // this reason.
    if (rope_dim > 0) {
#pragma unroll
        for (int r = 0; r < kRowValues; ++r) {
            const int d = lane + 32 * r;
            if (d >= rope_dim) { continue; }
            const int pair  = (d < half) ? d : d - half;
            const int other = (d < half) ? r + half_regs : r - half_regs;
            float sine, cosine;
            sincosf(static_cast<float>(position) *
                        rerope_inv_frequency(pair, rope_dim, theta),
                    &sine, &cosine);
            const float first  = raw_row[r];
            const float second = raw_row[other];
            values[r] = (d < half) ? (first * cosine - second * sine)
                                   : (second * cosine + first * sine);
        }
    }

    // The KV tensor the append kernel reads is BF16; narrow here so the rows we Hadamard are the
    // rows it would have Hadamarded.
#pragma unroll
    for (float& value : values) { value = __bfloat162float(__float2bfloat16_rn(value)); }

    normalized_hadamard_d256_inplace(values, lane);

    Store::template store<KVHeads>(values, lane, paged_kv_physical_page(block_table, position),
                                   kv_head, position & kPagedKVPageMask, codes, scales);
}

template <int KVHeads, typename Store>
__global__ void rerope_rebake_kernel(const __nv_bfloat16* __restrict__ raw_k,
                                     void* __restrict__ codes, void* __restrict__ scales,
                                     const std::int32_t* __restrict__ block_table,
                                     std::int32_t raw_token_begin, std::int32_t win_base,
                                     std::int32_t n_tokens, std::int32_t rope_dim, float theta) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kRowsPerCTA + warp;
    if (unit >= n_tokens * KVHeads) { return; }
    const int kv_head = unit % KVHeads;
    const int token   = unit / KVHeads;
    rerope_rebake_row<KVHeads, Store>(raw_k, codes, scales, block_table, raw_token_begin,
                                      win_base + token, kv_head, token, rope_dim, theta, lane);
}

template <int KVHeads, typename Store>
__global__ void rerope_rebake_batched_kernel(const __nv_bfloat16* __restrict__ raw_k,
                                             void* __restrict__ codes,
                                             void* __restrict__ scales,
                                             const std::int32_t* __restrict__ block_table,
                                             const std::int32_t* __restrict__ to_base,
                                             const std::int32_t* __restrict__ raw_token_begin,
                                             const std::int32_t* __restrict__ n_tokens,
                                             std::int32_t rope_dim, float theta) {
    const int bz = static_cast<int>(blockIdx.z);
    const int count = n_tokens[bz];
    if (count <= 0) { return; } // contract: no write at all
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int unit = static_cast<int>(blockIdx.x) * kRowsPerCTA + warp;
    if (unit >= count * KVHeads) { return; }
    const int kv_head = unit % KVHeads;
    const int token   = unit / KVHeads;
    rerope_rebake_row<KVHeads, Store>(raw_k, codes, scales, block_table, raw_token_begin[bz],
                                      to_base[bz] + token, kv_head, token, rope_dim, theta, lane);
}

// ---------------------------------------------------------------------------------------------
// Launchers. kv_heads dispatches onto the two geometries the engine instantiates (4 and 2).
// ---------------------------------------------------------------------------------------------
bool valid_geometry(std::int32_t kv_heads, std::int32_t rope_dim) {
    if (kv_heads != 2 && kv_heads != 4) { return false; }
    if (rope_dim <= 0 || rope_dim > kHeadDim) { return false; }
    // The split-half partner must stay inside the lane's own registers: half must be a multiple of
    // 32, i.e. rope_dim a multiple of 64.
    if ((rope_dim % 64) != 0) { return false; }
    return true;
}

int rows_for(std::int32_t tokens, std::int32_t kv_heads) {
    return (tokens * kv_heads + kRowsPerCTA - 1) / kRowsPerCTA;
}

template <int KVHeads, typename Store>
cudaError_t launch_rebake(const void* raw_k_bf16, std::int32_t raw_token_begin, void* k_codes,
                          void* k_scales, const std::int32_t* block_table, std::int32_t win_base,
                          std::int32_t n_tokens, std::int32_t rope_dim, float theta,
                          cudaStream_t stream) {
    rerope_rebake_kernel<KVHeads, Store><<<rows_for(n_tokens, KVHeads), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_k_bf16), k_codes, k_scales, block_table,
        raw_token_begin, win_base, n_tokens, rope_dim, theta);
    return cudaGetLastError();
}

template <int KVHeads, typename Store>
cudaError_t launch_rebake_batched(const void* raw_k_bf16, void* k_codes, void* k_scales,
                                  const std::int32_t* block_table, const std::int32_t* to_base,
                                  const std::int32_t* raw_token_begin,
                                  const std::int32_t* n_tokens, std::int32_t n_blocks,
                                  std::int32_t max_tokens, std::int32_t rope_dim, float theta,
                                  cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(rows_for(max_tokens, KVHeads)), 1U,
                    static_cast<unsigned>(n_blocks));
    rerope_rebake_batched_kernel<KVHeads, Store><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_k_bf16), k_codes, k_scales, block_table, to_base,
        raw_token_begin, n_tokens, rope_dim, theta);
    return cudaGetLastError();
}

// kv_heads x codec dispatch, kept in one place so both entry points agree.
template <int KVHeads>
cudaError_t launch_rebake_for(KvMemReropeCodec codec, const void* raw_k_bf16,
                              std::int32_t raw_token_begin, void* k_codes, void* k_scales,
                              const std::int32_t* block_table, std::int32_t win_base,
                              std::int32_t n_tokens, std::int32_t rope_dim, float theta,
                              cudaStream_t stream) {
    if (codec == KvMemReropeCodec::Int8Group64) {
        return launch_rebake<KVHeads, ReropeStoreInt8G64>(raw_k_bf16, raw_token_begin, k_codes,
                                                          k_scales, block_table, win_base,
                                                          n_tokens, rope_dim, theta, stream);
    }
    return launch_rebake<KVHeads, ReropeStoreFp8Row256>(raw_k_bf16, raw_token_begin, k_codes,
                                                        k_scales, block_table, win_base, n_tokens,
                                                        rope_dim, theta, stream);
}

template <int KVHeads>
cudaError_t launch_rebake_batched_for(KvMemReropeCodec codec, const void* raw_k_bf16,
                                      void* k_codes, void* k_scales,
                                      const std::int32_t* block_table, const std::int32_t* to_base,
                                      const std::int32_t* raw_token_begin,
                                      const std::int32_t* n_tokens, std::int32_t n_blocks,
                                      std::int32_t max_tokens, std::int32_t rope_dim, float theta,
                                      cudaStream_t stream) {
    if (codec == KvMemReropeCodec::Int8Group64) {
        return launch_rebake_batched<KVHeads, ReropeStoreInt8G64>(
            raw_k_bf16, k_codes, k_scales, block_table, to_base, raw_token_begin, n_tokens,
            n_blocks, max_tokens, rope_dim, theta, stream);
    }
    return launch_rebake_batched<KVHeads, ReropeStoreFp8Row256>(
        raw_k_bf16, k_codes, k_scales, block_table, to_base, raw_token_begin, n_tokens, n_blocks,
        max_tokens, rope_dim, theta, stream);
}

} // namespace
} // namespace detail

cudaError_t kvmem_rerope_rebake_launch(const void* raw_k_bf16, std::int32_t raw_token_begin,
                                       void* k_codes, void* k_scales,
                                       const std::int32_t* block_table, std::int32_t kv_heads,
                                       std::int32_t win_base, std::int32_t n_tokens,
                                       std::int32_t rope_dim, float theta, KvMemReropeCodec codec,
                                       cudaStream_t stream) {
    if (n_tokens <= 0) { return cudaSuccess; }
    if (raw_token_begin < 0 || win_base < 0) { return cudaErrorInvalidValue; }
    if (!detail::valid_geometry(kv_heads, rope_dim)) { return cudaErrorInvalidValue; }
    if (kv_heads == 4) {
        return detail::launch_rebake_for<4>(codec, raw_k_bf16, raw_token_begin, k_codes, k_scales,
                                            block_table, win_base, n_tokens, rope_dim, theta,
                                            stream);
    }
    return detail::launch_rebake_for<2>(codec, raw_k_bf16, raw_token_begin, k_codes, k_scales,
                                        block_table, win_base, n_tokens, rope_dim, theta, stream);
}

cudaError_t kvmem_rerope_rebake_batched_launch(const void* raw_k_bf16, void* k_codes,
                                               void* k_scales, const std::int32_t* block_table,
                                               const std::int32_t* to_base,
                                               const std::int32_t* raw_token_begin,
                                               const std::int32_t* n_tokens, std::int32_t n_blocks,
                                               std::int32_t max_tokens, std::int32_t kv_heads,
                                               std::int32_t rope_dim, float theta,
                                               KvMemReropeCodec codec, cudaStream_t stream) {
    if (n_blocks <= 0 || max_tokens <= 0) { return cudaSuccess; }
    if (!detail::valid_geometry(kv_heads, rope_dim)) { return cudaErrorInvalidValue; }
    if (kv_heads == 4) {
        return detail::launch_rebake_batched_for<4>(codec, raw_k_bf16, k_codes, k_scales,
                                                    block_table, to_base, raw_token_begin,
                                                    n_tokens, n_blocks, max_tokens, rope_dim,
                                                    theta, stream);
    }
    return detail::launch_rebake_batched_for<2>(codec, raw_k_bf16, k_codes, k_scales, block_table,
                                                to_base, raw_token_begin, n_tokens, n_blocks,
                                                max_tokens, rope_dim, theta, stream);
}

} // namespace ninfer::ops
