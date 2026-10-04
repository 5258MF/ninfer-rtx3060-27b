#pragma once

// Signed int8, per-token G64 KV-cache codec shared by append and causal-attention kernels.
// This header owns index math, vectorized decode, and scalar encode; there is deliberately no
// standalone transcode kernel in the production path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheInt8HeadDim = 256;
inline constexpr int kKVCacheInt8Group   = 64;
inline constexpr int kKVCacheInt8Groups  = kKVCacheInt8HeadDim / kKVCacheInt8Group;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_code_index(int physical_page, int kv_head, int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_quant_src_index(int kv_head, int d,
                                                                      int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheInt8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheInt8QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/127); codes always use the reciprocal of that represented FP16 value.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int8_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ int4 kv_cache_int8_dequant_i8x8_from(const std::int8_t* codes8,
                                                                float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// ---------------------------------------------------------------------------
// Fork-local packed / rotated mode extensions (rk8v4, rk4v4, rk4v4-e8,
// rk2v4-e8). K and V are optionally rotated per 64-group by an H64 transform
// before encoding; packed modes store two signed 4-bit codes per byte. The E8
// lattice codecs themselves live in ops/kernel/e8_lattice.cuh and
// ops/kernel/e8_root_codec.cuh; this header only owns the shared index math
// and scalar helpers.
// ---------------------------------------------------------------------------

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_i4_code_index(int physical_page, int kv_head,
                                                               int packed_d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim / 2, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, packed_d);
}

__device__ __forceinline__ std::int8_t kv_cache_i4_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ std::uint8_t kv_cache_pack_i4(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

__device__ __forceinline__ std::int8_t kv_cache_unpack_i4(std::uint8_t packed, int high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

__device__ __forceinline__ void kv_cache_hadamard64(float& x0, float& x1,
                                                    unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        const bool hi  = (static_cast<int>(threadIdx.x) & offset) != 0;
        x0             = hi ? y0 - x0 : x0 + y0;
        x1             = hi ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}

template <int QHeads>
__global__ void kv_cache_inverse_rotate_output_kernel(__nv_bfloat16* output, int width,
                                                      int full_width, int column_begin,
                                                      const std::int32_t* valid_columns) {
    const int unit = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) { return; }
    const int group  = unit % kKVCacheInt8Groups;
    const int tmp    = unit / kKVCacheInt8Groups;
    const int q_head = tmp % QHeads;
    const int row    = tmp / QHeads;
    const int batch  = row / width;
    const int token  = row - batch * width;
    const int column = column_begin + token;
    if (token >= width || (valid_columns != nullptr && column >= valid_columns[batch])) { return; }
    const int d0            = group * kKVCacheInt8Group + lane;
    const int d1            = d0 + 32;
    const std::int64_t base = static_cast<std::int64_t>(kKVCacheInt8HeadDim) *
                              (q_head + static_cast<std::int64_t>(QHeads) *
                                            (column + static_cast<std::int64_t>(full_width) * batch));
    float x0 = __bfloat162float(output[base + d0]);
    float x1 = __bfloat162float(output[base + d1]);
    kv_cache_hadamard64(x0, x1);
    output[base + d0] = __float2bfloat16(x0);
    output[base + d1] = __float2bfloat16(x1);
}

__device__ __forceinline__ void kv_cache_unpack_i4x16(const std::uint8_t* src8,
                                                      std::int8_t* dst16) {
    const std::uint64_t raw = load_vec<std::uint64_t>(src8);
    const auto* bytes       = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst16[2 * i]     = kv_cache_unpack_i4(bytes[i], 0);
        dst16[2 * i + 1] = kv_cache_unpack_i4(bytes[i], 1);
    }
}

// ---------------------------------------------------------------------------
// rk8v4 value codec (sm_86 port of ninfer-all f118551f). Keys stay the rotated INT8 G64 coding;
// values are signed int4, two per byte (even dimension in the low nibble), one FP16 scale
// FP16_RNE(absmax / 7) per 32 values, codes = clamp(rint(x / represented_scale), -7, 7).
// ---------------------------------------------------------------------------
inline constexpr int kKVCacheInt4ValueGroup  = 32;
inline constexpr int kKVCacheInt4ValueGroups = kKVCacheInt8HeadDim / kKVCacheInt4ValueGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int4_value_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt4ValueGroups, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int4_value_code_index(int physical_page, int kv_head, int packed_d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim / 2, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, packed_d);
}

__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int4_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

// One warp stores one (token, kv head, 64-dimension group): lane l holds v0 = dim 64g+l and
// v1 = dim 64g+32+l, so v0 lanes form value group 2g and v1 lanes value group 2g+1. A packed
// byte holds the adjacent pair (2p, 2p+1), i.e. lanes l and l^1. Must be called by the full warp.
template <typename Geometry>
__device__ __forceinline__ void
kv_cache_rk8v4_store_value_group(std::int8_t* cache_v, __half* scale_v, int physical_page,
                                 int kv_head, int group, int page_offset, int lane, float v0,
                                 float v1) {
    constexpr unsigned FullMask = 0xffffffffu;
    const auto lo        = kv_cache_int4_quant_params(warp_max(fabsf(v0), FullMask));
    const auto hi        = kv_cache_int4_quant_params(warp_max(fabsf(v1), FullMask));
    const std::int8_t c0 = kv_cache_i4_quant_code(v0, lo.inverse_scale);
    const std::int8_t c1 = kv_cache_i4_quant_code(v1, hi.inverse_scale);
    const int partner0   = __shfl_xor_sync(FullMask, static_cast<int>(c0), 1);
    const int partner1   = __shfl_xor_sync(FullMask, static_cast<int>(c1), 1);
    if ((lane & 1) == 0) {
        auto* packed            = reinterpret_cast<std::uint8_t*>(cache_v);
        const std::int64_t base = kv_cache_int4_value_code_index<Geometry>(
            physical_page, kv_head, group * (kKVCacheInt8Group / 2), page_offset);
        packed[base + (lane >> 1)]      = kv_cache_pack_i4(c0, static_cast<std::int8_t>(partner0));
        packed[base + (lane >> 1) + 16] = kv_cache_pack_i4(c1, static_cast<std::int8_t>(partner1));
    }
    if (lane == 0) {
        scale_v[kv_cache_int4_value_scale_index<Geometry>(physical_page, kv_head, 2 * group,
                                                         page_offset)] = lo.scale;
        scale_v[kv_cache_int4_value_scale_index<Geometry>(physical_page, kv_head, 2 * group + 1,
                                                         page_offset)] = hi.scale;
    }
}

// Eight consecutive dimensions [d, d+8) from the four packed bytes that hold them, as FP16.
__device__ __forceinline__ int4 kv_cache_int4_dequant_f16x8_from(const std::uint8_t* packed4,
                                                                 float s) {
    const unsigned raw        = load_vec<unsigned>(packed4);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(kv_cache_unpack_i4(bytes[i], 0)) * s;
        const float x1 = static_cast<float>(kv_cache_unpack_i4(bytes[i], 1)) * s;
        out[i]         = pack_f16x2(x0, x1);
    }
    return make_int4(static_cast<int>(out[0]), static_cast<int>(out[1]), static_cast<int>(out[2]),
                     static_cast<int>(out[3]));
}

} // namespace ninfer::ops
