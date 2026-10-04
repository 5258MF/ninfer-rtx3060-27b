#pragma once

// Device kernels for the folded (rotated-basis) ternary transform.
//
// Split out of ternary_rotation.cu so a standalone nvcc harness can compile and exercise the
// REAL kernels against a numpy oracle -- the transform is the one piece of this port that cannot
// be validated by byte or size checks, and a wrong sign row, a wrong register span or a wrong
// permutation all produce fluent-looking nonsense rather than an error.
//
// This header is self-contained on purpose: it includes only cuda_runtime.h, cuda_bf16.h and the
// in-tree D256 building block (which itself only needs cuda_runtime.h), so the harness compiles it
// without dragging in Tensor/Weight/arena.

#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlockSize      = 1024;
constexpr int kWarpsPerBlock  = 8;
constexpr int kThreadsPerWarp = 32;

// D1024 = 32 lanes x 32 registers, i.e. exactly the in-tree D256 construction
// (src/ops/kv_cache/hadamard_d256.cuh: 32 lanes x 8 registers, 3 register spans, 2^-4) with two
// more register spans and 2^-5 = 1/sqrt(1024) as the normalizer. hadamard_d32_columns_inplace is
// already templated on the register count, so the lane stages are reused verbatim rather than
// copied -- the butterfly convention (low = x + y, high = x - y) therefore cannot drift.
__device__ __forceinline__ void normalized_hadamard_d1024_inplace(float (&values)[32], int lane) {
    hadamard_d32_columns_inplace(values, lane);

#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 32; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-5f); }
}

// One warp owns one (1024-block, token) pair of the [k, tokens] activation. Lane l carries
// dimensions l + 32*r of the block in values[r].
//
// LAYOUT: ninfer follows the ggml convention where ne[0] is the CONTIGUOUS axis, so an activation
// tensor with ne = (k, tokens) stores element (column, token) at token * k + column -- TOKEN-major,
// not row-major. Applying the transform with a k * tokens + t stride scrambles every prefill while
// still passing any test at tokens == 1, where the two layouts coincide exactly.
//
// Forward (the folded weights read the activation):
//     out = H * (s * (P * x))
// The permutation P is a pure index permutation and is folded into the load. Given the post-P
// column j, the source column is recovered as
//     hd = j % perm_hd, q = j / perm_hd, nk = q / perm_rep, rep = q % perm_rep
//     i  = hd + perm_hd * nk + perm_hd * perm_nk * rep
// which is the inverse of llama-graph.cpp's reshape(perm_hd, perm_nk, perm_rep) + permute(0,2,1).
__global__ void ternary_rotate_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ out,
                                           const float* __restrict__ signs, int n_blk, int k,
                                           int tokens, int perm_hd, int perm_nk, int perm_rep,
                                           int inverse) {
    const int lane   = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp   = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const int blocks = k >> 10;
    if (warp >= blocks * tokens) { return; }
    const int block = warp % blocks;
    const int token = warp / blocks;

    const float* const signs_row  = signs + (block % n_blk) * kBlockSize;
    const bool permuted           = !inverse && perm_rep > 1;
    const __nv_bfloat16* x_token  = x + static_cast<std::int64_t>(token) * k;
    __nv_bfloat16* out_token      = out + static_cast<std::int64_t>(token) * k;

    float values[32];

    if (inverse) {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            values[r] = __bfloat162float(x_token[(block << 10) + r * 32 + lane]);
        }
    } else {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const int column = (block << 10) + r * 32 + lane;
            int source       = column;
            if (permuted) {
                const int hd = column % perm_hd;
                const int q  = column / perm_hd;
                source = hd + perm_hd * (q / perm_rep) + perm_hd * perm_nk * (q % perm_rep);
            }
            const float value = __bfloat162float(x_token[source]);
            values[r]         = __fmul_rn(value, signs_row[r * 32 + lane]);
        }
    }

    normalized_hadamard_d1024_inplace(values, lane);

    if (inverse) {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const float value = __fmul_rn(values[r], signs_row[r * 32 + lane]);
            out_token[(block << 10) + r * 32 + lane] = __float2bfloat16_rn(value);
        }
    } else {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            out_token[(block << 10) + r * 32 + lane] = __float2bfloat16_rn(values[r]);
        }
    }
}

// Inverse mapping of a freshly gathered embedding, in place, over the layout the embedding op
// ACTUALLY writes.
//
// Every embed_gather_* kernel in this tree indexes `out + token * hidden + k` -- the output buffer
// is TOKEN-major, even though the tensor's extents read [hidden, T]. Transforming it with the
// [hidden, T] stride (k * T + t) would scramble every prefill: at T == 1 the two layouts coincide
// exactly, so a decode-only test cannot see the difference, and a whole prefill window comes out
// as noise (which reads as a perplexity close to uniform).
//
// Single pointer, so there is no restrict-aliasing contract to violate: each warp loads its 32
// registers, transforms them, and writes the same 32 slots back.
__global__ void ternary_rotate_inverse_inplace_bf16_kernel(__nv_bfloat16* __restrict__ data,
                                                           const float* __restrict__ signs,
                                                           int n_blk, int k, int tokens) {
    const int lane   = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp   = static_cast<int>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const int blocks = k >> 10;
    if (warp >= blocks * tokens) { return; }
    const int block = warp % blocks;
    const int token = warp / blocks;

    const float* const signs_row = signs + (block % n_blk) * kBlockSize;
    __nv_bfloat16* const base    = data + static_cast<std::int64_t>(token) * k;

    float values[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
        values[r] = __bfloat162float(base[(block << 10) + r * 32 + lane]);
    }

    normalized_hadamard_d1024_inplace(values, lane);

#pragma unroll
    for (int r = 0; r < 32; ++r) {
        const float value = __fmul_rn(values[r], signs_row[r * 32 + lane]);
        base[(block << 10) + r * 32 + lane] = __float2bfloat16_rn(value);
    }
}

} // namespace
} // namespace ninfer::ops::detail
