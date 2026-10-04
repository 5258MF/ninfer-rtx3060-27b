#pragma once

// int8 tensor-core rung for PQ2_0 ternary weights (roadmap item #14).
//
// WHY: the bf16 wide rung is not bandwidth bound and not occupancy bound. Measured on the real
// shape mix (tscale_bench, 2026-09-20): one pass at T=64 costs 53.9-56.5 ms against a bf16 mma floor
// of 30.6 ms, i.e. 57% of the card's 112 TFLOPS bf16 peak, while ncu reports No Eligible 77.3% and
// DRAM at 19.7%; raising occupancy to 16 warps/SM made the achieved rate WORSE (63.7 -> 48.0
// TFLOPS). So the lever is fewer instructions per unit of work, and that is what int8 buys:
// m16n8k32.s8 covers 2x the K per instruction (4 mma per 128-K chunk instead of 8) at 2x the bf16
// peak (measured 225 vs 112 TFLOP/s-equiv on this card).
//
// The recipe is the one the Prism fork uses for this exact format (read locally, not guessed):
//   E:\llama-cpp\prism-b10687\ggml\src\ggml-cuda\
//     * mmq-config-ampere.cuh:36-51  16 CASE rows for GGML_TYPE_PQ2_0, SRAM layout Q8_0.
//     * mma.cuh:946                  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.
//     * mmq-load-tiles.cuh:215-235   2-bit to int8 expansion with __byte_perm + 0x020100FF.
//     * quantize.cu                  activations quantized to int8 before the mma.
// What we deliberately do NOT copy: their 256 threads / occupancy=1 / I=128xJ / stream-K structure
// (that is llama.cpp's MMQ, and occupancy is measured to be the wrong knob here anyway), and their
// q8_1 per-32 activation scales (we use one scale per token over the whole K row, which is what
// keeps the epilogue a cvt+fma and matches this engine's existing fp8 A8 quantizer).
//
// MEASURED (real shape mix, 6.65 GiB of weights, quantization pass included):
//   T      bf16 small_t   bf16 wide_t   s8      best bf16 rung
//   16     22.19          53.88         43.29   small_t
//   24     32.92          54.95         40.15   small_t
//   32     42.49          53.79         43.25   tie
//   40     53.04          55.28         44.24   s8 by 1.20x
//   48     66.56          54.72         44.89   s8 by 1.22x
//   64     85.78          56.51         43.47   s8 by 1.30x
// Oracle: rel_l2 ~ 9.0e-3, which IS the int8 activation quantization error itself (measured
// independently at 7.4e-3), i.e. the GEMM is exact on the operands it is given. Because the
// activations are quantized, the bit-exact PPL gate no longer applies: this rung must be judged by
// the E4 criterion (6.44 <= PPL <= 6.50 and >= 19/20 on the 20-question suite).

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>  // getenv (NINFER_TERNARY_QUANT_VEC)
#include <cstdio>   // device printf, used by the TERNARY_S8_DEBUG instrumentation

namespace ninfer::ops::detail {

inline constexpr int kTernaryS8RowsPerWarp    = 16;
inline constexpr int kTernaryS8ChunkK         = PQ2RowSplitStorage::kGroupK;   // 128 == one group
inline constexpr int kTernaryS8Stages         = 2;
inline constexpr int kTernaryS8RowStride      = 48;   // packed codes: 32 used, padded for banks
inline constexpr int kTernaryS8KSteps         = kTernaryS8ChunkK / 32;         // 4 mma steps/chunk
inline constexpr int kTernaryS8SegmentsPerRow = kTernaryS8ChunkK / 16;         // int8 tile swizzle

// Raw PTQ1_0 staging row for the kPtq1 instantiation: the 24 base bytes + the 2 high bytes, padded
// to 32. The pad is what makes every row start 8-byte aligned -- the alignment the cp.async of that
// size needs -- and it is the same 32-byte row the wide-t rung uses
// (kTernaryWideRawStride), so the repack helper sees an identical source layout in both rungs.
inline constexpr int kTernaryS8RawStride = 32;

// The kPtq1 instantiation's int8 weight plane (2026-09-23, this file only): ONE single-buffered
// plane per warp holding a whole 128-weight group ALREADY DECODED TO INT8 -- the bytes the ldmatrix
// below reads. 4 warps x 16 rows x 144 B == 9216 B, which is exactly what the PQ2_0 layout spends on
// its 32-byte code plane (6144) plus its per-step expansion plane (3072), so this cannot move
// occupancy (3 CTA/SM, see the kernel header's smem argument).
//
// The 144 pitch (128 used + 16 pad) is not decoration: an ldmatrix row address must stay 16-byte
// aligned, so a plain 128 pitch would put all eight rows of one 8x16 matrix on the SAME bank
// (128/4 = 32 words, 32r mod 32 == 0). At 36 words per row the eight rows land on 4r mod 32, i.e.
// eight distinct banks, conflict-free -- the same trick the expansion plane's 48-byte pitch plays
// with 12r mod 32.
inline constexpr int kTernaryS8PlaneStride = kTernaryS8ChunkK + 16;   // 144

// 2-bit code byte -> four int8 values.
//
// TWO __byte_perm steps, and that is not decoration: a byte_perm selector nibble is an INDEX INTO
// THE EIGHT BYTES of {x,y}, not a row number of a table, and one packed code byte's nibble carries
// TWO codes (bits 0-1 and 2-3). A single call therefore only ever pulls two of the four values and
// fills the rest with a table byte -- measured, it produced [v(c0), v(c2), -1, -1], i.e. two
// spurious -1s per operand register, which failed the oracle with rel_l2 1.17 instead of ~9e-3.
//   step 1: expand the low nibble of each half  -> [v(c0), v(c2), _, _]
//   step 2: same on the >>2 view               -> [v(c1), v(c3), _, _]
//   step 3: interleave with 0x5140             -> [v(c0), v(c1), v(c2), v(c3)]
// Same shape as the Prism fork's expansion (mmq-load-tiles.cuh:215-235). The duplicated constant
// makes bit 2 of a nibble harmless and the hardware ignores bit 3.
__device__ __forceinline__ unsigned ternary_s8_expand_codes(unsigned code_byte) {
    const unsigned even = __byte_perm(0x020100FFu, 0x020100FFu, code_byte);
    const unsigned odd  = __byte_perm(0x020100FFu, 0x020100FFu, code_byte >> 2);
    return __byte_perm(even, odd, 0x5140u);
}

// XOR swizzle so the 16-byte activation rows ldmatrix pulls land on distinct banks.
__device__ __forceinline__ int ternary_s8_shared_byte(int row, int logical_byte) {
    const int segment      = logical_byte >> 4;
    const int byte_in_seg  = logical_byte & 15;
    const int physical_seg = segment ^ (row & (kTernaryS8SegmentsPerRow - 1));
    return (physical_seg << 4) | byte_in_seg;
}

// --- PTQ1_0 -> int8, direct (no PQ2_0 code relabelling anywhere in between) ---------------------
//
// WHAT THIS IS: the trit values themselves, landed in the plane the ldmatrix reads. The rung used to
// relabel PTQ1_0's base-3 bytes as PQ2_0's 2-bit codes in shared and then expand those codes to int8
// per k-step; both of those steps are gone here. The arithmetic is PrismML's
// (ggml/src/ggml-cuda/mmq-load-tiles.cuh:259 ggml_cuda_mmq_decode_ptq1_0_qs4, MIT), re-aimed from
// their int* sram tile at our int8 plane -- their `dst[t*stride]` and our `dst + t*dst_step` are the
// same one-word-per-stage store, only the stride convention differs.
//
// WHY ONE CHAIN AND NOT ONE CHAIN PER STAGE: PTQ1_0's trit for element e is (v*3)>>8 of the base-3
// remainder recurrence v = (v*3)&0xFF applied n = e's stage index times (ggml-cuda/common.cuh:995
// ptq1_0_trit -- and note it is the SAME recurrence our PTQ1SimtDecodeAtom::decode_one and the
// reader's dequantizer implement). Four consecutive weights of a 4-byte qs group share that one
// chain: stage t's product already contains stage t-1's remainder in its low byte, so five
// multiplies by 3 produce all five stages of the group where five separate chains would need
// 1+2+3+4+5. The two __byte_perm calls park the four base-3 digits in the even lanes of two
// registers (the *3 then advances all four digits at once) and the final permute gathers each
// product's HIGH byte, i.e. (v*3)>>8; __vsub4 subtracts the 1 of (xi - 1) per byte, so four weights
// come out of one instruction as four signed bytes in weight order.
//
// PROVEN, NOT ASSUMED: E:\infer-build\exp\reader-candidates\verify_native_ptq1_map.py expands real
// weights (token_embd.weight, 64 whole rows == 327,680 grid cells) through a bit-faithful model of
// these two helpers placed at exactly the offsets the kernel writes, and finds 0 mismatches against
// ggml's ptq1_0_trit, against PTQ1SimtDecodeAtom::decode_one and against the reader's dequantizer;
// the seven destination words also cover the 128-byte group exactly once, at 4-byte granularity.
//
// SLOT -> DESTINATION, i.e. the k-step table this rung's A operand is built from (weights 0-79: byte
// qs[e&15] at stage e>>4; 80-119: byte qs[16+((e-80)&7)] at stage (e-80)>>3; 120-127: qh[e&1]):
//   slot 0..3  source qs[4g..4g+4)   -> words at 16t + 4g     (dst_step 16)  weights 16t + 4g + m
//   slot 4..5  source qs[16+4p..)    -> words at 80 + 8t + 4p (dst_step 8)   weights 80 + 8t+4p+m
//   slot 6     source qh[0..1]       -> words at 120 and 124
// so group byte e IS weight e, and the four 32-byte blocks of the plane are k-steps 0..3:
//   step 0 <- slots 0-3 stages 0,1 | step 1 <- slots 0-3 stages 2,3
//   step 2 <- slots 0-3 stage 4 + slots 4-5 stages 0,1 | step 3 <- slots 4-5 stages 2,3,4 + slot 6
__device__ __forceinline__ void ternary_s8_decode_qs4(unsigned packed, std::uint8_t* dst,
                                                      int dst_step) {
    unsigned v_lo = __byte_perm(packed, 0u, 0x4140u);
    unsigned v_hi = __byte_perm(packed, 0u, 0x4342u);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const unsigned w_lo = v_lo * 3u;
        const unsigned w_hi = v_hi * 3u;
        v_lo                = w_lo & 0x00FF00FFu;
        v_hi                = w_hi & 0x00FF00FFu;
        *reinterpret_cast<unsigned*>(dst + t * dst_step) =
            __vsub4(__byte_perm(w_lo, w_hi, 0x7531u), 0x01010101u);
    }
}

// The 2-byte qh tail: weights 120 + n*2 + h read qh[h] at stage n, so one word carries stages
// (2t, 2t+1) of both bytes: [h0@2t, h1@2t, h0@2t+1, h1@2t+1].
__device__ __forceinline__ void ternary_s8_decode_qh(const std::uint8_t* qh, std::uint8_t* dst) {
    unsigned v = static_cast<unsigned>(qh[0]) | (static_cast<unsigned>(qh[1]) << 16);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const unsigned w0 = v * 3u;
        v                 = w0 & 0x00FF00FFu;
        const unsigned w1 = v * 3u;
        v                 = w1 & 0x00FF00FFu;
        *reinterpret_cast<unsigned*>(dst + 120 + 4 * t) =
            __vsub4(__byte_perm(w0, w1, 0x7531u), 0x01010101u);
    }
}

// Per-token symmetric int8 quantization of one activation row (token-major layout: x[token*k + i]).
// One block per token; the absmax over the whole K row is reduced first, then a second pass writes
// the codes, so the scale is already known when they are produced.
__global__ __launch_bounds__(256, 2) void ternary_s8_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ codes,
    float* __restrict__ scales, std::int32_t k) {
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tid   = static_cast<std::int32_t>(threadIdx.x);
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(token) * k;

    __shared__ float warp_max[8];
    float local = 0.0f;
    for (std::int32_t i = tid; i < k; i += blockDim.x) {
        local = fmaxf(local, fabsf(__bfloat162float(row[i])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
    }
    if ((tid & 31) == 0) { warp_max[tid >> 5] = local; }
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int w = 0; w < (static_cast<int>(blockDim.x) >> 5); ++w) { m = fmaxf(m, warp_max[w]); }
        warp_max[0] = m;
    }
    __syncthreads();
    const float absmax = warp_max[0];
    const float scale  = absmax > 0.0f ? absmax / 127.0f : 1.0f;   // never divide by zero
    if (tid == 0) { scales[token] = scale; }

    std::int8_t* dst     = codes + static_cast<std::int64_t>(token) * k;
    const float inverse  = 1.0f / scale;
    for (std::int32_t i = tid; i < k; i += blockDim.x) {
        const int q = __float2int_rn(__bfloat162float(row[i]) * inverse);
        dst[i] = static_cast<std::int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
}

// Vectorized twin of ternary_s8_quantize_kernel (2026-09-28, RTX 3060 prefill work). Same block
// shape (one 256-thread block per token); each thread keeps up to kTernaryQzVecs 16-byte vectors
// (8 bf16) of the row in registers between the absmax pass and the code pass, so the row is read
// once, and codes leave as 8-byte packets. absmax is a max (order-independent) and every code uses
// the reference expression verbatim => codes AND scales are byte-identical to the scalar kernel
// (kbench/qz.cu oracle: EXACT on K 100/4104/5120/5128/6144/17408 x T 17/64/256/1024, zero rows and
// outliers included). Measured T=1024 per forward (288 calls): 40.7 -> 22.8 ms, ~305-322 GB/s.
// Preconditions (else the scalar kernel runs): k % 8 == 0, k <= 256*8*kTernaryQzVecs, x 16-byte
// aligned, codes 8-byte aligned. NINFER_TERNARY_QUANT_VEC=0 restores the scalar kernel.
constexpr int kTernaryQzThreads = 256;
constexpr int kTernaryQzVecs    = 9;   // 256 * 8 * 9 = 18432 >= 17408 (mlp_down K)

__device__ __forceinline__ std::int8_t ternary_qz_code(float v, float inverse) {
    const int q = __float2int_rn(v * inverse);
    return static_cast<std::int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
}

__global__ __launch_bounds__(kTernaryQzThreads) void ternary_s8_quantize_vec_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ codes,
    float* __restrict__ scales, std::int32_t k) {
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tid   = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t nvec  = k >> 3;
    const uint4* row = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * k);

    uint4 v[kTernaryQzVecs];
    float local = 0.0f;
#pragma unroll
    for (int j = 0; j < kTernaryQzVecs; ++j) {
        const std::int32_t i = tid + j * kTernaryQzThreads;
        if (i < nvec) {
            v[j] = __ldg(row + i);
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[j]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(h[e]);
                local = fmaxf(local, fabsf(f.x));
                local = fmaxf(local, fabsf(f.y));
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
    }
    __shared__ float warp_max[kTernaryQzThreads / 32];
    if ((tid & 31) == 0) { warp_max[tid >> 5] = local; }
    __syncthreads();
    float absmax = warp_max[0];
#pragma unroll
    for (int w = 1; w < kTernaryQzThreads / 32; ++w) { absmax = fmaxf(absmax, warp_max[w]); }
    const float scale   = absmax > 0.0f ? absmax / 127.0f : 1.0f;   // never divide by zero
    if (tid == 0) { scales[token] = scale; }
    const float inverse = 1.0f / scale;

    uint2* dst = reinterpret_cast<uint2*>(codes + static_cast<std::int64_t>(token) * k);
#pragma unroll
    for (int j = 0; j < kTernaryQzVecs; ++j) {
        const std::int32_t i = tid + j * kTernaryQzThreads;
        if (i < nvec) {
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[j]);
            unsigned packed[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float2 a = __bfloat1622float2(h[2 * half]);
                const float2 b = __bfloat1622float2(h[2 * half + 1]);
                const unsigned c0 = static_cast<unsigned char>(ternary_qz_code(a.x, inverse));
                const unsigned c1 = static_cast<unsigned char>(ternary_qz_code(a.y, inverse));
                const unsigned c2 = static_cast<unsigned char>(ternary_qz_code(b.x, inverse));
                const unsigned c3 = static_cast<unsigned char>(ternary_qz_code(b.y, inverse));
                packed[half] = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
            }
            dst[i] = make_uint2(packed[0], packed[1]);
        }
    }
}

[[nodiscard]] inline bool ternary_quant_vec_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_QUANT_VEC");
        return value == nullptr || !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// The one entry point both int8 rungs use for their activation pass.
inline void launch_ternary_s8_quantize(const __nv_bfloat16* x, std::int8_t* codes, float* scales,
                                       std::int32_t tokens, std::int32_t k, cudaStream_t stream) {
    const bool vec_ok = k > 0 && (k & 7) == 0 && k <= kTernaryQzThreads * 8 * kTernaryQzVecs &&
                        (reinterpret_cast<std::uintptr_t>(x) & 15u) == 0 &&
                        (reinterpret_cast<std::uintptr_t>(codes) & 7u) == 0;
    if (vec_ok && ternary_quant_vec_enabled()) {
        ternary_s8_quantize_vec_kernel<<<tokens, kTernaryQzThreads, 0, stream>>>(x, codes, scales, k);
    } else {
        ternary_s8_quantize_kernel<<<tokens, 256, 0, stream>>>(x, codes, scales, k);
    }
}

// Raw PTQ1_0 staging plane, present ONLY in the kPtq1 instantiation. An empty base is used instead
// of a zero-length array (not standard C++) or a 1-byte dummy member (which would move the layout):
// empty base optimisation makes TernaryS8RawPlane<false, ...> contribute exactly nothing, so the
// PQ2_0 instantiation's shared footprint and member offsets are literally unchanged.
template <bool kPtq1_, int Tokens_, int Warps_>
struct TernaryS8RawPlane {};

template <int Tokens_, int Warps_>
struct TernaryS8RawPlane<true, Tokens_, Warps_> {
    alignas(16) std::uint8_t raw[kTernaryS8Stages][Warps_][kTernaryS8RowsPerWarp]
                                [kTernaryS8RawStride];
};

// The trailing `bool kPtq1 = false` is defaulted on purpose: every existing instantiation of this
// storage (and of the kernel below) keeps compiling and keeps meaning PQ2_0, so this header can land
// ahead of the dispatch change without breaking an in-flight build. NOTE the parameter ORDER for the
// PTQ1_0 instantiation: ternary_pq2_mma_s8_kernel<64, 4, 3, true>.
//
// kPtq1 == false is the layout as first shipped, member for member and byte for byte: a two-stage
// 48-byte-pitch code plane (6144 B) plus the per-step expansion plane (3072 B). kPtq1 == true
// collapses both into the single decoded int8 plane (9216 B, plane pitch 144): the direct decode
// writes int8 straight into it and the ldmatrix reads it, so neither a 32-byte code row nor an
// expansion row exists on that instantiation. The three constexpr shape selectors below are the
// whole of that difference -- for kPtq1 == false each one reduces to the literal it replaced, so
// ptxas sees the shipped PQ2_0 program (checked: 159 regs / 25856 B / 0 spill, unchanged).
template <int Tokens_, int Warps_, bool kPtq1 = false>
struct TernaryS8Storage : TernaryS8RawPlane<kPtq1, Tokens_, Warps_> {
    static constexpr int kTokens     = Tokens_;
    static constexpr int kWarps      = Warps_;
    static constexpr int kRowsPerCta = kTernaryS8RowsPerWarp * kWarps;
    static constexpr int kThreads    = kWarps * 32;

    static constexpr int kCodeStages = kPtq1 ? 1 : kTernaryS8Stages;
    static constexpr int kCodePitch  = kPtq1 ? kTernaryS8PlaneStride : kTernaryS8RowStride;
    // The expansion plane exists only for PQ2_0; kPtq1 keeps a 16-byte stub so the two member
    // bindings at the top of the kernel stay valid without a conditional declaration (the code that
    // indexes it is inside `if constexpr (!kPtq1)` and is never instantiated here). 16 B against
    // 29952 B of shared is 0.05% -- it cannot move occupancy.
    static constexpr int kExpWarps  = kPtq1 ? 1 : Warps_;
    static constexpr int kExpRows   = kPtq1 ? 1 : kTernaryS8RowsPerWarp;
    static constexpr int kExpPitch  = kPtq1 ? 16 : kTernaryS8RowStride;

    alignas(16) std::uint8_t codes[kCodeStages][kWarps][kTernaryS8RowsPerWarp][kCodePitch];
    std::uint16_t wscales[kTernaryS8Stages][kWarps][kTernaryS8RowsPerWarp];
    alignas(16) std::int8_t act[kTernaryS8Stages][kTokens][kTernaryS8ChunkK];
    // Per-warp expansion of ONE 32-K step (16 rows x 32 used bytes, 48-byte stride). The A operand
    // is then built with ldmatrix_x4, the shape the engine's fp8 mma kernel already ships
    // (ops/linear/fp8/fp8_a8_mma.cuh). Hand-placing the four int8 register groups without ldmatrix
    // was tried first and produced the wrong register-to-row mapping.
    alignas(16) std::int8_t wexp[kExpWarps][kExpRows][kExpPitch];
};

// A = ternary weights (16 rows x 32 K int8), B = activations (32 K x 8 tokens int8), C = int32
// accumulator folded into fp32 with the weight group scale; the per-token activation scale is
// applied once at the store, which is what keeps the per-chunk fold down to cvt+fma.
//
// kPtq1 == false: `w_codes` is the PQ2_0 code plane (32 bytes per row per group, positional) and
//                 `w_high` is unused. UNCHANGED and bit-identical to the rung as first shipped: the
//                 template parameter is trailing and defaulted, and the PTQ1_0 staging below is a
//                 discarded `if constexpr` branch, so ptxas sees the same program.
// kPtq1 == true : `w_codes` is the PTQ1_0 base plane (24 base-3 bytes) and `w_high` its 2-byte high
//                 plane. The 24+2-byte row is cp.async'd RAW into a 32-byte shared row one chunk
//                 ahead of use, then DECODED DIRECTLY TO INT8 after cp_wait (ternary_s8_decode_qs4 --
//                 no PTQ1_0 -> PQ2_0 code relabelling and no 2-bit expansion anywhere on this rung;
//                 see the decode helpers above for the slot -> offset table and its verification).
//                 From the ldmatrix down everything is shared with the PQ2_0 path: the same int8
//                 plane rows, the same ldmatrix, the same mma, the same scale fold and epilogue.
//                 The point is not the 17.6% smaller artifact (this rung is instruction limited, not
//                 bandwidth limited) but the 1.20-1.30x the s8 operands buy over bf16 wide_t, which
//                 PTQ1_0 could not take at all until this instantiation existed -- and which the
//                 repack this replaced threw away again (measured 2026-09-23: 55.37 ms/forward
//                 against PQ2_0's 39.75 at T=64, i.e. the repack was this rung's first-order cost).
//
// STAGING WIDTH, DERIVED NOT GUESSED (asked, because PQ2_0 stages in 16-byte copies and 24 is not a
// multiple of 16): the copy is THREE 8-BYTE cp.async per row, not 16+8.
//   * Source alignment: the per-chunk stride is exactly 24 bytes, so chunk c starts at 24c; 24c % 16
//     = 8 * (3c mod 2), i.e. 0 for even chunks and 8 for odd ones. An odd chunk is therefore 8-byte
//     but NOT 16-byte aligned, and a 16-byte cp.async there is a misaligned copy -- the same finding
//     (and the same 8-byte-granular answer) the wide-t rung records.
//   * 8-byte alignment always holds: a row pitch is groups_per_row * 24 and 24 % 8 == 0, so the row
//     base is 8-aligned for any groups_per_row; the chunk offset is a multiple of 8 by construction.
//   * Destination: raw rows have a 32-byte pitch from an alignas(16) base, and the three in-row
//     offsets are 0/8/16, so every 8-byte store target is 8-aligned.
//   * 4-byte copies were rejected: same byte count as 8-byte here (24 = 6 x 4) for twice the
//     instructions, and the copy loop is per-warp (16 rows) so there is no occupancy to gain.
// Copy count per warp per chunk: 16 rows x 3 = 48 copies of 8 bytes = 384 B -- against the PQ2_0
// path's 16 rows x 2 copies of 16 bytes = 512 B, which IS the 17.6% (384/512 = 0.75 in the base
// plane; the whole-format ratio is 28/34 = 0.824).
template <int Tokens_, int Warps_, int MinBlocks_, bool kPtq1 = false>
__global__ __launch_bounds__(Warps_ * 32, MinBlocks_)
void ternary_pq2_mma_s8_kernel(const std::int8_t* __restrict__ act_codes,
                               const float* __restrict__ act_scales,
                               const std::uint8_t* __restrict__ w_codes,
                               const std::uint8_t* __restrict__ w_scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t k, std::int32_t tokens,
                               std::int32_t out_row_stride,
                               // PTQ1_0 high plane; unused (and unread) when kPtq1 == false, hence
                               // the default so the PQ2_0 call site does not have to name it.
                               const std::uint8_t* __restrict__ w_high = nullptr,
                               // SCHED3 CHANGE 3 on the int8 rung; see the long note in
                               // ternary_rowsplit_mma_small_t.cuh. The token tile is grid.y when
                               // true, and the old serial walk when false.
                               bool token_grid = false) {
    using Storage = TernaryS8Storage<Tokens_, Warps_, kPtq1>;
    static_assert(Tokens_ % 8 == 0, "token tile must be whole 8-token mma column groups");
    static_assert(Tokens_ <= 64, "the activation tile is sized for one wide token tile");

    constexpr int kTokens     = Tokens_;
    constexpr int kWarps      = Warps_;
    constexpr int kThreads    = kWarps * 32;
    constexpr int kSubTiles   = kTokens / 8;
    constexpr int kRowsPerCta = kTernaryS8RowsPerWarp * kWarps;

    __shared__ Storage staging;
    auto& codes_sh  = staging.codes;
    auto& wscale_sh = staging.wscales;
    auto& act_sh    = staging.act;
    auto& wexp_sh   = staging.wexp;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;
    if (row0 >= rows) { return; }

    const std::int32_t groups_per_row = k / PQ2RowSplitStorage::kGroupK;
    // Both planes are group-strided; only the widths differ (32/0 vs 24/2).
    constexpr int kCodeBytes = kPtq1 ? PTQ1RowSplitStorage::kCodeBytesPerGroup
                                     : PQ2RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kHighBytes = kPtq1 ? PTQ1RowSplitStorage::kHighBytesPerGroup : 0;
    static_assert(kCodeBytes == kTernaryS8ChunkK / 4 || kPtq1,
                  "the PQ2_0 code plane IS the 32-byte window this kernel reads");
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kCodeBytes;
    const std::int64_t high_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kHighBytes;
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * PQ2RowSplitStorage::kScaleBytesPerGroup;
    const std::int32_t chunks = k / kTernaryS8ChunkK;
    const std::int64_t warp_row_base =
        static_cast<std::int64_t>(row0) + static_cast<std::int64_t>(warp) * kTernaryS8RowsPerWarp;

    const auto stage_weights = [&](int buf, int chunk) {
        if constexpr (kPtq1) {
            // PREFETCH ONLY: the raw 24+2-byte rows, one chunk ahead. The repack is a separate step
            // (see repack_ptq1 below) because folding it into this loop would put the global read
            // latency directly on top of the dependent repack arithmetic -- LDG -> byte_perm chain
            // -> shared store -- which is exactly the ordering the wide-t rung was rewritten to
            // remove. Here the global read is an async copy issued one chunk ahead and hidden behind
            // the previous chunk's mma, and the repack reads SHARED.
            //
            // THREE 8-byte copies, not 16+8: an odd chunk's qs starts at 24c with c odd, i.e. 8 mod
            // 16, so a 16-byte cp.async is misaligned there. See the kernel header for the full
            // alignment derivation. 8-byte alignment of both sides always holds (24 % 8 == 0 on the
            // source pitch, 32-byte shared rows from an alignas(16) base on the destination).
            constexpr int kQsCopies     = PTQ1RowSplitStorage::kCodeBytesPerGroup / 8; // 3
            constexpr int kQsCopiesWarp = kTernaryS8RowsPerWarp * kQsCopies;           // 48
            static_assert(PTQ1RowSplitStorage::kCodeBytesPerGroup % 8 == 0,
                          "a PTQ1_0 row copy is 8-byte granular, which is what cp.async needs");
            static_assert(kTernaryS8RawStride >= PTQ1RowSplitStorage::kCodeBytesPerGroup,
                          "the raw row must hold a whole qs");
            static_assert((kTernaryS8RawStride % 8) == 0,
                          "every raw row start must be 8-byte aligned for the copy above");
#pragma unroll
            for (int copy = lane; copy < kQsCopiesWarp; copy += 32) {
                const int local_row           = copy / kQsCopies;
                const int byte_off            = (copy % kQsCopies) * 8;
                const std::int64_t global_row = warp_row_base + local_row;
                const bool valid              = global_row < rows;
                // An invalid row zero-fills (src_bytes = 0) rather than branching to a shared
                // store: a zeroed raw row repacks to a zeroed code row by construction, so the
                // repack needs no validity test at all (wide-t's argument, and this rung's codes
                // were already zero-filled for OOB rows on the PQ2_0 path).
                cp_async_zfill<8, Cache::ca>(
                    &staging.raw[buf][warp][local_row][byte_off],
                    valid ? w_codes + global_row * code_row_bytes +
                                static_cast<std::int64_t>(chunk) * kCodeBytes + byte_off
                          : w_codes,
                    valid ? 8 : 0);
            }
            // qh is 2 bytes, so it is the one part that must NOT go through cp.async: 4 bytes is the
            // smallest copy and it would read past the end of the high plane on the last row's last
            // group. One plain load + one plain shared store per row per chunk, lane < 16.
            if (lane < kTernaryS8RowsPerWarp) {
                const std::int64_t global_row = warp_row_base + lane;
                const std::uint16_t qh =
                    (global_row < rows)
                        ? load_vec<std::uint16_t>(w_high + global_row * high_row_bytes +
                                                  static_cast<std::int64_t>(chunk) * kHighBytes)
                        : 0u;
                // qs is 24 bytes and the raw row is 32, so this 2-byte store is naturally aligned.
                store_vec<std::uint16_t>(
                    reinterpret_cast<std::uint16_t*>(
                        &staging.raw[buf][warp][lane][PTQ1RowSplitStorage::kCodeBytesPerGroup]),
                    qh);
            }
        } else {
            constexpr int kCopiesPerRow  = (kTernaryS8ChunkK / 4) / 16;
            constexpr int kCopiesPerWarp = kTernaryS8RowsPerWarp * kCopiesPerRow;
#pragma unroll
            for (int copy = lane; copy < kCopiesPerWarp; copy += 32) {
                const int local_row = copy / kCopiesPerRow;
                const int byte_off  = (copy % kCopiesPerRow) * 16;
                const std::int64_t global_row = warp_row_base + local_row;
                std::uint8_t* dst = &codes_sh[buf][warp][local_row][byte_off];
                if (global_row < rows) {
                    cp_async<16, Cache::cg>(dst, w_codes + global_row * code_row_bytes +
                                                     static_cast<std::int64_t>(chunk) * kCodeBytes +
                                                     byte_off);
                } else {
                    store_vec<std::uint8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
                }
            }
        }
        if (lane < kTernaryS8RowsPerWarp) {
            const std::int64_t global_row = warp_row_base + lane;
            std::uint16_t value = 0u;
            if (global_row < rows) {
                value = load_vec<std::uint16_t>(
                    w_scales + global_row * scale_row_bytes +
                    static_cast<std::int64_t>(chunk) * PQ2RowSplitStorage::kScaleBytesPerGroup);
            }
            wscale_sh[buf][warp][lane] = value;
        }
    };

    // DIRECT DECODE (kPtq1 only): the raw 24+2-byte rows that cp_wait just landed -> the int8
    // weight plane the ldmatrix reads, in shared, after cp_wait. 16 rows x 7 slots = 112 jobs.
    //
    // THREE UNIFORM LOOPS, not one flat job/7 loop. With 7 slots over 32 lanes a flat loop puts
    // lanes of different slots in the same instruction, so its three branches serialise (the
    // small-t rung hit this and split it; the repack this replaces paid it -- ~4 unrolled rounds x 3
    // serialised branches). Each loop below is divergence-free, and each lane's destination offsets
    // are compile-time immediates off one base register:
    //   loop A  qs[0..16) words x 16 rows = 64 jobs, lane = row*4 + slot, two rounds of 32
    //   loop B  qs[16..24) words x 16 rows = 32 jobs, lane = row*2 + parity, one round
    //   loop C  the 2-byte qh tail x 16 rows = 16 jobs, one round on lanes 0..15
    // Warp-local (this warp owns both the raw rows and its own plane rows), hence a __syncwarp and
    // NOT a CTA barrier -- the same reasoning the repack carried: it is on the mma's critical path.
    const auto decode_ptq1 = [&](int buf) {
        if constexpr (kPtq1) {
            const int row_a = lane >> 2;
            const int slot  = lane & 3;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = row_a + half * 8;
                ternary_s8_decode_qs4(
                    load_vec<unsigned>(&staging.raw[buf][warp][row][4 * slot]),
                    &codes_sh[0][warp][row][4 * slot], 16);
            }
            {
                const int row = lane >> 1;
                const int par = lane & 1;
                ternary_s8_decode_qs4(
                    load_vec<unsigned>(&staging.raw[buf][warp][row][16 + 4 * par]),
                    &codes_sh[0][warp][row][80 + 4 * par], 8);
            }
            if (lane < kTernaryS8RowsPerWarp) {
                ternary_s8_decode_qh(
                    &staging.raw[buf][warp][lane][PTQ1RowSplitStorage::kCodeBytesPerGroup],
                    &codes_sh[0][warp][lane][0]);
            }
        }
    };

    // int8 activation tile: kTokens rows of kTernaryS8ChunkK bytes, 16 bytes per copy, swizzled.
    const auto stage_act = [&](int buf, int chunk, int tok_base) {
        constexpr int kCopiesPerToken = kTernaryS8SegmentsPerRow;
        constexpr int kCopies         = kTokens * kCopiesPerToken;
        for (int copy = tid; copy < kCopies; copy += kThreads) {
            const int row        = copy / kCopiesPerToken;
            const int segment    = copy - row * kCopiesPerToken;
            const int logical    = segment * 16;
            const int physical   = ternary_s8_shared_byte(row, logical);
            const int global_tok = tok_base + row;
            std::int8_t* dst     = &act_sh[buf][row][physical];
            if (global_tok < tokens) {
                cp_async<16, Cache::cg>(dst, act_codes + static_cast<std::int64_t>(global_tok) * k +
                                                 static_cast<std::int64_t>(chunk) * kTernaryS8ChunkK +
                                                 logical);
            } else {
                // Zero-filled: the mma still runs, and the guarded store drops the result.
                store_vec<std::int8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }
    };

    const int row_lo = row0 + warp * kTernaryS8RowsPerWarp + gid;
    const int row_hi = row_lo + 8;
    const int b_row  = lane & 7;                 // activation row (token) for ldmatrix matrix 0
    const int b_col  = ((lane >> 3) & 1) * 16;

    // SCHED3 CHANGE 3: the token tile is grid.y when the switch is on (one tile per CTA); with
    // NINFER_TERNARY_TOKEN_GRID=0 the tiles are walked serially here, exactly as before. Per-tile
    // state is declared inside the loop, so the tiles are independent and this is a
    // re-partitioning of work, not a numerics change.
    const int tok_tiles = token_grid ? 1 : ((tokens + kTokens - 1) / kTokens);
    const int tok_base0 = token_grid ? static_cast<int>(blockIdx.y) * kTokens : 0;
    if (tok_base0 >= tokens) { return; }

    for (int tok_tile = 0; tok_tile < tok_tiles; ++tok_tile) {
        const int tok_base = tok_base0 + tok_tile * kTokens;
        float acc[kSubTiles][4];
#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[sub][i] = 0.0f; }
        }

        stage_weights(0, 0);
        stage_act(0, 0, tok_base);
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int buf = chunk & 1;
            if (chunk + 1 < chunks) {
                stage_weights(buf ^ 1, chunk + 1);
                stage_act(buf ^ 1, chunk + 1, tok_base);
                cp_commit();
            }
            // PTQ1_0 only: turn the raw 24+2-byte rows that cp_wait just landed into the int8 bytes
            // the ldmatrix below reads. Semantically the same slot the repack this replaced occupied;
            // here it needs no __syncwarp()-per-step aftermath because there is no second plane to
            // read back. Warp-local, hence a __syncwarp and NOT a CTA barrier -- the loop is on the
            // mma's critical path.
            if constexpr (kPtq1) {
                decode_ptq1(buf);
                __syncwarp();
            }

            int tmp[kSubTiles][4];
#pragma unroll
            for (int sub = 0; sub < kSubTiles; ++sub) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { tmp[sub][i] = 0; }
            }

#pragma unroll
            for (int step = 0; step < kTernaryS8KSteps; ++step) {
                // PQ2_0: expand this step's codes to int8 in shared. Per warp, so __syncwarp
                // suffices: 32 lanes x 4 packed bytes == 128 bytes == the whole 16-row step.
                // kPtq1 : NOTHING TO DO. The plane already holds the group in int8 (decode_ptq1
                //         wrote exactly these bytes), so the ldmatrix below simply reads this step's
                //         32 columns -- the 2-bit expansion and its round trip through the 32-byte
                //         code plane are what the direct decode removes.
                if constexpr (!kPtq1) {
                    const int word_off = step * 8;   // 32 K == 8 packed bytes per row
                    const int row4  = lane >> 1;
                    const int half8 = (lane & 1) * 4;
                    const std::uint8_t* src = &codes_sh[buf][warp][row4][word_off + half8];
                    std::int8_t* dst = &wexp_sh[warp][row4][half8 * 4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        *reinterpret_cast<unsigned*>(dst + j * 4) =
                            ternary_s8_expand_codes(static_cast<unsigned>(src[j]));
                    }
                }
                __syncwarp();

                // A operand via ldmatrix_x4: same address pattern as the shipped fp8 mma kernel.
                const int a_matrix = lane >> 3;
                const int a_row    = (lane & 7) + ((a_matrix & 1) << 3);
                const int a_col    = (a_matrix >> 1) * 16;
                unsigned a0 = 0u;
                unsigned a1 = 0u;
                unsigned a2 = 0u;
                unsigned a3 = 0u;
                if constexpr (kPtq1) {
                    ldmatrix_x4(a0, a1, a2, a3,
                                smem_addr(&codes_sh[0][warp][a_row][step * (kTernaryS8ChunkK /
                                                                        kTernaryS8KSteps) + a_col]));
                } else {
                    ldmatrix_x4(a0, a1, a2, a3, smem_addr(&wexp_sh[warp][a_row][a_col]));
                }

#pragma unroll
                for (int sub = 0; sub < kSubTiles; ++sub) {
                    // B operand: int8 activation rows, 32 K per step, already n-major in shared.
                    unsigned bf0 = 0u;
                    unsigned bf1 = 0u;
                    const int row     = sub * 8 + b_row;
                    const int logical = step * 32 + b_col;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&act_sh[buf][row][ternary_s8_shared_byte(row, logical)]));
                    mma_s8(tmp[sub][0], tmp[sub][1], tmp[sub][2], tmp[sub][3], a0, a1, a2, a3, bf0,
                           bf1);
                }
            }

            // One group per chunk => one weight scale per row; the per-token activation scale is
            // applied once at the store.
            const float w_top = __half2float(__ushort_as_half(wscale_sh[buf][warp][gid]));
            const float w_bot = __half2float(__ushort_as_half(wscale_sh[buf][warp][gid + 8]));
#pragma unroll
            for (int sub = 0; sub < kSubTiles; ++sub) {
                acc[sub][0] = fmaf(__int2float_rn(tmp[sub][0]), w_top, acc[sub][0]);
                acc[sub][1] = fmaf(__int2float_rn(tmp[sub][1]), w_top, acc[sub][1]);
                acc[sub][2] = fmaf(__int2float_rn(tmp[sub][2]), w_bot, acc[sub][2]);
                acc[sub][3] = fmaf(__int2float_rn(tmp[sub][3]), w_bot, acc[sub][3]);
            }

            __syncthreads();   // every read of the single-buffered tiles is done
            if (chunk + 1 < chunks) {
                cp_wait<0>();
                __syncthreads();
            }
        }

#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
            const int token_a = tok_base + sub * 8 + 2 * lid;
            const int token_b = token_a + 1;
            const float a_left  = (token_a < tokens) ? act_scales[token_a] : 0.0f;
            const float a_right = (token_b < tokens) ? act_scales[token_b] : 0.0f;
            if (row_lo < rows) {
                if (token_a < tokens) {
                    out[static_cast<std::int64_t>(token_a) * out_row_stride + row_lo] =
                        __float2bfloat16_rn(acc[sub][0] * a_left);
                }
                if (token_b < tokens) {
                    out[static_cast<std::int64_t>(token_b) * out_row_stride + row_lo] =
                        __float2bfloat16_rn(acc[sub][1] * a_right);
                }
            }
            if (row_hi < rows) {
                if (token_a < tokens) {
                    out[static_cast<std::int64_t>(token_a) * out_row_stride + row_hi] =
                        __float2bfloat16_rn(acc[sub][2] * a_left);
                }
                if (token_b < tokens) {
                    out[static_cast<std::int64_t>(token_b) * out_row_stride + row_hi] =
                        __float2bfloat16_rn(acc[sub][3] * a_right);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
