#pragma once

// Ternary PQ2_0 tensor-core path for the small-T regime (T = 2..8) -- which is exactly the
// speculative VERIFY pass, T = draft + 1.
//
// WHY THIS EXISTS (measured, not assumed)
// The SIMT tile GEMV is INSTRUCTION-ISSUE bound at T > 1, not bandwidth bound. Its register
// footprint is 36 (tile<4>) against 40 for the T=1 GEMV, so occupancy is NOT the constraint --
// the handover's "extra accumulators cost registers and dropped the resident warp count" story
// does not hold. What actually scales with T is work per weight: one FMA per weight PER TOKEN,
// plus two activation loads and a bf16->f32 convert per token. Measured with a synthetic PQ2_0
// payload (E:/infer-build/tscale_bench.cu), on N=248320 K=5120:
//
//     T=1   gemv      0.652 ms   483 GiB/s     (the plain warp-per-row GEMV)
//     T=3   gemv_tile 1.493 ms   211 GiB/s     (2.29x the T=1 time, identical weight bytes)
//     T=3   reference 5.402 ms    60 GiB/s     (amortises well but 8x slower in absolute terms)
//
// So the tile GEMV does reuse each weight across the tile, but pays a per-token instruction bill
// that eats the entire gain. The tensor core removes that bill: one mma.m16n8k16 covers
// 16 rows x 8 tokens x 16 k in ONE instruction, so every token in the tile rides the same weight
// read at zero extra instruction cost. Weights are decoded in registers: each lane shifts its two
// code pairs out of the staged k-step word and builds the bf16 fragment halves with two byte
// permutes against kTritPairLo/kTritPairHi (the same construction the PTQ1_0 path uses).
//
// LAYOUT NOTES THAT ARE LOAD-BEARING (violate them and T == 1 still looks perfect):
//   * ninfer/ggml keep ne[0] contiguous, so an activation with ne=(k,T) is TOKEN-major: element
//     (column, token) lives at token*k + column. The mma B operand wants n rows of k with k
//     contiguous, so the activation tile is staged verbatim, with no transpose.
//   * the output is token-major as well: (row, token) at token*out_row_stride + row.
//   * PQ2_0 packs FOUR 2-bit codes per byte, and code c denotes weight value (c - 1). The group
//     scale is applied AFTER the mma, per row, because one group spans 8 mma k-steps.
//
// GEOMETRY: one warp owns 16 output rows and walks the whole K; the CTA is 8 warps, so a CTA
// covers 128 rows. Warps never cooperate, so there is no cross-warp reduction and no barrier
// beyond the shared staging handshake. K is consumed in 256-wide chunks (two PQ2_0 groups),
// double buffered with cp.async so the weight stream is not serialised behind the mma.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

inline constexpr int kTernaryMmaRowsPerWarp     = 16;
inline constexpr int kTernaryMmaTokens          = 8;
// Warps (and therefore rows) per CTA. This is the knob that decides how many CTAs a narrow
// projection launches: a PQ2_0 weight is tiled 16 rows per warp, so with 8 warps a N=5120 projection
// gets only 40 CTAs and leaves half of this card's 80 SMs idle, measuring 322 GiB/s instead of the
// 618 GiB/s the same kernel reaches on the 272-CTA MLP shape. Four warps doubles the CTA count
// (N=5120 -> 80 CTAs, one per SM) at the cost of fewer resident warps per SM, which the cp.async
// pipeline is there to absorb.
inline constexpr int kTernaryMmaWarpsPerCta     = 4;
inline constexpr int kTernaryMmaRowsPerCta      = kTernaryMmaRowsPerWarp * kTernaryMmaWarpsPerCta;
inline constexpr int kTernaryMmaGroupK          = PQ2RowSplitStorage::kGroupK;
// K consumed per staged chunk. Doubling this from 256 to 512 halves both the number of
// __syncthreads pairs (two per chunk) and the number of times each CTA re-stages its activation
// tile -- the two overheads that keep this kernel at ~400 GiB/s against the ~540 GiB/s the T=1
// GEMV reaches on the same weights. The cost is more shared memory per CTA, so it is an empirical
// question, not an obvious win.
inline constexpr int kTernaryMmaChunkK          = 512;
inline constexpr int kTernaryMmaGroupsPerChunk  = kTernaryMmaChunkK / kTernaryMmaGroupK;
inline constexpr int kTernaryMmaKStepsPerChunk  = kTernaryMmaChunkK / 16;
inline constexpr int kTernaryMmaCodesPerRow     = kTernaryMmaChunkK / 4; // four codes per byte
inline constexpr int kTernaryMmaRowStride       = 144;                   // 128 used + pad
inline constexpr int kTernaryMmaActStride       = kTernaryMmaChunkK + 8; // 16B-aligned, padded
inline constexpr int kTernaryMmaStages          = 2;
inline constexpr int kTernaryMmaThreads         = kTernaryMmaWarpsPerCta * 32;

static_assert((kTernaryMmaRowStride % 16) == 0, "cp.async needs a 16-byte aligned row start");
static_assert(kTernaryMmaRowStride >= kTernaryMmaCodesPerRow, "the row must hold a whole chunk");
static_assert((kTernaryMmaActStride % 8) == 0, "ldmatrix wants 8-element row starts");
static_assert((kTernaryMmaChunkK % kTernaryMmaGroupK) == 0, "chunks must be whole groups");
static_assert((kTernaryMmaChunkK % 16) == 0, "chunks must be whole mma k-steps");

// ---- PTQ1_0 (type 143) staging: 24 raw base-3 bytes + 2 high bytes per 128 weights -----------
// The tensor core consumes the PQ2 code layout, so a PTQ1_0 chunk is staged RAW and then repacked
// in shared by repack_ptq1() below -- arithmetic identical to the wide-t kernel's (validated
// bit-exact against the real artifact: E:\infer-build\exp\reader-candidates\verify_pq2_repack.py).
// Raw layout inside one weight row of a chunk (2 groups at the PTQ1_0 chunk width):
//   qs  [0, 48)   -- 24 bytes per group, two groups, contiguous
//   qh  [48, 52) --  2 bytes per group, two groups, contiguous
// 48 and 52 are multiples of 4, and so is the row pitch (24 * groups_per_row) and the per-chunk
// qs stride (48), so every copy can be an 8-byte cp.async (qs) or a 4-byte one (qh). A 16-byte qs
// copy would additionally need an even groups_per_row -- true for every width in this model, but
// not a property of the format.
// PTQ1_0 consumes a NARROWER chunk than PQ2_0, and that is the difference that buys the occupancy.
// MEASURED, not assumed (E:\infer-build\exp\reader-candidates\REPORT-ptq1-smallt-occupancy-20260923.md):
// the PTQ1 instance was 47360 B of static shared against PQ2's 38144 B, which pins BOTH at two CTAs
// per SM -- and at two CTAs the 148 registers are 37888 of the 65536 the SM has, i.e. they do not
// clamp anything. Shared is the clamp, so the resident warp count only moves if a chunk gets
// smaller. At 256 wide the PTQ1 instance is 26368 B, which fits THREE CTAs per SM (82176 B of the
// 100 KB sm_89 has) for 12 resident warps instead of 8, with the register file still at 56832.
//
// Nothing about the pipeline changes: the same three rotating weight-row buffers, the same one-chunk
// cp.async prefetch depth in ROWS (every row is still prefetched a whole chunk ahead of the repack
// that consumes it), the same repack, the same mma loop. Only the number of iterations doubles
// (20 chunks of 256 instead of 10 of 512 for K=5120), i.e. two extra barriers per 512 wide.
inline constexpr int kTernaryMmaPtq1ChunkK = 256;
// Weight-row buffers per CTA. PQ2 keeps its two (one being read by the mma, one being cp.async'd).
// PTQ1 needs THREE, and this is the whole design, so it is worth the paragraph:
// its raw bytes and its repacked codes occupy the same rows (a raw chunk needs 52 B, the codes it
// becomes need 64 B, so ONE buffer plays both roles one chunk apart), and the raw bytes of chunk c+1
// must be prefetched while chunk c is still being repacked AND read by the mma. That is three
// simultaneously live roles -- raw(c+1) being filled, raw(c) being read by the repack, and codes(c)
// being read by the mma -- hence three buffers, rotating raw -> codes -> prefetch.
// 3 * 4 * 16 * 80 = 15360 B, which is what keeps the PTQ1 CTA at 26368 B: under the 48 KB static
// cap (no dynamic shared, no cudaFuncSetAttribute) and at THREE CTAs/SM (79104 B of the 100 KB).
inline constexpr int kTernaryMmaPtq1WeightBufs = 3;

// Everything that depends on how wide a chunk is, per instantiation. PQ2_0 keeps 512-wide chunks
// (and therefore byte for byte its old shared layout); PTQ1_0 runs 256 wide so three CTAs fit.
//
// The row stride is not a free parameter: it has to hold a whole chunk (codes for PQ2, raw bytes for
// PTQ1), stay 8-byte aligned so every cp.async destination is legal, and keep the mma's per-k-step
// 4-byte word loads bank-conflict free. A whole k-step is one word and the four lanes of a quad read
// the SAME word, so only rows can conflict: at 144 bytes the word offset is 36*row + kstep and at 80
// it is 20*row + kstep, both of which are distinct across the eight rows one load instruction
// touches (36 == 4 mod 32 and 20 == 20 mod 32 are coprime-ish strides over 0..7 rows), while the
// natural 64 collapses rows onto two banks.
template <bool kPtq1>
struct TernaryMmaGeom {
    static constexpr int kChunkK      = kPtq1 ? kTernaryMmaPtq1ChunkK : kTernaryMmaChunkK;
    static constexpr int kGroups      = kChunkK / kTernaryMmaGroupK;
    static constexpr int kCodesPerRow = kChunkK / 4; // four codes per byte
    static constexpr int kRowStride   = kPtq1 ? 80 : kTernaryMmaRowStride;
    static constexpr int kActStride   = kChunkK + 8; // 16B-aligned, padded
    static constexpr int kWeightBufs  = kPtq1 ? kTernaryMmaPtq1WeightBufs : kTernaryMmaStages;
    // PTQ1_0 raw planes inside a weight row, per chunk.
    static constexpr int kQhOffset = kGroups * PTQ1RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kQhBytes  = kGroups * PTQ1RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kRawBytes = kQhOffset + kQhBytes;
};

using TernaryMmaGeomPq2  = TernaryMmaGeom<false>;
using TernaryMmaGeomPtq1 = TernaryMmaGeom<true>;

static_assert(TernaryMmaGeomPq2::kRowStride == kTernaryMmaRowStride &&
                  TernaryMmaGeomPq2::kCodesPerRow == kTernaryMmaCodesPerRow &&
                  TernaryMmaGeomPq2::kGroups == kTernaryMmaGroupsPerChunk &&
                  TernaryMmaGeomPq2::kActStride == kTernaryMmaActStride,
              "the PQ2_0 instantiation must keep its original geometry byte for byte");
static_assert(TernaryMmaGeomPtq1::kRowStride >= TernaryMmaGeomPtq1::kRawBytes,
              "a weight row must hold a whole raw PTQ1_0 chunk");
static_assert(TernaryMmaGeomPtq1::kRowStride >= TernaryMmaGeomPtq1::kCodesPerRow,
              "a weight row must hold a whole repacked PTQ1_0 chunk");
static_assert((TernaryMmaGeomPtq1::kRowStride % 8) == 0 && (TernaryMmaGeomPtq1::kQhOffset % 4) == 0,
              "every PTQ1_0 staging copy destination must be aligned to its cp.async width");
static_assert(TernaryMmaGeomPtq1::kQhBytes == 4 || TernaryMmaGeomPtq1::kQhBytes == 8,
              "the qh copy is one cp.async of the whole chunk's high bytes");
static_assert((PTQ1RowSplitStorage::kCodeBytesPerGroup % 8) == 0,
              "a PTQ1_0 qs row copy is 8-byte granular, that is what cp.async needs");

// ---- register operand production (both formats): the trit -> bf16 table ------------------------
// A ternary weight is +-1 or 0, so its bf16 encoding is one of THREE constants and the whole table
// fits in two registers (the PQ2_0 path used a 2 KiB shared `uint2 lut[256]` before, see
// TernaryMmaStorage):
//   table bytes [2c, 2c+1] = bf16 of (c - 1)   ->   code 0 = 0xBF80 (-1), 1 = 0x0000, 2 = 0x3F80 (+1)
// __byte_perm's selector is a REGISTER, so once the selector is built, "two trit codes -> one bf16
// pair in A-fragment order" is exactly one PRMT:
//   t = c0 | (c1 << 8)  ->  sel = 34 * t + 0x1010  ->  __byte_perm(kTritPairLo, kTritPairHi, sel)
// 34 = 0x22 places table byte 2c at result byte 0 and 2c+1 at byte 1 for c0 (and the same for c1 at
// result bytes 2,3); every selector nibble stays <= 7, so two registers hold the whole table.
inline constexpr unsigned kTritPairLo = 0x0000BF80u; // table bytes: 80 BF 00 00
inline constexpr unsigned kTritPairHi = 0x00003F80u; // table bytes: 80 3F 00 00

// One PRMT builds the pair (low half = c0's weight, high half = c1's weight).
__device__ __forceinline__ unsigned ternary_trit_pair(unsigned codes) {
    return __byte_perm(kTritPairLo, kTritPairHi, 34u * codes + 0x1010u);
}

// ---- SWEEP VARIANT B (2026-09-23): TWO pair selectors out of ONE IMAD ---------------------------
// Four code bytes are packed in one register (the k pair in bytes 0,1 and the k+8 pair in bytes
// 2,3), and 34 * codes4 + 0x10101010 is byte-wise clean: 34 * 2 + 0x10 = 0x54 < 0x100, so no byte
// ever carries into its neighbour and byte k of the result is exactly the selector for the pair
// whose two codes are bytes k and k+1. __byte_perm reads only the low 16 bits of its selector, so
// the low half serves the k pair and `sel >> 16` the k+8 pair: one gather PRMT + one IMAD + one SHF
// replace two gathers + two IMADs. The pair selectors themselves are unchanged.
__device__ __forceinline__ unsigned ternary_pair_selectors(unsigned codes4) {
    return 34u * codes4 + 0x10101010u;
}

// One base-3 chain step applied to TWO raw bytes at once, laid out as v = b0 | (b1 << 16).
// w = v*3 then holds 3*b0 in [15:0] and 3*b1 in [31:16] with no cross-talk (each starts below 256
// and peaks at 765), and the HIGH byte of each product IS the code the format wants, ((3*b)>>8):
//   codes = __byte_perm(w, 0, 0x4431)  ->  byte0 = code(b0), byte1 = code(b1)
//   next  = w & 0x00FF00FF             ->  the same two values mod 256, ready for the next stage
// The caller keeps `next` in a register, so a whole group costs ONE advance per k-step instead of
// restarting the chain from raw * 3^stage every step.
__device__ __forceinline__ unsigned ternary_chain_step(unsigned v, unsigned& next) {
    const unsigned w = v * 3u;
    next             = w & 0x00FF00FFu;
    return __byte_perm(w, 0u, 0x4431u);
}

// The kPtq1 instantiation's raw -> PQ2_0 code repack. Native operand production does not need it:
// the five code bytes it built were only ever consumed by the 256-entry LUT, which this path
// replaces with ternary_chain_step + ternary_trit_pair. Kept as a named switch (not deleted) so the
// old path stays one edit away and the arithmetic above stays reviewable.
inline constexpr bool kTernaryMmaPtq1Repack = false;

// NINFER_TERNARY_MMA=0 forces the SIMT paths, which is the A/B switch for validating this kernel
// end to end on the real artifact.
//
// It is needed because the perplexity harness runs with --context 512, i.e. T = 512, which is far
// outside this kernel's 2..8 token window and therefore falls through to the reference kernel. A
// perplexity run with a window inside 2..8 IS covered here, so comparing the two values with the
// switch on and off is a direct numeric equivalence test of the tensor-core path on real weights
// rather than on a synthetic payload.
//
// Read once: the choice decides whether a kernel appears in the captured CUDA graph at all, so it
// must not change between graph construction and replay (same rule as the rotation switch).
[[nodiscard]] inline bool ternary_mma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_MMA");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// NINFER_TERNARY_TOKEN_GRID=0 restores the serial token-tile walk inside each CTA of the small-T,
// wide-T and int8 rungs (SCHED3 "A3", author's 2026-09-27 sm_86 upgrade package). Default on: the
// token tiles become grid.y, which fills the machine when N / rows-per-CTA alone does not (a
// 1024-row k/v projection is 16 CTAs of the int8 rung on a 28-SM part). Pure re-partitioning --
// bit-identical outputs. Read once (graph capture), like the switches above.
[[nodiscard]] inline bool ternary_token_grid_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_TOKEN_GRID");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// NINFER_TERNARY_T1_MMA=0 sends PQ2_0 decode (T == 1) back to the warp-per-row SIMT GEMV instead of
// this kernel at kActRows == 1. The end-to-end A/B switch for that routing decision (see
// launch_ternary_gemm_t1); read once for the same graph-capture reason as above.
[[nodiscard]] inline bool ternary_t1_mma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_T1_MMA");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// Shared staging. The row strides are chosen so that the shared accesses in the hot loop are
// bank-conflict free, which is why they are not just the natural 64 / 256:
//   * codes: a whole k-step is one 4-byte word, and the four lanes of a quad read the SAME word,
//     so the only conflicts are between rows. With a 64-byte row stride the word offset is
//     16*row + kstep and rows collapse onto two banks; 80 bytes gives 20*row + kstep, whose
//     low five bits are distinct across the eight rows a single load touches.
//   * activations: ldmatrix pulls 16 bytes per row, so rows must start on distinct banks.
//     528 bytes = 132 words => 4 banks apart, which lands rows 0..7 on banks 0,4,...,28 exactly.
//
// kPtq1 makes the weight rows play two roles: a PTQ1_0 chunk is staged there RAW (qs at [0,48),
// qh at [48,52)) and repacked in place into the PQ2 code layout the mma reads. See
// kTernaryMmaPtq1WeightBufs for why that needs three of them instead of two, and TernaryMmaGeom for
// why the PTQ1_0 rows are 80 bytes wide instead of 144 (a 256-wide chunk, so that three CTAs fit).
// PQ2_0 decode happens in registers (see the kernel), so there is no decode table in shared
// memory. The former 256-entry `uint2 lut[256]` was measured on sm_86 (RTX 3060, 28 SMs) as the
// binding cost of this kernel: every LUT variant of the staging geometry (chunk 128/256/512,
// 2..6 stages, 2..8 warps) sat at 132-146 GiB/s against a 296 GiB/s streaming-read roof, while
// the register decode reached 179-217 GiB/s at T=5 on the same shapes, bit-identical outputs.
// The lookups are data-dependent addresses from eight rows per load, so they serialise on bank
// conflicts that the quad broadcast does not remove.
//
// kActRows is the number of token rows actually staged: min(tokens, kTernaryMmaTokens). The mma
// still computes eight token columns, but columns past the live tokens are never stored, so their
// B rows only have to be addressable, not zero -- ldmatrix clamps those lanes onto the last live
// row. At T=5 (the MTP d4 verify) that trims 6.1 KiB of activation staging, at T=1 14.2 KiB,
// which is what lets three (T=5) or four (T=1) CTAs share an sm_86 SM instead of two; measured
// +3..14% on top of the register decode.
template <bool kPtq1, int kActRows>
struct TernaryMmaStorage {
    using Geom = TernaryMmaGeom<kPtq1>;
    alignas(16) std::uint8_t codes[Geom::kWeightBufs][kTernaryMmaWarpsPerCta]
                                  [kTernaryMmaRowsPerWarp][Geom::kRowStride];
    alignas(16) __nv_bfloat16 act[kTernaryMmaStages][kActRows][Geom::kActStride];
    std::uint16_t scales[kTernaryMmaStages][kTernaryMmaWarpsPerCta][kTernaryMmaRowsPerWarp]
                        [Geom::kGroups];
};

// kPtq1 == false: `codes` is the PQ2_0 32-code-byte-per-group base plane and `high` is unused
//                 (the PQ2_0 layout has no high plane, so callers pass nullptr).
// kPtq1 == true : `codes` is the PTQ1_0 24-byte base plane and `high` its 2-byte high plane. The
//                 staging cp.async's both raw planes into a weight row, the repack turns the
//                 24+2 bytes into the same 32-byte PQ2 code bytes the PQ2 path consumes, and every
//                 line below the repack is byte for byte the PQ2 path (mma, scale, stores).
// kActRows      : staged token rows, min(tokens, kTernaryMmaTokens) -- see TernaryMmaStorage.
template <bool kPtq1, int kActRows>
__global__ __launch_bounds__(kTernaryMmaThreads)
void ternary_mma_small_t_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::uint8_t* __restrict__ codes,
                                const std::uint8_t* __restrict__ high,
                                const std::uint8_t* __restrict__ scales,
                                __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                std::int32_t k, std::int32_t tokens,
                                std::int32_t out_row_stride, bool token_grid = false) {
    // STATIC shared memory, deliberately not `extern __shared__`: the footprint is a compile-time
    // constant (36,096 B at kPtq1 == false and kActRows == 8, 29,856 B at 5, 21,536 B at 1; the
    // kPtq1 == true instantiation is smaller still -- all under the 48 KB per-block
    // static cap), and declaring an extern shared VARIABLE OF A NAMED AGGREGATE type emits a device
    // symbol that nvlink then cannot resolve
    // ("nvlink error : Undefined reference to 'ninfer::ops::detail::shared'"). Static also removes
    // the need for a cudaFuncSetAttribute opt-in and for a size at every launch site.
    //
    // The footprint is what decides occupancy on this kernel: on sm_86 (100 KB per SM) the PQ2_0
    // instantiation fits two CTAs at kActRows == 8, three at 5 and four at 1 (see
    // TernaryMmaStorage); the PTQ1_0 one is three at 8 (see kTernaryMmaPtq1ChunkK).
    using Geom = TernaryMmaGeom<kPtq1>;
    static_assert(kActRows >= 1 && kActRows <= kTernaryMmaTokens, "1..8 staged token rows");
    __shared__ TernaryMmaStorage<kPtq1, kActRows> staging;
    auto& codes_sh = staging.codes;
    auto& act_sh   = staging.act;
    auto& scale_sh = staging.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2; // 0..7  -- mma row within the fragment
    const int lid  = lane & 3;  // 0..3  -- mma k pair / column pair index

    const int row0 = static_cast<int>(blockIdx.x) * kTernaryMmaRowsPerCta;
    if (row0 >= rows) { return; }

    constexpr int kCodeBytes =
        kPtq1 ? PTQ1RowSplitStorage::kCodeBytesPerGroup : PQ2RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kHighBytes = kPtq1 ? PTQ1RowSplitStorage::kHighBytesPerGroup : 0;
    const std::int32_t groups_per_row = k / kTernaryMmaGroupK;
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kCodeBytes;
    [[maybe_unused]] const std::int64_t high_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kHighBytes;
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * PQ2RowSplitStorage::kScaleBytesPerGroup;
    const std::int32_t chunks = k / Geom::kChunkK;

    const std::int64_t warp_row_base =
        static_cast<std::int64_t>(row0) + static_cast<std::int64_t>(warp) * kTernaryMmaRowsPerWarp;

    // Stage one chunk: this warp's 16x64 code window, its scales, and (cooperatively) the
    // CTA-wide activation tile. Out-of-range rows are zero-filled rather than read, so a partial
    // CTA never touches global memory past the end of the weight.
    //
    // `tok_base` selects which tile of kTernaryMmaTokens tokens this staging is for. A CTA walks
    // the whole sequence in such tiles, which is what lets one kernel serve both the 2..8 token
    // speculative verify pass and a full prefill, instead of falling back to the reference kernel
    // above 8 tokens -- the fallback that made prefill measure 55 tok/s against 412-437 for the
    // non-ternary engine.
    const auto stage = [&](int buf, int chunk, int tok_base) {
        const std::int64_t chunk_byte =
            static_cast<std::int64_t>(chunk) * Geom::kCodesPerRow;
        const std::int64_t chunk_k = static_cast<std::int64_t>(chunk) * Geom::kChunkK;

        // Which weight-row buffer this staging writes. PQ2_0 writes the codes of `chunk` into the
        // ping-pong slot the next iteration's mma will read (`buf`, passed in by the caller). PTQ1_0
        // writes the RAW bytes of `chunk` into the third buffer of the rotating set -- see
        // kTernaryMmaPtq1WeightBufs: it cannot use `buf`, because the codes of the PREVIOUS chunk
        // (repacked there by the last iteration's repack) are still being read by the mma.
        const int wbuf = kPtq1 ? (chunk % Geom::kWeightBufs) : buf;

        if constexpr (kPtq1) {
            // PREFETCH ONLY: the raw 24+2-byte row is cp.async'd here and repacked by repack_ptq1()
            // after cp_wait, so the global latency is absorbed by the previous chunk's mma instead
            // of sitting on top of the dependent repack arithmetic (the failure mode the wide-t
            // kernel measured and fixed first).
            //
            // qs: 6 x 8-byte copies per row (24 bytes per group x 2 groups = 48 contiguous bytes).
            // The row-major lane mapping (three consecutive lanes cover one group's 24 bytes) keeps
            // consecutive lanes on consecutive addresses; 48 is a multiple of 16, so every 32-byte
            // sector a warp touches is covered by whole copies even at 8-byte granularity.
            constexpr int kQsCopies     = Geom::kGroups *
                                          (PTQ1RowSplitStorage::kCodeBytesPerGroup / 8);
            constexpr int kQsCopiesWarp = kTernaryMmaRowsPerWarp * kQsCopies;
            const std::int64_t qs_chunk_byte =
                static_cast<std::int64_t>(chunk) * Geom::kGroups *
                PTQ1RowSplitStorage::kCodeBytesPerGroup;
#pragma unroll
            for (int copy = lane; copy < kQsCopiesWarp; copy += 32) {
                const int local_row           = copy / kQsCopies;
                const int byte_off            = (copy % kQsCopies) * 8;
                const std::int64_t global_row = warp_row_base + local_row;
                const bool valid              = global_row < rows;
                // An invalid row zero-fills (src_bytes == 0) instead of branching to a plain shared
                // store, so the repack produces zeros by construction and needs no validity test.
                // The fallback source is the plane base, which is always mapped.
                cp_async_zfill<8, Cache::ca>(
                    &codes_sh[wbuf][warp][local_row][byte_off],
                    valid ? codes + global_row * code_row_bytes + qs_chunk_byte + byte_off : codes,
                    valid ? 8 : 0);
            }
            // qh: the whole chunk's high bytes are 2 groups x 2 bytes = 4 CONTIGUOUS bytes per row
            // (the high plane is per-row, group-major, exactly like the base plane), so one cp.async
            // of that width covers it. 4-byte aligned because groups_per_row is a multiple of 4 on
            // this path (K is a whole number of 512-wide chunks, and a 256-wide chunk is two 128-wide
            // groups) so the high-plane row pitch 2*groups_per_row is a multiple of 4, and the
            // per-chunk offset is a multiple of 4 as well. No overrun: the copy ends exactly at the
            // next row's start.
            static_assert(Geom::kQhBytes == 4 || Geom::kQhBytes == 8,
                          "the qh copy below is one cp.async of the chunk's high bytes");
            if (lane < kTernaryMmaRowsPerWarp) {
                const std::int64_t global_row = warp_row_base + lane;
                const bool valid              = global_row < rows;
                cp_async_zfill<Geom::kQhBytes, Cache::ca>(
                    &codes_sh[wbuf][warp][lane][Geom::kQhOffset],
                    valid ? high + global_row * high_row_bytes +
                                static_cast<std::int64_t>(chunk) * Geom::kQhBytes
                          : high,
                    valid ? Geom::kQhBytes : 0);
            }
        } else {
        // Codes: kTernaryMmaCodesPerRow bytes per row, moved as 16-byte copies filled round-robin
        // across the lanes. Written generically on purpose -- sizing this loop by hand for one
        // chunk width silently stages only PART of each row at another width, which presents as a
        // suspiciously fast kernel that produces garbage (all four correctness checks went to NaN).
        constexpr int kCopiesPerRow  = Geom::kCodesPerRow / 16;
        constexpr int kCopiesPerWarp = kTernaryMmaRowsPerWarp * kCopiesPerRow;
#pragma unroll
        for (int copy = lane; copy < kCopiesPerWarp; copy += 32) {
            const int local_row           = copy / kCopiesPerRow;
            const int byte_off            = (copy % kCopiesPerRow) * 16;
            const std::int64_t global_row = warp_row_base + local_row;
            std::uint8_t* dst             = &codes_sh[wbuf][warp][local_row][byte_off];
            if (global_row < rows) {
                cp_async<16, Cache::cg>(
                    dst, codes + global_row * code_row_bytes + chunk_byte + byte_off);
            } else {
                store_vec<std::uint8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }
        }

        { // scales: one entry per (row, group-in-chunk) -- 16 x 4 at a 512-wide chunk
            constexpr int kScaleEntries = kTernaryMmaRowsPerWarp * Geom::kGroups;
#pragma unroll
            for (int entry = lane; entry < kScaleEntries; entry += 32) {
                const int local_row           = entry / Geom::kGroups;
                const int group_in_chunk      = entry % Geom::kGroups;
                const std::int64_t global_row = warp_row_base + local_row;
                std::uint16_t value           = 0u;
                if (global_row < rows) {
                    value = load_vec<std::uint16_t>(
                        scales + global_row * scale_row_bytes +
                        (static_cast<std::int64_t>(chunk) * Geom::kGroups +
                         group_in_chunk) *
                            PQ2RowSplitStorage::kScaleBytesPerGroup);
                }
                scale_sh[buf][warp][local_row][group_in_chunk] = value;
            }
        }

        { // activation tile: 8 tokens x 256 k, staged cooperatively by the whole CTA.
            // Written as a stride loop rather than `tok = tid >> 5` so it stays correct for any
            // warp count: with 4 warps that expression would only ever cover tokens 0..3.
            // These offsets are in bf16 ELEMENTS, not bytes: x is a __nv_bfloat16* while the code
            // staging above is byte-based because `codes` is a std::uint8_t*. A byte offset here
            // over-advances the source window by 2x and walks past the end of each token's row,
            // which presents as "values plausible but wrong, NaN at the tail" rather than as a
            // clean bounds error.
            // 16 bytes (8 bf16) per cp.async, and a chunk is Geom::kChunkK elements per token, so
            // the number of copies a lane issues scales with the chunk width -- at 512 each lane
            // covers an element offset and that offset + 256, at 256 (the PTQ1_0 width) one copy
            // per lane already covers the whole chunk row.
            for (int tok = warp; tok < kActRows; tok += kTernaryMmaWarpsPerCta) {
                const int global_tok = tok_base + tok;
                // Tokens past the end of the sequence are zero-filled (src_bytes 0 reads nothing),
                // so the mma sees a defined activation and the guarded store drops the result.
                const int safe_tok = (global_tok < tokens) ? global_tok : tok_base;
                for (int elem = lane * 8; elem < Geom::kChunkK; elem += 32 * 8) {
                    cp_async_zfill<16>(
                        &act_sh[buf][tok][elem],
                        x + static_cast<std::int64_t>(safe_tok) * k + chunk_k + elem,
                        (global_tok < tokens) ? 16 : 0);
                }
            }
        }
    };

    // REPACK: the staged raw row (qs at [0,96), qh at [96,104)) -> the PQ2 code layout the mma
    // consumes (32 bytes per group, code byte b holding weights 4b..4b+3 as two-bit codes), so every
    // line downstream of this step is byte for byte the PQ2 path. THE ARITHMETIC IS UNCHANGED from
    // the wide-t kernel: it is bit-exact against the verified decoders
    // (E:\infer-build\exp\reader-candidates\verify_pq2_repack.py -> 0 / 122,880 mismatches), and the
    // only edits here are the per-group base offsets (the wide-t kernel stages ONE group per chunk,
    // this one stages two) and the fact that the source is the third rotating buffer rather than a
    // dedicated raw one.
    //
    // Called one chunk after the cp.async that fills `src_buf`, i.e. after cp_wait, so the global
    // latency has already been absorbed. Writes the codes into `dst_buf`, which the caller reads
    // with the mma in the same iteration -- hence the __syncwarp()/__syncthreads() pair around it.
    //
    // Seven jobs per (row, group), flattened over the warp's 32 lanes in THREE divergence-free
    // loops rather than one branchy flattened loop: with 32 lanes and 7 slots per row a single flat
    // loop puts lanes with different `slot` values in the same warp instruction, so the three
    // branches (`slot < 4` / `slot < 6` / qh) would all be executed masked for most instructions.
    // Splitting them costs one extra warp-iteration and removes that 3-way serialisation.
    //   slots 0..3: qs[4g..4g+4)     at stages 0..4 -> code bytes 4t+g    (weights 0..79)
    //   slots 4..5: qs[16+4p..20+4p) at stages 0..4 -> code bytes 20+2t+p (weights 80..119)
    //   qh job    : qh[0..2) -> byte 30 (trits 0,0,1,1) and byte 31 (trits 2,2,3,3)
    // code_of() is PTQ1SimtDecodeAtom's xi = ((uint8)(raw * 3^t) * 3) >> 8, which IS the PQ2 code
    // (PQ2 decodes as (code - 1) * d), so no sign juggling is needed anywhere.
    const auto repack_ptq1 = [&](int src_buf, int dst_buf) {
        if constexpr (kPtq1) {
            const auto code_of = [](std::uint8_t raw, int pow3) {
                const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3);
                return (static_cast<unsigned>(q) * 3u) >> 8;
            };
            // One 4-byte source group -> FIVE code bytes (all five stages at once). Two changes
            // against the wide-t kernel's stage_codes(), both verified bit-exact against the direct
            // formula code = ((uint8)(raw * 3^t) * 3) >> 8 for all 256 byte values x 5 stages by
            // E:\infer-build\exp\ptq1-mmasmallt\verify_smallt_repack.py:
            //   * the CHAIN is advanced once instead of re-run per stage: stage_codes(g4, t)
            //     restarts from stage 0 every call, so the wide-t kernel's five calls execute
            //     1+2+3+4+5 = 15 chain steps, while iterating once and emitting a byte per step is
            //     5 steps for the same five bytes.
            //   * the PACK folds four 2-bit codes with one multiply: the codes sit in bytes 0..3 of
            //     `word` with only their low two bits meaningful, and multiplying by 0x01041040
            //     lands b0 + 4*b1 + 16*b2 + 64*b3 in the top byte (each field <= 3, so byte 2 peaks
            //     at 252 and cannot carry into it). That replaces ~11 shift/and/or instructions.
            // The vectorised byte_perm extraction and the base-3 multiply chain themselves are
            // PrismML's (ggml/src/ggml-cuda/mmq-load-tiles.cuh:261, MIT) and are unchanged.
            // `dst_step` is the distance between consecutive stages' destination bytes (4 for the
            // qs[0..16) words, 2 for qs[16..24)) and is a literal at every call site, so it folds.
            const auto emit_stages = [](std::uint8_t* dst, int dst_step, unsigned group4) {
                unsigned v_lo = __byte_perm(group4, 0u, 0x4140);
                unsigned v_hi = __byte_perm(group4, 0u, 0x4342);
#pragma unroll
                for (int t = 0; t < 5; ++t) {
                    const unsigned w_lo = v_lo * 3u;
                    const unsigned w_hi = v_hi * 3u;
                    v_lo                = w_lo & 0x00FF00FFu;
                    v_hi                = w_hi & 0x00FF00FFu;
                    const unsigned word = __byte_perm(w_lo, w_hi, 0x7531);
                    dst[t * dst_step]   = static_cast<std::uint8_t>(
                        ((word & 0x03030303u) * 0x01041040u) >> 24);
                }
            };
#pragma unroll
            for (int group = 0; group < Geom::kGroups; ++group) {
                const int qs_base = group * PTQ1RowSplitStorage::kCodeBytesPerGroup;
                const int cd_base = group * PQ2RowSplitStorage::kCodeBytesPerGroup;

                // qs[0..16): four 4-byte source words per row -> 64 jobs, two per lane, slot == lane&3
#pragma unroll
                for (int job = lane; job < kTernaryMmaRowsPerWarp * 4; job += 32) {
                    const int local_row = job >> 2;
                    const int slot      = job & 3;
                    const unsigned g4 =
                        load_vec<unsigned>(&codes_sh[src_buf][warp][local_row][qs_base + 4 * slot]);
                    emit_stages(&codes_sh[dst_buf][warp][local_row][cd_base + slot], 4, g4);
                }

                // qs[16..24): two 4-byte source words per row -> 32 jobs, one per lane
                {
                    const int local_row = lane >> 1;
                    const int par       = lane & 1;
                    const unsigned g4   = load_vec<unsigned>(
                        &codes_sh[src_buf][warp][local_row][qs_base + 16 + 4 * par]);
                    emit_stages(&codes_sh[dst_buf][warp][local_row][cd_base + 20 + par], 2, g4);
                }

                // qh: 16 rows, one lane per row
                if (lane < kTernaryMmaRowsPerWarp) {
                    const std::uint8_t* src_row = &codes_sh[src_buf][warp][lane][0];
                    const std::uint8_t h0       = src_row[Geom::kQhOffset + 2 * group];
                    const std::uint8_t h1       = src_row[Geom::kQhOffset + 2 * group + 1];
                    std::uint8_t* dst           = &codes_sh[dst_buf][warp][lane][cd_base];
                    dst[30] = static_cast<std::uint8_t>(code_of(h0, 1) | (code_of(h1, 1) << 2) |
                                                        (code_of(h0, 3) << 4) | (code_of(h1, 3) << 6));
                    dst[31] = static_cast<std::uint8_t>(code_of(h0, 9) | (code_of(h1, 9) << 2) |
                                                        (code_of(h0, 27) << 4) | (code_of(h1, 27) << 6));
                }
            }
        }
    };

    // Walk the sequence in tiles of kTernaryMmaTokens tokens.
    //
    // The weight window is re-staged for every tile, because accumulators for the whole sequence
    // would not fit in registers. That amortises each weight read across kTernaryMmaTokens tokens
    // -- the same factor the reference kernel gets from its 8-token tile, but through a kernel that
    // sustains ~400 GiB/s instead of ~60 GiB/s. Above 8 tokens this replaces a fallback that cost
    // prefill a factor of ~8 against the non-ternary engine.
    const int row_lo = row0 + warp * kTernaryMmaRowsPerWarp + gid;
    const int row_hi = row_lo + 8;

    // ---- SWEEP VARIANT A (2026-09-23): hoisted base-3 stage multiplier for the qh bytes --------
    // Step 7 of every group decodes the qh pair at stage `lid`, and `lid = lane & 3` is 0..3, so the
    // chain that used to walk it one step at a time in a runtime loop is a multiplication by
    // 3^lid mod 256 -- a per-LANE constant, not a per-data one. Computing it once here (3 iterations
    // at most, once per kernel, outside the chunk loop) removes a divergent loop from the innermost
    // block: the old `for (i < lid) q = (q * 3) & 0x00FF00FF` was ~48 SASS instructions per row group
    // in a region where every other k-step costs ~25, and it carried BSSY/BSYNC/BRA divergence
    // management with it. Arithmetic is unchanged: (v * 3^lid) mod 256 == the loop's result, and
    // v <= 242, 3^lid <= 27, so the 16-bit lanes of a packed pair cannot carry into each other.
    [[maybe_unused]] unsigned pow3_lid = 1u;
    if constexpr (kPtq1) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i < lid) { pow3_lid *= 3u; }
        }
    }

    // SCHED3 CHANGE 3 (token tile on grid.y, NINFER_TERNARY_TOKEN_GRID): with the switch on each
    // CTA owns exactly one 8-token tile (blockIdx.y); off, the tiles are walked serially here as
    // before. Accumulators are zeroed and the weight window re-staged from chunk 0 inside the tile
    // body, so tiles share no state and re-partitioning them cannot reorder a single addition.
    const int tok_tiles = token_grid ? 1 : ((tokens + kTernaryMmaTokens - 1) / kTernaryMmaTokens);
    const int tok_base0 = token_grid ? static_cast<int>(blockIdx.y) * kTernaryMmaTokens : 0;
    if (tok_base0 >= tokens) { return; }

    for (int tok_tile = 0; tok_tile < tok_tiles; ++tok_tile) {
        const int tok_base = tok_base0 + tok_tile * kTernaryMmaTokens;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        stage(0, 0, tok_base);
        cp_commit();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int buf = chunk & 1;
            // The weight-row buffer the MMA reads this iteration. PQ2_0: the ping-pong slot staged
            // one iteration ago. PTQ1_0: the buffer the repack below is about to write, i.e. the
            // third one -- see kTernaryMmaPtq1WeightBufs. (Both fold to `chunk & 1` for PQ2_0, so
            // the PQ2_0 instantiation compiles to exactly the code it did before this change.)
            // Unused (but harmless) on the native path: nothing repacks, so the raw buffer of this
            // chunk is what the mma reads and `wbuf` has no consumer there.
            [[maybe_unused]] const int wbuf = kPtq1 ? ((chunk + 2) % Geom::kWeightBufs) : buf;

            if (chunk + 1 < chunks) {
                stage(buf ^ 1, chunk + 1, tok_base);
                cp_commit();
                cp_wait<1>(); // chunk `buf` has landed; chunk+1 may still be in flight
            } else {
                cp_wait<0>();
            }
            if constexpr (kPtq1) {
                // cp.async completion is only ordered for the thread that ISSUED the copy, while
                // everything that reads the raw bytes copied one of the other lanes of this warp
                // -- so one warp fence first, on BOTH operand-production paths. The
                // __syncthreads() right after publishes them (and the activation tile, which the
                // whole CTA stages cooperatively) for every lane.
                __syncwarp();
                if constexpr (kTernaryMmaPtq1Repack) {
                    repack_ptq1(chunk % Geom::kWeightBufs, wbuf);
                }
            }
            __syncthreads();

#pragma unroll 1
        for (int group_in_chunk = 0; group_in_chunk < Geom::kGroups;
             ++group_in_chunk) {
            float g0 = 0.0f;
            float g1 = 0.0f;
            float g2 = 0.0f;
            float g3 = 0.0f;

            if constexpr (kPtq1) {
                // ---- NATIVE PTQ1_0 OPERAND PRODUCTION ------------------------------------------
                // The raw 24+2-byte row is decoded STRAIGHT into the A fragment: no repack into
                // PQ2_0 code bytes, no 256-entry LUT, no k-step expansion plane. Each lane owns the
                // few raw bytes its own fragment quarter needs and carries the base-3 chain across
                // the eight k-steps, so a stage costs ONE advance per (row, word) instead of a
                // chain restart -- that carrying is the reason this is competitive at all.
                //
                // WHERE THE LANE'S WEIGHTS LIVE (weights 0..79 / 80..119 / 120..127 of a 128-wide
                // group; e is the weight index inside the group, and every line below is the
                // verified map of verify_native_ptq1_map.py, not a re-derivation):
                //   e < 80   : byte qs[e & 15],        stage e >> 4      (5 trits per byte, byte steps 1)
                //   e < 120  : byte qs[16+((e-80)&7)], stage (e-80) >> 3
                //   else     : byte qh[(e-120)&1],     stage (e-120) >> 1
                // A fragment slot k of a k-step is weight 16*step + 2*lid (lane lid, quad gid), and
                // slot k+8 is weight 16*step + 2*lid + 8 (see the a0..a3 block of the PQ2_0 path):
                //   step 0..4 : pairs (2*lid, 2*lid+1) and (2*lid+8, 2*lid+9), all stage `step`
                //               -- two BYTES OF ONE qs WORD, so one 4-byte LDS feeds a whole pair
                //   step 5,6  : pairs from qs[16+2*lid ..+2) at stages (0,1) then (2,3)
                //   step 7    : k from that same word at stage 4, k+8 from qh at stage `lid`
                // Every pair is therefore either "two bytes, same stage" (one word, one stage) or
                // "one byte, two consecutive stages" -- never a straddling pair, which is exactly
                // what a per-lane register decoder needs.
                //
                // r = 0 is row gid, r = 1 is row gid+8: the mma's two row groups.
                const int rbuf    = chunk % Geom::kWeightBufs; // raw bytes of THIS chunk live here
                const int qs_base = group_in_chunk * PTQ1RowSplitStorage::kCodeBytesPerGroup;
                const int half    = lid >> 1; // which 4-byte word inside a 16-byte qs half
                // Bytes b and b+1 of one qs word, gathered as b | (b << 16) so the chain can run on
                // both at once. Even lids keep the low byte pair, odd lids the high one.
                const unsigned pair_sel = (lid & 1) ? 0x4342u : 0x4140u;
                // vA: qs pair (2*lid, 2*lid+1)      -- weights 16*step + 2*lid        (k,   region 0-79)
                // vB: qs pair (2*lid+8, 2*lid+9)    -- weights 16*step + 2*lid + 8    (k+8, region 0-79)
                // vC: qs pair (16+2*lid, 16+2*lid+1)-- weights 80+2*lid .. (regions used by steps 5-7)
                // vQ: qh[0], qh[1]                  -- weights 120+2*lid (k+8 of step 7)
                unsigned vA[2];
                unsigned vB[2];
                unsigned vC[2];
                unsigned vQ[2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const std::uint8_t* row = &codes_sh[rbuf][warp][gid + 8 * r][0];
                    vA[r] = __byte_perm(load_vec<unsigned>(row + qs_base + 4 * half), 0u, pair_sel);
                    vB[r] =
                        __byte_perm(load_vec<unsigned>(row + qs_base + 8 + 4 * half), 0u, pair_sel);
                    vC[r] =
                        __byte_perm(load_vec<unsigned>(row + qs_base + 16 + 4 * half), 0u, pair_sel);
                    // The chunk's whole high plane is 4 CONTIGUOUS bytes (2 per group, group-major),
                    // so one word at Geom::kQhOffset serves both groups: bytes 0,1 are group 0's
                    // qh pair and bytes 2,3 group 1's.
                    vQ[r] = __byte_perm(load_vec<unsigned>(row + Geom::kQhOffset), 0u,
                                        (group_in_chunk & 1) ? 0x4342u : 0x4140u);
                }

#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    const int kstep = group_in_chunk * 8 + step;
                    unsigned a0;
                    unsigned a1;
                    unsigned a2;
                    unsigned a3;

                    if (step < 5) {
                        // Weights 16*step + 2*lid + {0, 1, 8, 9} -- all below 80, so both bytes of
                        // each pair sit in ONE qs word at the SAME stage: one advance per word.
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            // VARIANT B: both pairs of this k-step share one code gather and one
                            // selector IMAD (low half -> k pair, high half -> k+8 pair).
                            const unsigned wa = vA[r] * 3u;
                            const unsigned wb = vB[r] * 3u;
                            vA[r]             = wa & 0x00FF00FFu;
                            vB[r]             = wb & 0x00FF00FFu;
                            const unsigned sel = ternary_pair_selectors(__byte_perm(wa, wb, 0x7531u));
                            const unsigned lo  = __byte_perm(kTritPairLo, kTritPairHi, sel);
                            const unsigned hi  = __byte_perm(kTritPairLo, kTritPairHi, sel >> 16);
                            if (r == 0) {
                                a0 = lo;
                                a2 = hi;
                            } else {
                                a1 = lo;
                                a3 = hi;
                            }
                        }
                    } else if (step < 7) {
                        // Weights 80..119: (k, k+1) is the same two bytes at stage s and (k+8, k+9)
                        // the same two bytes at stage s+1 -- steps 5 and 6 consume stages (0,1) and
                        // (2,3), so two advances produce both pairs.
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            // VARIANT B, same sharing as above: the two pairs are stages s and s+1
                            // of the SAME word, so they merge even more naturally.
                            const unsigned wa  = vC[r] * 3u;
                            const unsigned mid = wa & 0x00FF00FFu;
                            const unsigned wb  = mid * 3u;
                            vC[r]              = wb & 0x00FF00FFu;
                            const unsigned sel = ternary_pair_selectors(__byte_perm(wa, wb, 0x7531u));
                            const unsigned lo  = __byte_perm(kTritPairLo, kTritPairHi, sel);
                            const unsigned hi  = __byte_perm(kTritPairLo, kTritPairHi, sel >> 16);
                            if (r == 0) {
                                a0 = lo;
                                a2 = hi;
                            } else {
                                a1 = lo;
                                a3 = hi;
                            }
                        }
                    } else {
                        // Step 7: k is weights 112+2*lid (stage 4 of the same qs word, one more
                        // advance) and k+8 is weights 120+2*lid, i.e. qh[0], qh[1] at stage `lid`
                        // -- a per-LANE stage, hence the hoisted pow3_lid multiplier instead of a
                        // runtime chain walk (SWEEP VARIANT A).
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            // VARIANT B: k's product and qh's product merge into one gather.
                            const unsigned wa = vC[r] * 3u;
                            vC[r]             = wa & 0x00FF00FFu;
                            const unsigned qn = (vQ[r] * pow3_lid) & 0x00FF00FFu;
                            const unsigned wq = qn * 3u;
                            const unsigned sel = ternary_pair_selectors(__byte_perm(wa, wq, 0x7531u));
                            const unsigned lo  = __byte_perm(kTritPairLo, kTritPairHi, sel);
                            const unsigned hi  = __byte_perm(kTritPairLo, kTritPairHi, sel >> 16);
                            if (r == 0) {
                                a0 = lo;
                                a2 = hi;
                            } else {
                                a1 = lo;
                                a3 = hi;
                            }
                        }
                    }

                    // B operand: the activation tile is already n-major (token rows, k contiguous),
                    // which is exactly the .col layout the mma wants -- no transpose needed.
                    unsigned bf0 = 0u;
                    unsigned bf1 = 0u;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&act_sh[buf][min(lane & 7, kActRows - 1)]
                                                [kstep * 16 + ((lane >> 3) & 1) * 8]));

                    mma_bf16(g0, g1, g2, g3, a0, a1, a2, a3, bf0, bf1);
                }
            } else {
            // Register decode. Lane (gid, lid) needs, per k-step word of row r, the code pair for
            // k = 2*lid, 2*lid+1 (byte lid>>1, nibble lid&1) -> a0/a1, and the pair 8 k further on
            // (byte (lid>>1)+2, same nibble) -> a2/a3. One shift by a per-lane constant brings both
            // nibbles to bits 0 and 16; spreading the four 2-bit codes into four bytes gives
            // exactly the selector input the PTQ1_0 path feeds ternary_pair_selectors, and two byte
            // permutes against kTritPairLo/kTritPairHi produce the bf16 pairs {c-1, c'-1}.
            // The group's 32 code bytes per row are fetched once with two 16-byte loads
            // (conflict-free: the 144-byte row pitch puts the eight rows on disjoint bank quads).
            const unsigned pair_shift = 8u * static_cast<unsigned>(lid >> 1) +
                                        4u * static_cast<unsigned>(lid & 1);
            unsigned words[2][8];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const std::uint8_t* row = &codes_sh[wbuf][warp][gid + 8 * r][group_in_chunk * 32];
                const uint4 q0 = load_vec<uint4>(row);
                const uint4 q1 = load_vec<uint4>(row + 16);
                words[r][0] = q0.x; words[r][1] = q0.y; words[r][2] = q0.z; words[r][3] = q0.w;
                words[r][4] = q1.x; words[r][5] = q1.y; words[r][6] = q1.z; words[r][7] = q1.w;
            }
            const auto decode_pair = [pair_shift](unsigned word, unsigned& near, unsigned& far) {
                const unsigned t      = (word >> pair_shift) & 0x000F000Fu;
                const unsigned spread = (t & 0x00030003u) | ((t & 0x000C000Cu) << 6);
                const unsigned sel    = ternary_pair_selectors(spread);
                near                  = __byte_perm(kTritPairLo, kTritPairHi, sel);
                far                   = __byte_perm(kTritPairLo, kTritPairHi, sel >> 16);
            };

#pragma unroll
            for (int step = 0; step < 8; ++step) {
                const int kstep = group_in_chunk * 8 + step;

                // A operand: rows gid and gid+8, each contributing k and k+8 pairs.
                unsigned a0;
                unsigned a1;
                unsigned a2;
                unsigned a3;
                decode_pair(words[0][step], a0, a2);
                decode_pair(words[1][step], a1, a3);

                // B operand: the activation tile is already n-major (token rows, k contiguous),
                // which is exactly the .col layout the mma wants -- no transpose needed. Lanes past
                // the live token rows re-read the last live row (their columns are never stored).
                unsigned bf0 = 0u;
                unsigned bf1 = 0u;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&act_sh[buf][min(lane & 7, kActRows - 1)]
                                            [kstep * 16 + ((lane >> 3) & 1) * 8]));

                mma_bf16(g0, g1, g2, g3, a0, a1, a2, a3, bf0, bf1);
            }
            }

            // One PQ2_0 group spans eight k-steps, so its scale is applied once, per row.
            const float top = __half2float(
                __ushort_as_half(scale_sh[buf][warp][gid][group_in_chunk]));
            const float bottom = __half2float(
                __ushort_as_half(scale_sh[buf][warp][gid + 8][group_in_chunk]));
            acc0 = fmaf(g0, top, acc0);
            acc1 = fmaf(g1, top, acc1);
            acc2 = fmaf(g2, bottom, acc2);
            acc3 = fmaf(g3, bottom, acc3);
        }

            __syncthreads(); // the next stage reuses this buffer
        }

        // C fragment: c0/c1 are rows gid, columns 2*lid / 2*lid+1; c2/c3 are rows gid+8.
        const int token_a = tok_base + 2 * lid;
        const int token_b = token_a + 1;

        if (row_lo < rows) {
            if (token_a < tokens) {
                out[static_cast<std::int64_t>(token_a) * out_row_stride + row_lo] =
                    __float2bfloat16_rn(acc0);
            }
            if (token_b < tokens) {
                out[static_cast<std::int64_t>(token_b) * out_row_stride + row_lo] =
                    __float2bfloat16_rn(acc1);
            }
        }
        if (row_hi < rows) {
            if (token_a < tokens) {
                out[static_cast<std::int64_t>(token_a) * out_row_stride + row_hi] =
                    __float2bfloat16_rn(acc2);
            }
            if (token_b < tokens) {
                out[static_cast<std::int64_t>(token_b) * out_row_stride + row_hi] =
                    __float2bfloat16_rn(acc3);
            }
        }
    }
}

} // namespace ninfer::ops::detail
