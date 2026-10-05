#pragma once

// Set to 1 only to run the LUT-cost ablation described at the decode-table read below.
#ifndef TERNARY_WIDE_ABLATE_LUT
#define TERNARY_WIDE_ABLATE_LUT 0
#endif

// Ternary PQ2_0 tensor-core path for the LARGE-T regime (prefill, and any batched
// multi-sequence pass): the weight window is staged once per K-chunk and every token
// in the tile streams past it, instead of the small-t kernel's "walk 8 tokens, then
// re-stage the whole weight window".
//
// WHY THIS EXISTS -- the weight-pass law (measured 2026-09-20)
// The small-t kernel puts TOKENS in the outer loop and restages the weight window for
// every 8-token tile, so one forward pass reads the entire weight tensor
// ceil(T/8) times. Measured on the real engine: prefill at T=62 costs 131 ms
// (= 8 passes x 6.70 GiB = 438 GiB/s), and the fitted law over a 14x range of T is
//
//     t ~= 12.31 ms * ceil(T/8) + 44 ms          (<= 6% error, 544 GiB/s at the slope)
//
// while the official runtime on this same card reads the weights about once
// (pp512 = 2087 t/s against our 471-630). The harness carries the same fingerprint:
// the small-t kernel measures 14.56 / 16.50 / 15.58 ms at T = 1 / 3 / 8 -- one pass
// each, flat, which a compute-bound kernel could not be -- and then 22.66 -> 90.00 ms
// from T = 16 to 64, i.e. 2 -> 8 passes, exactly 3.97x against the predicted 4x.
// The same law explains the other two oddities: 5-lane concurrency at 25% efficiency
// (11 tokens do not fit one 8-token tile, so it pays 2 passes) and why attaching MTP
// paid off immediately (a T=3 verify pass already rides one pass).
//
// STRUCTURE -- the one thing two independent sources agree on
//   * A operand (weights): staged once per K-chunk, then reused by ALL tokens.
//   * B operand (activations): double buffered.
//   * K is the outer loop; the token sub-tile is inner.
// Upstream ninfer's q4/q8 rowsplit MMA has exactly this shape (its As[BM*BK] carries
// no stage dimension while Bs[S][BN*BK] does), and Marlin states the same preference
// from the other side (activations served from L2, weights loaded asynchronously and
// evicted immediately so they never pollute it). The small-t kernel here is the
// opposite of both, which is why it caps prefill at one weight pass per 8 tokens.
//
// WHY BK IS ONE GROUP (128) AND NOT 512
// The activation tile costs BK*2 bytes per token per stage: at BK=512 that is
// 1 KB/token/stage, so 48 KB of static shared caps the token tile at ~10 -- which is
// exactly why the shipped kernel chose 8 tokens. At BK=128 it costs 272 B/token/stage,
// which is what buys a 64-token tile. Shared budget (stages = 2):
//   codes 2*4*16*48 = 6144 | act 2*64*136*2 = 34816 | lut 2048 | scales 2*4*16*2 = 256
//   total 43264 B = 42.25 KB  <=  48 KB static  =>  2 CTAs/SM on sm_89 (100 KB/SM).
// (The sizing formula is validated against the shipped kernel: it predicts the
// 38144 B that ptxas reports for the small-t configuration.)
//
// LAYOUT CONTRACTS (same as the small-t kernel -- violating them makes T == 1 look
// perfect while every prefill scrambles):
//   * activations are TOKEN-major: (column, token) lives at token*k + column.
//   * output is token-major too: (row, token) at token*out_row_stride + row.
//   * PQ2_0 packs four 2-bit codes per byte and code c means value (c - 1); the group
//     scale is applied AFTER the mma, per row. With BK == group size there is exactly
//     one scale application per chunk, which is also why that is the natural chunk.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {

inline constexpr int kTernaryWideRowsPerWarp = 16;
// 4 warps (64 rows) per CTA -- the MEASURED optimum of the configurations tried on
// 2026-09-20, with the reason recorded, because the naive reading of the profiler says
// otherwise and would send the next person down the same dead end:
//   * Nsight Compute on this build: Active Warps/Scheduler 2.72 (max 12), No Eligible 77.3%,
//     12 warp-cycles per issued instruction, "Est. Local Speedup 52.49% from occupancy",
//     DRAM at only 19.7%. It reads like a pure occupancy problem.
//   * Raising occupancy to 16 warps/SM (8 warps/CTA + a 128-register cap) made everything
//     WORSE: T=64 went 52.88 -> 70.89 ms (+34%). The register cap is what did it -- this
//     kernel's accumulators alone are acc[8][4] + tmp[8][4] = 64 floats per thread, so the
//     cap spilled (40 B spill stores / 32 B spill loads) and the local-memory traffic cost
//     more than the four extra warps bought.
//   * Occupancy here is NOT free: it has to be earned by making the kernel need fewer
//     registers, not by capping them. The way to do that is a half-tile structure (run 32 of
//     the 64 tokens at a time while keeping the weight panel staged for both halves), which
//     halves acc[] and tmp[] to 16 floats each and leaves the pass count at ceil(T/64).
//     That is the open TODO. Do NOT simply raise the warp count and cap registers.
inline constexpr int kTernaryWideWarpsPerCta = 4;
inline constexpr int kTernaryWideRowsPerCta  = kTernaryWideRowsPerWarp * kTernaryWideWarpsPerCta;
// Token tile. Evidence for the value: upstream's per-T rungs keep BN=32 for the
// T ~ 12..110 band and reach for 64 only in the middle band and 128 at the very top;
// 64 is the widest that still fits 2 CTAs/SM here, and this kernel exists to serve
// prefill, not the verify pass.
inline constexpr int kTernaryWideTokens   = 64;
// K consumed per staged chunk == one PQ2_0 group.
inline constexpr int kTernaryWideChunkK   = PQ2RowSplitStorage::kGroupK;
inline constexpr int kTernaryWideKSteps   = kTernaryWideChunkK / 16;   // mma k-steps per chunk
inline constexpr int kTernaryWideCodesPerRow = kTernaryWideChunkK / 4; // four codes per byte
// 32 used bytes, padded to 48 so the four-byte words the quad reads land on distinct
// banks across the eight rows one load touches (12 words apart => 12*r mod 32 is
// distinct for r = 0..7, the same argument the small-t kernel makes for its stride).
inline constexpr int kTernaryWideRowStride = 48;
// Raw PTQ1_0 staging row: the 24 base bytes + the 2 high bytes, padded to 32. The pad is what
// makes every row start 8-byte aligned, which is the alignment a cp.async of that size needs
// (see stage_weights), and it keeps the row inside one bank-friendly power of two.
inline constexpr int kTernaryWideRawStride = 32;
inline constexpr int kTernaryWideActStride = kTernaryWideChunkK + 8;   // 16B-aligned, padded
inline constexpr int kTernaryWideStages    = 2;
inline constexpr int kTernaryWideThreads   = kTernaryWideWarpsPerCta * 32;
// Minimum resident CTAs per SM, i.e. the occupancy ptxas must budget registers for.
//
// This is the number the 2026-09-20 occupancy experiment pinned down. At BN=64 the first
// version of this kernel used 166 registers and 43.3 KB of shared, landed at 2 CTAs/SM
// (8 warps/SM) and measured 53.3 ms per weight pass -- about 31% of this card's FLOP peak,
// meaning it was LATENCY bound, not bandwidth bound (the same pass moved 6.65 GiB, i.e. an
// effective ~125 GiB/s against the ~595 GiB/s the card actually delivers on a streaming
// read). Holding everything else fixed and dropping only the token tile to 32 cut shared to
// 25.9 KB, raised occupancy to 12 warps/SM and took that same pass to 32.6 ms -- +63%
// per-pass efficiency from occupancy alone.
//
// So the target is 3 CTAs/SM while KEEPING BN=64 (the wide tile is what keeps the pass
// count at ceil(T/64), and at T=64 that pass count matters more than the per-pass win:
// BN=32 at T=64 costs 2 x 32.6 = 65.2 ms against 53.3 ms for one 64-wide pass). Three CTAs
// needs shared <= ~33 KB, which is why the activation tile is single buffered, and
// registers <= 170, which is why this constant exists.
inline constexpr int kTernaryWideMinBlocksPerSm = 3;
// Below this the 64-wide tile wastes more than it saves; the small-t kernel (one pass
// up to 8 tokens) is the right rung and stays the validated default for the verify pass.
//
// 2026-09-20 (SECOND measurement): 41, not 9. The original 9 counted only the number of
// passes each kernel needs and missed that ONE wide pass costs ~5x one small pass
// (53.8 ms vs 10.2 ms over the artifact's real shape mix). Measured with the token sweep
// extended to cover the 16..64 gap (tscale_bench.cu -> E:\infer-build\tscale_thr.txt):
//
//   T     small_t   wide_t     winner
//    9     20.29     50.69     small 2.50x
//   11     19.67     48.59     small 2.47x   <- where a multi-lane decode lands
//   16     20.84     49.24     small 2.36x
//   24     29.74     53.75     small 1.81x
//   32     40.75     54.54     small 1.34x
//   40     51.02     54.74     small 1.07x   <- last token count small_t wins
//   48     60.90     54.79     wide  1.11x
//   64     83.60     53.81     wide  1.55x
//
// small_t == 10.2 ms * ceil(T/8) and wide_t is flat at ~54 ms, so the crossover is where
// ceil(T/8) = 6, i.e. T = 41. At 9 every pass of 9..40 tokens ran the slower kernel, and
// that band is exactly where a batched pass lands (the linear op sees
// T = tokens-per-lane * active lanes, so 3+ lanes with MTP drafts sit inside it).
inline constexpr int kTernaryWideMinTokens = 41;

// Runtime override, for the same-binary A/B this threshold requires: decode speed has to
// be judged by the engine, not by the harness (the harness ranks T=1 mma 34% faster and
// the engine measures it 3.1% slower -- see the note in ternary_rowsplit_gemm.cu).
//   NINFER_TERNARY_WIDE_MIN_TOKENS=<n>   candidate threshold; unset keeps the default.
// Read once: this decides which kernel enters the captured CUDA graph, so it must not
// change between capture and replay (same rule as NINFER_TERNARY_MMA / _HADAMARD).
[[nodiscard]] inline int ternary_wide_min_tokens() {
    static const int value = [] {
        const char* text = std::getenv("NINFER_TERNARY_WIDE_MIN_TOKENS");
        if (text == nullptr) { return kTernaryWideMinTokens; }
        const int parsed = std::atoi(text);
        return parsed > 0 ? parsed : kTernaryWideMinTokens;
    }();
    return value;
}

static_assert((kTernaryWideRowStride % 16) == 0, "cp.async needs a 16-byte aligned row start");
static_assert(kTernaryWideRowStride >= kTernaryWideCodesPerRow, "the row must hold a whole chunk");
static_assert((kTernaryWideActStride % 8) == 0, "ldmatrix wants 8-element row starts");
static_assert((kTernaryWideChunkK % PQ2RowSplitStorage::kGroupK) == 0,
              "a chunk must be a whole number of groups");
static_assert(kTernaryWideChunkK == PQ2RowSplitStorage::kGroupK,
              "this kernel applies one scale per chunk, so a chunk IS one group");

union TernaryWidePairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

struct TernaryWideStorage {
    uint2 lut[256];
    alignas(16) std::uint8_t codes[kTernaryWideStages][kTernaryWideWarpsPerCta]
                                  [kTernaryWideRowsPerWarp][kTernaryWideRowStride];
    // Raw PTQ1_0 staging window, cp.async'd and THEN repacked (see repack_ptq1): one 24-byte
    // qs plus a 2-byte qh per (warp, row), per stage. Only the kPtq1 instantiation touches it.
    // 2*4*16*32 = 4 KB, taking the CTA from 25.25 KB to 29.25 KB -- 3 * 29.25 KB = 87.75 KB
    // still fits the 100 KB/SM sm_89 has, so kTernaryWideMinBlocksPerSm stays reachable.
    alignas(16) std::uint8_t raw[kTernaryWideStages][kTernaryWideWarpsPerCta]
                                [kTernaryWideRowsPerWarp][kTernaryWideRawStride];
    // Activation: SINGLE buffer, deliberately. This is what takes shared from 43.3 KB to
    // 25.25 KB and therefore what buys the third CTA per SM (see kTernaryWideMinBlocksPerSm).
    // It costs the overlap of one 16 KB activation copy per chunk, paid against ~1.3 ms of
    // compute per chunk at BN=64 -- under 1%. The weight window above keeps its double
    // buffer, because that is the operand whose latency actually needs hiding.
    alignas(16) __nv_bfloat16 act[kTernaryWideTokens][kTernaryWideActStride];
    std::uint16_t scales[kTernaryWideStages][kTernaryWideWarpsPerCta][kTernaryWideRowsPerWarp];
};

// kPtq1 == false: `codes` are PQ2_0 (32 code bytes per row per group, positional: byte b holds
//                 weights 4b..4b+3).
// kPtq1 == true : `codes` are PTQ1_0 (24 base bytes) and `high` carries its 2 extra bytes. Staging
//                 is TWO steps, so the global latency is not on the critical path: stage_weights
//                 cp.async's the raw 24+2-byte row into `raw` one chunk ahead of use, and
//                 repack_ptq1() turns it into the same in-shared PQ2 code layout after cp_wait.
//                 Every line below the repack is byte for byte the PQ2 path (LUT, mma, scale
//                 folding, stores). The repack is bit-exact: verified against real weights with
//                 both verified decoders (exp\reader-candidates\verify_pq2_repack.py ->
//                 0 / 122,880 mismatches, and the same run shows the two packings hold the same
//                 trits with no code==3).
template <bool kPtq1>
__global__ __launch_bounds__(kTernaryWideThreads, kTernaryWideMinBlocksPerSm)
void ternary_wide_t_kernel(const __nv_bfloat16* __restrict__ x,
                           const std::uint8_t* __restrict__ codes,
                           const std::uint8_t* __restrict__ high,
                           const std::uint8_t* __restrict__ scales,
                           __nv_bfloat16* __restrict__ out, std::int32_t rows,
                           std::int32_t k, std::int32_t tokens,
                           std::int32_t out_row_stride,
                           // SCHED3 CHANGE 3 on the PREFILL rung (this is the one a T >= 41 call
                           // takes; small_t covers the 9..40 band). See the long note in
                           // ternary_rowsplit_mma_small_t.cuh for the argument; the shape of the
                           // edit is identical.
                           bool token_grid) {
    __shared__ TernaryWideStorage staging;
    auto& lut      = staging.lut;
    auto& codes_sh = staging.codes;
    auto& raw_sh   = staging.raw;
    auto& act_sh   = staging.act;
    auto& scale_sh = staging.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2; // 0..7 -- mma row within the fragment
    const int lid  = lane & 3;  // 0..3 -- mma k pair / column pair index

    // Decode table: packed byte -> four ternary weights in bf16. Code c means (c - 1),
    // so the multiply is exact (+-1 is a sign change). Read only after the first
    // __syncthreads() below.
    for (int i = tid; i < 256; i += kTernaryWideThreads) {
        TernaryWidePairBits low;
        TernaryWidePairBits high;
        low.pair  = __floats2bfloat162_rn(static_cast<float>((i & 3) - 1),
                                         static_cast<float>(((i >> 2) & 3) - 1));
        high.pair = __floats2bfloat162_rn(static_cast<float>(((i >> 4) & 3) - 1),
                                          static_cast<float>(((i >> 6) & 3) - 1));
        uint2 entry;
        entry.x = low.bits;
        entry.y = high.bits;
        lut[i]  = entry;
    }

    const int row0 = static_cast<int>(blockIdx.x) * kTernaryWideRowsPerCta;
    if (row0 >= rows) { return; }

    constexpr int kCodeBytes =
        kPtq1 ? PTQ1RowSplitStorage::kCodeBytesPerGroup : PQ2RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kHighBytes = kPtq1 ? PTQ1RowSplitStorage::kHighBytesPerGroup : 0;
    const std::int32_t groups_per_row = k / PQ2RowSplitStorage::kGroupK;
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kCodeBytes;
    const std::int64_t high_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * kHighBytes;
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * PQ2RowSplitStorage::kScaleBytesPerGroup;
    const std::int32_t chunks = k / kTernaryWideChunkK;
    const std::int64_t warp_row_base =
        static_cast<std::int64_t>(row0) + static_cast<std::int64_t>(warp) * kTernaryWideRowsPerWarp;

    // Weight staging (double buffered): one chunk = this warp's 16 x 32-byte code window
    // plus its 16 per-row group scales.
    const auto stage_weights = [&](int buf, int chunk) {
        if constexpr (kPtq1) {
            // PREFETCH ONLY. The repack is now repack_ptq1() below, which runs after these
            // copies have landed. Splitting the two is the entire point of this rewrite: the
            // 24+2-byte source row used to be read with plain loads whose latency sat directly
            // on top of the dependent repack arithmetic (LDG -> the byte_perm chain -> shared
            // store), so every chunk paid a full global round trip before it could store a
            // single code byte. Now the global read is an asynchronous copy issued one chunk
            // ahead (hidden behind the previous chunk's mma) and the repack reads SHARED.
            //
            // COPY WIDTH: qs is 24 bytes and the per-chunk stride is exactly 24 bytes, so an
            // odd chunk starts at an address that is 8-byte but NOT 16-byte aligned. The copy
            // is therefore three 8-byte cp.async rather than the 16+8 the 24-byte size
            // suggests -- a 16-byte cp.async on an odd chunk is a misaligned copy. 8-byte
            // alignment always holds: the chunk stride is 24 (24 % 8 == 0) and so is a row
            // (groups_per_row * 24).
            constexpr int kQsCopies     = PTQ1RowSplitStorage::kCodeBytesPerGroup / 8; // 3
            constexpr int kQsCopiesWarp = kTernaryWideRowsPerWarp * kQsCopies;
            static_assert(PTQ1RowSplitStorage::kCodeBytesPerGroup % 8 == 0,
                          "a PTQ1_0 row copy is 8-byte granular, that is what cp.async needs");
            static_assert(kTernaryWideRawStride >= PTQ1RowSplitStorage::kCodeBytesPerGroup,
                          "the raw row must hold a whole qs");
#pragma unroll
            for (int copy = lane; copy < kQsCopiesWarp; copy += 32) {
                const int local_row           = copy / kQsCopies;
                const int byte_off            = (copy % kQsCopies) * 8;
                const std::int64_t global_row = warp_row_base + local_row;
                const bool valid              = global_row < rows;
                // An invalid row zero-fills (src_bytes = 0) instead of branching to a plain
                // shared store: the repack then produces zeros by construction, exactly what
                // the old `valid ? repack : 0u` wrote, and it needs no validity test at all.
                // The fallback source is the plane base, which is always mapped -- with
                // src_bytes == 0 nothing is read from it.
                cp_async_zfill<8, Cache::ca>(
                    &raw_sh[buf][warp][local_row][byte_off],
                    valid ? codes + global_row * code_row_bytes +
                                static_cast<std::int64_t>(chunk) * kCodeBytes + byte_off
                          : codes,
                    valid ? 8 : 0);
            }
            // qh is only 2 bytes, so it is the one part that must NOT go through cp.async: the
            // smallest copy is 4 bytes, which would read past the end of the high plane on the
            // last row's last group. One plain load + plain shared store per row per chunk.
            if (lane < kTernaryWideRowsPerWarp) {
                const std::int64_t global_row = warp_row_base + lane;
                const std::uint16_t qh =
                    (global_row < rows)
                        ? load_vec<std::uint16_t>(high + global_row * high_row_bytes +
                                                  static_cast<std::int64_t>(chunk) * kHighBytes)
                        : 0u;
                // 24 is even and the raw row is 32 bytes, so this 2-byte store is naturally aligned.
                store_vec<std::uint16_t>(
                    reinterpret_cast<std::uint16_t*>(
                        &raw_sh[buf][warp][lane][PTQ1RowSplitStorage::kCodeBytesPerGroup]),
                    qh);
            }
        } else {
        // Codes: kTernaryWideCodesPerRow bytes per row as 16-byte copies, round-robin
        // across the lanes. Written generically (never sized by hand for one chunk
        // width) because a hand-sized loop silently stages only PART of each row at a
        // different width and presents as a fast kernel producing garbage.
        constexpr int kCopiesPerRow  = kTernaryWideCodesPerRow / 16;
        constexpr int kCopiesPerWarp = kTernaryWideRowsPerWarp * kCopiesPerRow;
#pragma unroll
        for (int copy = lane; copy < kCopiesPerWarp; copy += 32) {
            const int local_row = copy / kCopiesPerRow;
            const int byte_off  = (copy % kCopiesPerRow) * 16;
            const std::int64_t global_row = warp_row_base + local_row;
            std::uint8_t* dst             = &codes_sh[buf][warp][local_row][byte_off];
            if (global_row < rows) {
                cp_async<16, Cache::cg>(
                    dst, codes + global_row * code_row_bytes +
                             static_cast<std::int64_t>(chunk) * kTernaryWideCodesPerRow + byte_off);
            } else {
                store_vec<std::uint8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }
        }

        { // one scale per (row, chunk), because a chunk is exactly one group
            if (lane < kTernaryWideRowsPerWarp) {
                const std::int64_t global_row = warp_row_base + lane;
                std::uint16_t value           = 0u;
                if (global_row < rows) {
                    value = load_vec<std::uint16_t>(
                        scales + global_row * scale_row_bytes +
                        static_cast<std::int64_t>(chunk) * PQ2RowSplitStorage::kScaleBytesPerGroup);
                }
                scale_sh[buf][warp][lane] = value;
            }
        }
    };

    // REPACK: raw (24-byte qs + 2-byte qh) -> the in-shared PQ2 code layout, so every line
    // downstream of this step is byte for byte the PQ2 path. Called one chunk after
    // stage_weights issued the copies, i.e. after cp_wait, when the global latency has already
    // been absorbed by the previous chunk's mma.
    //
    // THE ARITHMETIC IS UNCHANGED. It is bit-exact against the verified decoders
    // (E:\infer-build\exp\reader-candidates\verify_pq2_repack.py -> 0 / 122,880 mismatches) and
    // the only edits here are WHERE the bytes come from (shared instead of global) and the loss
    // of the validity test, which the prefetch's zero-fill made redundant -- a zeroed raw row
    // repacks to a zeroed code row by construction.
    //
    // Seven jobs per row, flattened over the warp's 32 lanes; each job owns a DISJOINT set of
    // the 32 destination code bytes.
    //   slot 0..3: qs[4g..4g+4)      at stages 0..4 -> code bytes 4t+g    (weights 0..79)
    //   slot 4..5: qs[16+4p..20+4p)  at stages 0..4 -> code bytes 20+2t+p (weights 80..119)
    //   slot 6   : qh[0..2) -> byte 30 (trits 0,0,1,1) and byte 31 (trits 2,2,3,3)
    // code_of() is PTQ1SimtDecodeAtom's xi = ((uint8)(raw * 3^t) * 3) >> 8, which IS the PQ2
    // code (PQ2 decodes as (code - 1) * d), so no sign juggling is needed anywhere.
    const auto repack_ptq1 = [&](int buf) {
        if constexpr (kPtq1) {
            const auto code_of = [](std::uint8_t raw, int pow3) {
                const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3);
                return (static_cast<unsigned>(q) * 3u) >> 8;
            };
            // Vectorised form of the same arithmetic, copied from PrismML's PTQ1_0 MMQ tile loader
            // (ggml/src/ggml-cuda/mmq-load-tiles.cuh:261 ggml_cuda_mmq_decode_ptq1_0_qs4, MIT):
            // the two __byte_perm calls park the four code bytes in the even byte lanes so that ONE
            // multiply-by-3 advances all four base-3 digits at once, and the odd byte of each product
            // is exactly ((uint8)(raw * 3^t) * 3) >> 8 -- i.e. the PQ2 code itself. Five iterations
            // give all five stages of one 4-byte group; the pack folds the four codes into one byte.
            const auto stage_codes = [](unsigned group4, int stage) {
                unsigned v_lo = __byte_perm(group4, 0u, 0x4140);
                unsigned v_hi = __byte_perm(group4, 0u, 0x4342);
                unsigned word = 0u;
#pragma unroll
                for (int t = 0; t <= stage; ++t) {
                    const unsigned w_lo = v_lo * 3u;
                    const unsigned w_hi = v_hi * 3u;
                    v_lo                = w_lo & 0x00FF00FFu;
                    v_hi                = w_hi & 0x00FF00FFu;
                    word                = __byte_perm(w_lo, w_hi, 0x7531);
                }
                return static_cast<std::uint8_t>((word & 3u) | ((word >> 6) & 0x0Cu) |
                                                 ((word >> 12) & 0x30u) | ((word >> 18) & 0xC0u));
            };
#pragma unroll
            for (int job = lane; job < kTernaryWideRowsPerWarp * 7; job += 32) {
                const int local_row     = job / 7;
                const int slot          = job - local_row * 7;
                const std::uint8_t* src = &raw_sh[buf][warp][local_row][0];
                std::uint8_t* dst       = &codes_sh[buf][warp][local_row][0];
                if (slot < 4) {
                    const unsigned g4 = load_vec<unsigned>(src + 4 * slot);
#pragma unroll
                    for (int t = 0; t < 5; ++t) {
                        dst[4 * t + slot] = stage_codes(g4, t);
                    }
                } else if (slot < 6) {
                    const int par     = slot - 4;
                    const unsigned g4 = load_vec<unsigned>(src + 16 + 4 * par);
#pragma unroll
                    for (int t = 0; t < 5; ++t) {
                        dst[20 + 2 * t + par] = stage_codes(g4, t);
                    }
                } else {
                    const std::uint8_t h0 = src[PTQ1RowSplitStorage::kCodeBytesPerGroup];
                    const std::uint8_t h1 = src[PTQ1RowSplitStorage::kCodeBytesPerGroup + 1];
                    dst[30] = static_cast<std::uint8_t>(code_of(h0, 1) | (code_of(h1, 1) << 2) |
                                                        (code_of(h0, 3) << 4) | (code_of(h1, 3) << 6));
                    dst[31] = static_cast<std::uint8_t>(code_of(h0, 9) | (code_of(h1, 9) << 2) |
                                                        (code_of(h0, 27) << 4) | (code_of(h1, 27) << 6));
                }
            }
        }
    };

    // Activation staging (single buffer, whole CTA cooperates): the
    // kTernaryWideTokens x kTernaryWideChunkK tile for one chunk, 16 bytes (8 bf16) per
    // copy. Offsets are in bf16 ELEMENTS: x is a __nv_bfloat16* while the code staging
    // above is byte-based, and mixing the two over-advances the source window by 2x
    // (values plausible, NaN at the tail). Tokens past the end of the sequence are
    // zero-filled, so the mma sees a defined activation and the guarded store drops the
    // result.
    const auto stage_act = [&](int chunk, int tok_base) {
        constexpr int kCopiesPerToken = kTernaryWideChunkK / 8;
        constexpr int kActCopies      = kTernaryWideTokens * kCopiesPerToken;
        for (int copy = tid; copy < kActCopies; copy += kTernaryWideThreads) {
            const int tok        = copy / kCopiesPerToken;
            const int elem       = (copy % kCopiesPerToken) * 8;
            const int global_tok = tok_base + tok;
            const int safe_tok   = (global_tok < tokens) ? global_tok : tok_base;
            cp_async_zfill<16>(
                &act_sh[tok][elem],
                x + static_cast<std::int64_t>(safe_tok) * k +
                    static_cast<std::int64_t>(chunk) * kTernaryWideChunkK + elem,
                (global_tok < tokens) ? 16 : 0);
        }
    };

    // Walk the sequence in tiles of kTernaryWideTokens TOKENS (outer), and inside each
    // tile walk K (inner). This is the whole point of the kernel: the weight window for
    // a chunk is staged once and every token sub-tile consumes it, so the pass count is
    // ceil(T / kTernaryWideTokens) instead of ceil(T / 8).
    const int row_lo = row0 + warp * kTernaryWideRowsPerWarp + gid;
    const int row_hi = row_lo + 8;
    constexpr int kSubTiles = kTernaryWideTokens / 8;

    // SCHED3 CHANGE 3: the token tile is grid.y when the switch is on (one tile per CTA); with
    // NINFER_TERNARY_TOKEN_GRID=0 the tiles are walked serially in this CTA, which is the
    // pre-change behaviour and the A/B control arm. The tile body below is identical either way --
    // its accumulators are declared and zeroed inside the loop, and the weight window is re-staged
    // from the top of K for every tile, so tiles share no state and re-partitioning them cannot
    // reorder a single addition.
    const int tok_tiles = token_grid ? 1 : ((tokens + kTernaryWideTokens - 1) / kTernaryWideTokens);
    const int tok_base0 = token_grid ? static_cast<int>(blockIdx.y) * kTernaryWideTokens : 0;
    if (tok_base0 >= tokens) { return; }

    for (int tok_tile = 0; tok_tile < tok_tiles; ++tok_tile) {
        const int tok_base = tok_base0 + tok_tile * kTernaryWideTokens;
        float acc[kSubTiles][4];
#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[sub][i] = 0.0f; }
        }

        // Prologue: the first chunk's weights and activations, then one full wait. From here
        // the weight window is double buffered and the activation is not (see the struct).
        stage_weights(0, 0);
        stage_act(0, tok_base);
        cp_commit();
        cp_wait<0>();
        if constexpr (kPtq1) {
            // cp.async completion is only ordered for the thread that ISSUED the copy, while the
            // repack reads raw bytes that other lanes of this warp copied -- so one warp fence
            // first. The codes_sh writes the repack makes are read by this same warp in the loop
            // below, and the __syncthreads() right after orders them for every lane.
            __syncwarp();
            repack_ptq1(0);
        }
        __syncthreads();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int buf = chunk & 1;
            // Issue the next weight window straight away: its latency hides behind the
            // compute below, which is the whole reason this operand gets two buffers.
            if (chunk + 1 < chunks) {
                stage_weights(buf ^ 1, chunk + 1);
                cp_commit();
            }

            // One group per chunk, so the mma result for the whole chunk takes a single
            // scale per row: accumulate the k-steps in a temporary, then fold into the
            // running accumulator.
            float tmp[kSubTiles][4];
#pragma unroll
            for (int sub = 0; sub < kSubTiles; ++sub) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { tmp[sub][i] = 0.0f; }
            }

#pragma unroll
            for (int step = 0; step < kTernaryWideKSteps; ++step) {
                const int word_off = step * 4; // one k-step of 16 codes == one 4-byte word

                // A operand: rows gid and gid+8, each contributing k and k+8 pairs.
                const unsigned word_lo =
                    load_vec<unsigned>(&codes_sh[buf][warp][gid][word_off]);
                const unsigned word_hi =
                    load_vec<unsigned>(&codes_sh[buf][warp][gid + 8][word_off]);
                const unsigned shift_near = 8u * static_cast<unsigned>(lid >> 1);
                const unsigned shift_far  = 8u * static_cast<unsigned>((lid >> 1) + 2);

#if TERNARY_WIDE_ABLATE_LUT
                // ABLATION (measurement only, NOT a shipping configuration): replace the four
                // shared-memory decode-table reads with constants. The mma and ldmatrix work is
                // unchanged, so the difference against the normal build is exactly what the LUT
                // path costs. Results are wrong by construction; read the aggregate timings only.
                const uint2 near_lo = make_uint2(0x3F803F80u, 0x3F803F80u);
                const uint2 far_lo  = near_lo;
                const uint2 near_hi = near_lo;
                const uint2 far_hi  = near_lo;
#else
                const uint2 near_lo = lut[(word_lo >> shift_near) & 0xFFu];
                const uint2 far_lo  = lut[(word_lo >> shift_far) & 0xFFu];
                const uint2 near_hi = lut[(word_hi >> shift_near) & 0xFFu];
                const uint2 far_hi  = lut[(word_hi >> shift_far) & 0xFFu];
#endif

                // A quad of lanes shares a byte; lanes with an odd lid need its high pair.
                const bool odd = (lid & 1) != 0;
                const unsigned a0 = odd ? near_lo.y : near_lo.x;
                const unsigned a1 = odd ? near_hi.y : near_hi.x;
                const unsigned a2 = odd ? far_lo.y : far_lo.x;
                const unsigned a3 = odd ? far_hi.y : far_hi.x;

#pragma unroll
                for (int sub = 0; sub < kSubTiles; ++sub) {
                    // B operand: the activation tile is already n-major (token rows, k
                    // contiguous) -- exactly the .col layout the mma wants, no transpose.
                    unsigned bf0 = 0u;
                    unsigned bf1 = 0u;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&act_sh[sub * 8 + (lane & 7)]
                                                [step * 16 + ((lane >> 3) & 1) * 8]));
                    mma_bf16(tmp[sub][0], tmp[sub][1], tmp[sub][2], tmp[sub][3], a0, a1, a2,
                             a3, bf0, bf1);
                }
            }

            const float top =
                __half2float(__ushort_as_half(scale_sh[buf][warp][gid]));
            const float bottom =
                __half2float(__ushort_as_half(scale_sh[buf][warp][gid + 8]));
#pragma unroll
            for (int sub = 0; sub < kSubTiles; ++sub) {
                acc[sub][0] = fmaf(tmp[sub][0], top, acc[sub][0]);
                acc[sub][1] = fmaf(tmp[sub][1], top, acc[sub][1]);
                acc[sub][2] = fmaf(tmp[sub][2], bottom, acc[sub][2]);
                acc[sub][3] = fmaf(tmp[sub][3], bottom, acc[sub][3]);
            }

            __syncthreads(); // every read of the single activation buffer is done
            if (chunk + 1 < chunks) {
                stage_act(chunk + 1, tok_base); // reuse the activation buffer
                cp_commit();
                cp_wait<0>();                   // next chunk's weights AND activation landed
                if constexpr (kPtq1) {
                    // Repack the window this iteration prefetched (buf ^ 1 == (chunk+1) & 1) while
                    // nothing else needs this warp's lanes: its destination, codes_sh[buf ^ 1], is
                    // not read until the next iteration, and the __syncthreads() below publishes
                    // those shared writes before that happens. __syncwarp is the fence that makes
                    // this warp's own cp.async bytes readable by its other lanes (see the prologue).
                    __syncwarp();
                    repack_ptq1(buf ^ 1);
                }
                __syncthreads();
            }
        }

        // C fragment: c0/c1 are rows gid, columns 2*lid / 2*lid+1; c2/c3 are rows gid+8.
#pragma unroll
        for (int sub = 0; sub < kSubTiles; ++sub) {
            const int token_a = tok_base + sub * 8 + 2 * lid;
            const int token_b = token_a + 1;
            if (row_lo < rows) {
                if (token_a < tokens) {
                    out[static_cast<std::int64_t>(token_a) * out_row_stride + row_lo] =
                        __float2bfloat16_rn(acc[sub][0]);
                }
                if (token_b < tokens) {
                    out[static_cast<std::int64_t>(token_b) * out_row_stride + row_lo] =
                        __float2bfloat16_rn(acc[sub][1]);
                }
            }
            if (row_hi < rows) {
                if (token_a < tokens) {
                    out[static_cast<std::int64_t>(token_a) * out_row_stride + row_hi] =
                        __float2bfloat16_rn(acc[sub][2]);
                }
                if (token_b < tokens) {
                    out[static_cast<std::int64_t>(token_b) * out_row_stride + row_hi] =
                        __float2bfloat16_rn(acc[sub][3]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
