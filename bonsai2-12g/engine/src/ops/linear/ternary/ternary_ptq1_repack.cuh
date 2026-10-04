#pragma once

// THE shared PTQ1_0 -> PQ2_0 in-shared code repack (2026-09-23).
//
// WHY IT EXISTS: this arithmetic had grown a copy in the wide-t kernel, a second (faster, but
// bit-identical in output) copy in the small-t kernel, and now needs a third in the int8 rung. Three
// copies of a byte_perm/base-3 chain is exactly the shape of change that drifts silently -- and a
// drift here does not crash, it feeds the LUT/mma a wrong trit and shows up as a quality regression
// 30 minutes later. So the arithmetic lives here once, and every rung that repacks calls it.
//
// WHAT IT DOES: PTQ1_0 stores a 128-weight group as 24 base-3 bytes (five trits per byte, five
// stages of c=16/c=8/c=8...) plus 2 high bytes (the last 8 weights), while PQ2_0 stores the same
// group as 32 bytes of four 2-bit codes. The repack relabels one as the other, so every line
// downstream of it (LUT, mma, scale folding, stores) is byte for byte the PQ2_0 path.
//
// CORRECTNESS CONTRACT: the produced bytes must equal, for every weight index, the direct formula
//   code = ((uint8)(raw * 3^t) * 3) >> 8
// which is PTQ1SimtDecodeAtom's `xi` (ternary_rowsplit_storage.cuh:101) and therefore IS the PQ2_0
// code (PQ2_0 decodes as (code - 1) * d). Verified bit-exact against the two independent decoders
// over real weights: E:\infer-build\exp\reader-candidates\verify_pq2_repack.py -> 0 / 122,880
// mismatches, and (for the emit_stages() form below, all 256 byte values x 5 stages)
// E:\infer-build\exp\ptq1-mmasmallt\verify_smallt_repack.py. No sign juggling is needed anywhere.
//
// LAYOUT CONTRACT: `src` is the raw row -- qs[0..24) then qh[0..2) at offset
// PTQ1RowSplitStorage::kCodeBytesPerGroup; `dst` is the PQ2_0 code row, 32 bytes of which 0..31 are
// written by the seven slots below. src needs 4-byte alignment (the two 4-byte word loads), dst only
// byte alignment. The rows themselves are independent, so callers are free to distribute the
// (row, slot) jobs over a warp in any order -- see ternary_ptq1_repack_row().

#include "ops/common/memory.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

// Scalar form of the direct formula, used for the 2-byte qh tail.
__device__ __forceinline__ unsigned ternary_ptq1_code_of(std::uint8_t raw, int pow3) {
    const std::uint8_t q = static_cast<std::uint8_t>(raw * pow3);
    return (static_cast<unsigned>(q) * 3u) >> 8;
}

// One 4-byte qs group -> FIVE code bytes (all five stages), one chain step per stage instead of
// restarting the chain per stage: stage*_codes(g4, t) re-runs stages 0..t, so five calls execute
// 1+2+3+4+5 = 15 chain steps where this executes 5, for the same five bytes. `dst_step` is the
// distance between consecutive stages' destination bytes (4 for the qs[0..16) words, 2 for
// qs[16..24)) and is a literal at every call site, so it folds.
//
// The extraction (two __byte_perm calls parking the four base-3 bytes in the even lanes so ONE
// multiply-by-3 advances all four digits) and the multiply chain itself are PrismML's
// (ggml/src/ggml-cuda/mmq-load-tiles.cuh:261 ggml_cuda_mmq_decode_ptq1_0_qs4, MIT); the pack is the
// one-multiply fold: the four codes sit in bytes 0..3 of `word` with only their low two bits
// meaningful, and * 0x01041040 lands b0 + 4*b1 + 16*b2 + 64*b3 in the top byte (each field <= 3, so
// byte 2 peaks at 252 and cannot carry into it).
__device__ __forceinline__ void ternary_ptq1_emit_stages(std::uint8_t* dst, int dst_step,
                                                         unsigned group4) {
    unsigned v_lo = __byte_perm(group4, 0u, 0x4140);
    unsigned v_hi = __byte_perm(group4, 0u, 0x4342);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const unsigned w_lo = v_lo * 3u;
        const unsigned w_hi = v_hi * 3u;
        v_lo                = w_lo & 0x00FF00FFu;
        v_hi                = w_hi & 0x00FF00FFu;
        const unsigned word = __byte_perm(w_lo, w_hi, 0x7531);
        dst[t * dst_step]   = static_cast<std::uint8_t>(((word & 0x03030303u) * 0x01041040u) >> 24);
    }
}

// Seven jobs per row, each owning a DISJOINT set of the 32 destination code bytes:
//   slot 0..3: qs[4g..4g+4)      at stages 0..4 -> code bytes 4t+g    (weights 0..79)
//   slot 4..5: qs[16+4p..20+4p)  at stages 0..4 -> code bytes 20+2t+p (weights 80..119)
//   slot 6   : qh[0..2) -> byte 30 (trits 0,0,1,1) and byte 31 (trits 2,2,3,3)
// The three destination ranges are 0..15, 20..29 and 30..31: disjoint, and slots 4/5 stop at 29 so
// the qh slot's 30/31 are never raced by them.
__device__ __forceinline__ void ternary_ptq1_repack_slot(const std::uint8_t* src,
                                                         std::uint8_t* dst, int slot) {
    if (slot < 4) {
        ternary_ptq1_emit_stages(dst + slot, 4, load_vec<unsigned>(src + 4 * slot));
    } else if (slot < 6) {
        const int par = slot - 4;
        ternary_ptq1_emit_stages(dst + 20 + par, 2, load_vec<unsigned>(src + 16 + 4 * par));
    } else {
        const std::uint8_t h0 = src[PTQ1RowSplitStorage::kCodeBytesPerGroup];
        const std::uint8_t h1 = src[PTQ1RowSplitStorage::kCodeBytesPerGroup + 1];
        dst[30] = static_cast<std::uint8_t>(ternary_ptq1_code_of(h0, 1) |
                                            (ternary_ptq1_code_of(h1, 1) << 2) |
                                            (ternary_ptq1_code_of(h0, 3) << 4) |
                                            (ternary_ptq1_code_of(h1, 3) << 6));
        dst[31] = static_cast<std::uint8_t>(ternary_ptq1_code_of(h0, 9) |
                                            (ternary_ptq1_code_of(h1, 9) << 2) |
                                            (ternary_ptq1_code_of(h0, 27) << 4) |
                                            (ternary_ptq1_code_of(h1, 27) << 6));
    }
}

// The flattened form: 7 * Rows jobs walked lane-strided by a warp of 32 lanes, `row_src`/`row_dst`
// being the row-0 bases and the two strides the row pitches. This is the loop the wide-t and int8
// kernels both want; the small-t kernel walks the same seven slots in three divergence-free loops
// instead (see its note on why -- 7 slots over 32 lanes puts lanes with different `slot` in one
// instruction, so a single flat loop serialises its three branches) and therefore calls
// ternary_ptq1_repack_slot() directly.
__device__ __forceinline__ void ternary_ptq1_repack_warp(const std::uint8_t* row_src, int src_stride,
                                                         std::uint8_t* row_dst, int dst_stride,
                                                         int rows, int lane) {
    const int jobs = rows * 7;
#pragma unroll
    for (int job = lane; job < jobs; job += 32) {
        const int local_row = job / 7;
        const int slot      = job - local_row * 7;
        ternary_ptq1_repack_slot(row_src + static_cast<std::ptrdiff_t>(local_row) * src_stride,
                                 row_dst + static_cast<std::ptrdiff_t>(local_row) * dst_stride,
                                 slot);
    }
}

} // namespace ninfer::ops::detail
