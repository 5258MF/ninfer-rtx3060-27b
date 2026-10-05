#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Prism-private ternary weights, group 128, carried by NInfer's row-split-k128-v1 layout:
//
//   [ base plane ][ high plane ][ scale plane ]      scale = one binary16 per group
//
// PTQ1_0  base = qs[24]  base-3 trits, five per byte     high = qh[2], four per byte
// PQ2_0   base = qs[32]  two-bit slots, four per byte    high = absent
//
// The split is not a reinterpretation: it is exactly how row_split_geometry() cuts a
// payload for these formats (group_size 128, base 24/32, high 2/0, scale 2), and the
// sizes it derives were checked byte for byte against the artifact reader
// ([248320,5120] -> PTQ1_0 278,118,400 B / PQ2_0 337,715,200 B).

struct PTQ1RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 24;
    static constexpr int kHighBytesPerGroup  = 2;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct PQ2RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 0;
    static constexpr int kScaleBytesPerGroup = 2;
};

// 3^n for n in [0,5], as the table ggml's dequantizer reads. Kept as a branch chain so no
// device-side array lookup is needed on the hot path.
__device__ __forceinline__ constexpr std::uint8_t ternary_pow3(int n) {
    return n <= 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : n == 4 ? 81u : 243u;
}

__device__ __forceinline__ float ternary_scale(const std::uint8_t* scale_ptr) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
}

// PQ2_0: block_pq2_0 { fp16 d; uint8 qs[32] } -> (code - 1) * d, code in {0,1,2}.
// Per-weight form of dequantize_row_pq2_0 (ggml-quants.c). qs lives in the base plane.
struct PQ2SimtDecodeAtom {
    static constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;

    static __device__ __forceinline__ float load_scale(const std::uint8_t* scale_ptr) {
        return ternary_scale(scale_ptr);
    }

    static __device__ __forceinline__ float decode_one(const std::uint8_t* codes,
                                                       const std::uint8_t* high, float scale,
                                                       int index) {
        (void) high; // PQ2_0 stores no high plane; row_split_geometry reports 0 bytes for it
        const int code = (static_cast<int>(codes[index >> 2]) >> (2 * (index & 3))) & 0x3;
        return static_cast<float>(code - 1) * scale;
    }
};

// PTQ1_0: block_ptq1_0 { uint8 qs[24]; uint8 qh[2]; fp16 d } with ptq1_0_stages {32,16,8}.
//
// Literal port of dequantize_row_ptq1_0. The stage walk matters and is not symmetric:
// for a 24-byte qs the c=32 stage never runs (0 + 32 > 24), c=16 runs once and emits
// weights 0..79 reading qs[0..15], c=8 runs once and emits weights 80..119 reading
// qs[16..23]; the 2-byte qh then carries weights 120..127. That is 80 + 40 + 8 = 128.
// Each trit is recovered as ((uint16)(q * 3^n) * 3) >> 8, with the uint8 wrap-around that
// the C code relies on (`uint8_t q = qs[..] * pow3[n];` truncates on assignment).
struct PTQ1SimtDecodeAtom {
    static constexpr int kGroupK = PTQ1RowSplitStorage::kGroupK;

    static __device__ __forceinline__ float load_scale(const std::uint8_t* scale_ptr) {
        return ternary_scale(scale_ptr);
    }

    // qs lives in the base plane, qh in the high plane.
    static __device__ __forceinline__ float decode_one(const std::uint8_t* codes,
                                                       const std::uint8_t* high, float scale,
                                                       int index) {
        std::uint8_t raw;
        int trit;
        if (index < 80) {
            // c = 16 stage: weights n*16 + m, byte qs[m]
            trit = index >> 4;
            raw  = codes[index & 15];
        } else if (index < 120) {
            // c = 8 stage: weights 80 + n*8 + m, byte qs[16 + m]
            const int local = index - 80;
            trit            = local >> 3;
            raw             = codes[16 + (local & 7)];
        } else {
            // qh tail: weights 120 + n*2 + h, byte qh[h]
            const int local = index - 120;
            trit            = local >> 1;
            raw             = high[local & 1];
        }
        const std::uint8_t q = static_cast<std::uint8_t>(raw * ternary_pow3(trit));
        const int xi         = static_cast<int>((static_cast<std::uint16_t>(q) * 3u) >> 8);
        return static_cast<float>(xi - 1) * scale;
    }
};

} // namespace ninfer::ops::detail
