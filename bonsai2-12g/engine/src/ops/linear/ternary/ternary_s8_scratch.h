#pragma once

// Host-safe half of the int8 rung (#14): the token threshold, the scratch descriptor and its sizes.
//
// Separate from ternary_rowsplit_mma_s8.cuh on purpose: that header carries the CUDA kernel and
// device-only syntax, and it must never be included by a host translation unit. ternary_dispatch.cpp
// and ternary_rotation.cpp are host TUs, and including the kernel header there drags
// cuda_pipeline_helpers.h into MSVC, which fails with "unexpected volatile" -- measured, that is
// exactly how the first build of this rung broke.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// A/B and rollback switch for the int8 rung, in the same style as NINFER_TERNARY_MMA / _HADAMARD.
// It is needed because this rung CHANGES THE NUMERICS (activations get quantized to int8), so it must
// be comparable inside one binary and switchable off without a rebuild:
//   NINFER_TERNARY_S8=0   -> stay on the bf16 rungs
// Read once, because the choice decides which kernel enters the captured CUDA graph.
[[nodiscard]] inline bool ternary_s8_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// First token count at which the int8 rung beats both bf16 rungs. Measured on the real shape mix
// (tscale_bench, 2026-09-20, quantization pass included): T=32 is a tie (43.25 vs 42.49 small_t),
// T=40 s8 wins by 1.20x, T=48 by 1.22x, T=64 by 1.30x; below T=32 the small-t rung's 10.2 ms per
// 8-token pass is still cheaper.
// RTX 3060 refit (2026-09-28, ninfer_bench prefill sweep T=9..32 via NINFER_TERNARY_S8_MIN_TOKENS,
// 3 interleaved rounds): small_t steps with ceil(T/8) passes (74.4 ms @16 -> 105.6 @20 -> 138.6 @28)
// while s8 is flat (83.9 @16, 85.5 @20, 89.6 @28), so the crossover is T=17 -- the value the
// package's refit found on its 66-SM card too. PPL gate c32/16: 26.9722 vs 26.9633 at 33 (+0.033%,
// inside the rung's documented +0.0015..+0.05%).
inline constexpr int kTernaryS8MinTokens = 17;

// A/B knob for the threshold above: NINFER_TERNARY_S8_MIN_TOKENS=<n> (n >= 9; below that the call
// is a decode / verify step and never reaches the int8 rung's dispatcher). It is the crossover
// against the small-T rung, which is a property of the card, not a universal constant (author's
// sm_86 upgrade package, "per-card int8 threshold refit"). The SAME function gates both the policy
// in the dispatcher and the scratch allocation below, so the two can never disagree -- a policy
// threshold lowered below the allocation threshold would silently fall back to the bf16 rungs.
// Read once: the choice decides which kernel enters the captured CUDA graph.
[[nodiscard]] inline int ternary_s8_min_tokens() {
    static const int threshold = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8_MIN_TOKENS");
        if (value == nullptr) { return kTernaryS8MinTokens; }
        const int parsed = std::atoi(value);
        return parsed >= 9 ? parsed : kTernaryS8MinTokens;
    }();
    return threshold;
}

// Activation-quantization scratch for the int8 rung: one int8 code row per token (token-major, the
// same layout as the activation it is built from) plus one fp32 scale per token.
//
// The caller owns this memory. The ternary linear op runs inside captured CUDA graphs, so a lazy
// cudaMalloc at launch time is illegal; it must come from the op's workspace arena, and the arena's
// capacity must count it (see ternary_rotation_workspace_bytes()).
struct TernaryS8Scratch {
    std::int8_t* codes  = nullptr;
    float*       scales = nullptr;
};

inline constexpr std::size_t ternary_s8_codes_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens);
}

inline constexpr std::size_t ternary_s8_scales_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * sizeof(float);
}

} // namespace ninfer::ops::detail
