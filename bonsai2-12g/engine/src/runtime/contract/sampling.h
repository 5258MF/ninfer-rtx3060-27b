#pragma once

#include "ninfer/types.h"

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

[[nodiscard]] inline bool
has_post_thinking_sampling(const ResolvedSamplingParameters& sampling) noexcept {
    return sampling.post_thinking_temperature.has_value() ||
           sampling.post_thinking_top_k.has_value() ||
           sampling.post_thinking_top_p.has_value() ||
           sampling.post_thinking_min_p.has_value();
}

} // namespace ninfer::runtime
