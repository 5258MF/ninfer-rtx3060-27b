#include "runtime/contract/sampling.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

void validate(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        throw std::invalid_argument("temperature must be in [0,2]");
    }
    if (sampling.top_k < 1 || sampling.top_k > 20) {
        throw std::invalid_argument("resolved top_k must be in [1,20]");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
    }
    if ((sampling.post_thinking_temperature &&
         !std::isfinite(*sampling.post_thinking_temperature)) ||
        (sampling.post_thinking_top_p && !std::isfinite(*sampling.post_thinking_top_p)) ||
        (sampling.post_thinking_min_p && !std::isfinite(*sampling.post_thinking_min_p))) {
        throw std::invalid_argument("post-thinking sampling parameters must be finite");
    }
    if (sampling.post_thinking_temperature &&
        (*sampling.post_thinking_temperature < 0.0F ||
         *sampling.post_thinking_temperature > 2.0F)) {
        throw std::invalid_argument("post_thinking_temperature must be in [0,2]");
    }
    if (sampling.post_thinking_top_k &&
        (*sampling.post_thinking_top_k < 1 || *sampling.post_thinking_top_k > 20)) {
        throw std::invalid_argument("resolved post_thinking_top_k must be in [1,20]");
    }
    if (sampling.post_thinking_top_p &&
        (*sampling.post_thinking_top_p < 0.0F || *sampling.post_thinking_top_p > 1.0F)) {
        throw std::invalid_argument("post_thinking_top_p must be in [0,1]");
    }
    if (sampling.post_thinking_min_p &&
        (*sampling.post_thinking_min_p < 0.0F || *sampling.post_thinking_min_p > 1.0F)) {
        throw std::invalid_argument("post_thinking_min_p must be in [0,1]");
    }
}

} // namespace

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides) {
    const SamplingPreset& preset = defaults.for_mode(mode);
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .seed              = overrides.seed.value_or(0),
        .post_thinking_temperature = overrides.post_thinking_temperature
                                         ? overrides.post_thinking_temperature
                                         : preset.post_thinking_temperature,
        .post_thinking_top_k       = overrides.post_thinking_top_k
                                         ? overrides.post_thinking_top_k
                                         : preset.post_thinking_top_k,
        .post_thinking_top_p       = overrides.post_thinking_top_p
                                         ? overrides.post_thinking_top_p
                                         : preset.post_thinking_top_p,
        .post_thinking_min_p       = overrides.post_thinking_min_p
                                         ? overrides.post_thinking_min_p
                                         : preset.post_thinking_min_p,
    };
    // The registered sampling pipeline has an exact top-20 candidate domain. Preserve the
    // existing public meaning of zero (use the target cap), then expose only concrete values to
    // runtimes.
    if (resolved.top_k == 0) { resolved.top_k = 20; }
    if (resolved.post_thinking_top_k && *resolved.post_thinking_top_k == 0) {
        resolved.post_thinking_top_k = 20;
    }
    validate(resolved);
    return resolved;
}

} // namespace ninfer::runtime
