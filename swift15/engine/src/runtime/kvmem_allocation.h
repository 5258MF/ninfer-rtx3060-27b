#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {

// The allocation is immutable for a request. Retrieval excludes the complete rendered
// system/developer/tool prefix; output is all remaining resident capacity.
struct KvmemAllocation {
    std::uint32_t prefix_tokens = 0;
    std::uint32_t sink_tokens = 0;
    std::uint32_t retrieval_tokens = 0;
    std::uint32_t output_tokens = 0;
    std::uint32_t retrieval_min = 0;
    std::uint32_t retrieval_target = 0;

    [[nodiscard]] bool active() const noexcept { return output_tokens != 0; }
    [[nodiscard]] std::uint32_t history_tokens() const noexcept {
        return sink_tokens + retrieval_tokens;
    }
};

[[nodiscard]] inline bool kvmem_auto_allocation_enabled() noexcept {
    const char* value = std::getenv("NINFER_KVMEM_AUTO_ALLOCATION");
    return value != nullptr && value[0] == '1';
}

[[nodiscard]] inline KvmemAllocation plan_kvmem_allocation(
    std::uint32_t resident_tokens, std::uint32_t context_tokens,
    std::uint32_t prefix_tokens, std::uint32_t block_tokens) {
    if (block_tokens == 0) { throw std::invalid_argument("KVMem block size is zero"); }
    constexpr std::uint32_t output_min = 8192;
    const auto down = [block_tokens](std::uint64_t n) {
        return static_cast<std::uint32_t>(n / block_tokens * block_tokens);
    };
    const auto cap = down(resident_tokens);
    const std::uint64_t sink_wide = std::max<std::uint64_t>(block_tokens,
        (static_cast<std::uint64_t>(prefix_tokens) + block_tokens - 1) / block_tokens * block_tokens);
    if (sink_wide + output_min + 2ULL * block_tokens > cap) {
        throw std::invalid_argument("KVMem capacity " + std::to_string(cap) +
            " cannot hold the complete system/tool prefix " + std::to_string(prefix_tokens) +
            " (aligned " + std::to_string(sink_wide) + ") plus 8192 output tokens and two retrieval blocks");
    }
    KvmemAllocation result;
    result.prefix_tokens = prefix_tokens;
    result.sink_tokens = static_cast<std::uint32_t>(sink_wide);
    result.retrieval_min = std::max(block_tokens, down(static_cast<std::uint64_t>(context_tokens) / 8));
    result.retrieval_target = std::max(result.retrieval_min,
        down(static_cast<std::uint64_t>(context_tokens) * 9 / 64));
    result.retrieval_tokens = std::min(result.retrieval_target,
        down(cap - result.sink_tokens - output_min));
    result.output_tokens = cap - result.history_tokens();
    return result;
}

} // namespace ninfer::runtime
