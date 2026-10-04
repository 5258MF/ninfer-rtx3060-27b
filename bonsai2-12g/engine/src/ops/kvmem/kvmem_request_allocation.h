#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>

namespace ninfer::ops::detail {
struct KvmemRequestAllocation {
    std::uint32_t history_tokens;
    std::uint32_t sink_tokens;
    std::uint32_t retrieval_tokens;
    std::uint32_t output_tokens;
};
inline thread_local const KvmemRequestAllocation* active_kvmem_allocation = nullptr;

class ScopedKvmemAllocation {
public:
    ScopedKvmemAllocation(std::uint32_t history, std::uint32_t sink,
        std::uint32_t retrieval, std::uint32_t output) noexcept
        : current_{history, sink, retrieval, output}, previous_(active_kvmem_allocation) {
        active_kvmem_allocation = output != 0 ? &current_ : nullptr;
    }
    ~ScopedKvmemAllocation() { active_kvmem_allocation = previous_; }
    ScopedKvmemAllocation(const ScopedKvmemAllocation&) = delete;
    ScopedKvmemAllocation& operator=(const ScopedKvmemAllocation&) = delete;
private:
    KvmemRequestAllocation current_;
    const KvmemRequestAllocation* previous_;
};

[[nodiscard]] inline std::optional<std::int32_t> kvmem_request_value(const char* name) noexcept {
    if (!active_kvmem_allocation) { return std::nullopt; }
    const auto& a = *active_kvmem_allocation;
    if (std::strcmp(name, "NINFER_TERNARY_KVMEM_SCORE_BUDGET") == 0) {
        return static_cast<std::int32_t>(a.history_tokens);
    }
    if (std::strcmp(name, "NINFER_TERNARY_KVMEM_SCORE_SINK") == 0) {
        return static_cast<std::int32_t>(a.sink_tokens);
    }
    if (std::strcmp(name, "NINFER_TERNARY_KVMEM_GEN_RESERVE") == 0) {
        return static_cast<std::int32_t>(a.output_tokens);
    }
    if (std::strcmp(name, "NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW_MAX") == 0) {
        return static_cast<std::int32_t>(std::min(12288U, a.retrieval_tokens));
    }
    return std::nullopt;
}
} // namespace ninfer::ops::detail
