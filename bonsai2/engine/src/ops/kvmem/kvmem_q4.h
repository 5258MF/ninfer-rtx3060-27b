#pragma once

#include "ops/kvmem/kvmem_request_allocation.h"

// KVMem P4 (kv8) -- 2026-09-29.
// Shared, host-only bookkeeping for two opt-in upgrades of the boundary retrieval:
//   * NINFER_TERNARY_KVMEM_SCORE_QUERY_MODE=msg : the retrieval query is the head + tail of the LAST
//     USER MESSAGE (and, when that message is a tool result, also of the last real user message),
//     instead of the last N tokens of the final prefill chunk (QUERY_TAIL, "tail" mode).
//   * NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW=1 : the blocks this request prefilled (the new content
//     of this round: re-rendered previous answer + new user / tool message) are kept in the window
//     instead of competing with old history (capped by SCORE_PROTECT_NEW_MAX tokens).
// Both default OFF => the engine is bit-identical to kv7.
// Single lane only (the score probe and the mean-K index are process-wide singletons as well).

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {

inline std::int32_t kvmem_q4_env_i32(const char* name, std::int32_t fallback) noexcept {
    if (const auto value = kvmem_request_value(name)) { return *value; }
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') { return fallback; }
    const long parsed = std::strtol(v, nullptr, 10);
    return parsed >= 0 ? static_cast<std::int32_t>(parsed) : fallback;
}

inline bool kvmem_q4_msg_mode() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_SCORE_QUERY_MODE");
        if (v != nullptr && v[0] != '\0') { return std::strcmp(v, "msg") == 0; }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return kv != nullptr && kv[0] != '\0' && kv[0] != '0';
    }();
    return on;
}

// "split" (default): round-robin over the per-segment rankings, so every segment gets an equal share
// of the retrieved blocks.  "sum": per-segment normalised scores are averaged.
inline bool kvmem_q4_merge_sum() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_SCORE_QUERY_MERGE");
        return v != nullptr && std::strcmp(v, "sum") == 0;
    }();
    return on;
}

inline bool kvmem_q4_protect_new() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW");
        if (v != nullptr && v[0] != '\0') { return v[0] != '0'; }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return kv != nullptr && kv[0] != '\0' && kv[0] != '0';
    }();
    return on;
}

inline bool kvmem_q4_track() noexcept { return kvmem_q4_msg_mode() || kvmem_q4_protect_new(); }

// Request identity, derived from the prefill chunk stream: a chunk continues the current request
// iff it has the same prompt length and starts exactly where the previous chunk ended.
struct KvMemQ4Request {
    std::int64_t  prompt_tokens = -1;
    std::int32_t  req_start     = -1;   // absolute prompt position of the request's first chunk
    std::int32_t  last_end      = -1;   // absolute end of the last finished chunk
    std::uint64_t serial        = 0;    // bumps on every new request
};

inline KvMemQ4Request& kvmem_q4_request() {
    static KvMemQ4Request request;
    return request;
}

} // namespace ninfer::ops::detail
