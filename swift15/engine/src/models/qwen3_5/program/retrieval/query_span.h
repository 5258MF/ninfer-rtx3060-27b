#pragma once

#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/retrieval/kv9_plan.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

struct KvMemQuerySpan {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    bool exact         = false;
};

inline KvMemQuerySpan kvmem_query_span(std::uint32_t prompt_tokens, std::uint32_t reuse_base,
                                      const std::optional<TokenSpan>& query,
                                      std::span<const VisionItem> media = {},
                                      std::span<const TokenId> ids = {},
                                      std::uint32_t window_tokens = 0) {
    if (reuse_base > prompt_tokens) { throw std::logic_error("query reuse base exceeds prompt"); }
    const bool exact = query && query->begin >= reuse_base && query->begin <= prompt_tokens &&
                       query->count > 0 && query->count <= prompt_tokens - query->begin;
    auto end = exact ? static_cast<std::uint32_t>(query->begin + query->count) : prompt_tokens;
    const auto floor = exact ? static_cast<std::uint32_t>(query->begin) : reuse_base;
    auto begin = std::max(floor, end > 512U ? end - 512U : 0U);
    // kv9: with token ids the span is the replay range = the new input (from its <|im_start|>)
    // to the end, capped; the retrieval query itself comes from kv9_make_plan segments.
    if (!ids.empty() && kv9_enabled()) {
        const auto replay = kv9_replay_begin(ids, prompt_tokens, window_tokens);
        if (replay != 0) {
            begin = std::max(reuse_base, replay);
            end = prompt_tokens;
        }
    }
    for (const auto& item : media) {
        if (item.token_spans.empty()) { continue; }
        const auto first = item.token_spans.front().begin;
        const auto& last = item.token_spans.back();
        const auto consumer_begin = first == 0 ? 0 : first - 1;
        if (consumer_begin < begin && begin < last.begin + last.count) {
            begin = std::max(reuse_base, static_cast<std::uint32_t>(consumer_begin));
            break;
        }
    }
    // The replay checkpoint is captured at a chunk boundary after admission; a span that
    // starts exactly at a reuse base would never see one.
    if (reuse_base != 0 && begin == reuse_base && begin + 1U < end) { ++begin; }
    return {begin, end, exact};
}

// Replay returns to Scheduler after each chunk, including a chunk shortened by
// a rewrite frontier. These are real service units, although prompt progress is
// counted only once. Query-checkpoint splits inside the initial step add no unit.
inline std::uint64_t kvmem_replay_quanta(std::uint32_t prompt_tokens, std::uint32_t query_begin,
                                        std::uint32_t window_tokens, std::uint32_t prefill_chunk,
                                        std::span<const std::uint32_t> rewrite_frontiers) {
    if (window_tokens == 0 || prompt_tokens <= window_tokens || query_begin == 0 ||
        query_begin >= prompt_tokens) { return 0; }
    if (prefill_chunk == 0) { throw std::logic_error("query replay needs a nonzero chunk"); }
    std::uint64_t units = 0;
    std::uint32_t begin = query_begin;
    for (const auto frontier : rewrite_frontiers) {
        if (frontier <= begin) { continue; }
        if (frontier >= prompt_tokens) { break; }
        units += 1ULL + (frontier - begin - 1ULL) / prefill_chunk;
        begin = frontier;
    }
    return units + 1ULL + (prompt_tokens - begin - 1ULL) / prefill_chunk;
}

} // namespace ninfer::models::qwen3_5::detail
