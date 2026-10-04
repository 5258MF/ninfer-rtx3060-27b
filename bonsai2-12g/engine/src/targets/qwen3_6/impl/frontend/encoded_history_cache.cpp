#include "targets/qwen3_6/impl/frontend/encoded_history_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

bool asan_build() noexcept {
#if defined(__SANITIZE_ADDRESS__)
    return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

bool bytes_equal_prefix(std::string_view stored, std::string_view full) noexcept {
    return stored.size() <= full.size() &&
           std::memcmp(stored.data(), full.data(), stored.size()) == 0;
}

std::uint32_t checked_frontier(std::size_t value, std::string_view what) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(what) + " token frontier exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

bool ids_end_with(std::span<const int> ids, std::span<const int> tail) noexcept {
    if (tail.size() > ids.size()) { return false; }
    return std::equal(tail.begin(), tail.end(),
                      ids.end() - static_cast<std::ptrdiff_t>(tail.size()));
}

// `encode_with_boundaries` maps results positionally, so the rendered boundary list is not
// required to be sorted; stored entries are sorted by offset with duplicates dropped (equal
// offsets carry identical results within one stream).
void store_boundaries_sorted(std::vector<CommittedBoundary>& boundaries) {
    std::stable_sort(boundaries.begin(), boundaries.end(),
                     [](const CommittedBoundary& lhs, const CommittedBoundary& rhs) {
                         return lhs.offset < rhs.offset;
                     });
    std::vector<CommittedBoundary> unique;
    unique.reserve(boundaries.size());
    for (const CommittedBoundary& boundary : boundaries) {
        if (unique.empty() || unique.back().offset != boundary.offset) {
            unique.push_back(boundary);
        }
    }
    boundaries = std::move(unique);
}

const TokenBoundaryResult* committed_lookup(std::span<const CommittedBoundary> boundaries,
                                            std::size_t offset) noexcept {
    const auto it = std::lower_bound(boundaries.begin(), boundaries.end(), offset,
                                     [](const CommittedBoundary& entry, std::size_t value) {
                                         return entry.offset < value;
                                     });
    if (it != boundaries.end() && it->offset == offset) { return &it->result; }
    return nullptr;
}

// The suffix boundary list keeps mapper order, which is not guaranteed sorted: scan linearly
// (the list is bounded by checkpoint + execution + message + cache boundaries).
std::optional<std::size_t> suffix_index(std::span<const std::size_t> list,
                                        std::size_t offset) noexcept {
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (list[i] == offset) { return i; }
    }
    return std::nullopt;
}

} // namespace

bool host_encode_verify_enabled() noexcept {
    if (const char* raw = std::getenv("NINFER_VERIFY_HOST_ENCODE")) {
        return raw[0] != '\0' && std::strcmp(raw, "0") != 0;
    }
    if (asan_build()) { return true; }
#ifndef NDEBUG
    return true;
#else
    return false;
#endif
}

std::optional<CopiedCommitted>
EncodedHistoryCache::copy_longest_prefix(std::string_view full, const Tokenizer& tokenizer,
                                         std::optional<std::size_t> checkpoint_offset) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t best_index = entries_.size();
    std::size_t best_n     = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        const Entry& entry = entries_[i];
        const std::size_t n = entry.bytes.size();
        if (n == 0 || n > full.size() || n <= best_n) { continue; }
        if (std::memcmp(entry.bytes.data(), full.data(), n) != 0) { continue; }
        if (!tokenizer.is_encode_loop_pos(full, n)) { continue; }
        if (checkpoint_offset && *checkpoint_offset < n) { continue; }
        best_index = i;
        best_n     = n;
    }
    if (best_index == entries_.size()) { return std::nullopt; }
    entries_[best_index].stamp = ++clock_;
    return CopiedCommitted{.bytes                = entries_[best_index].bytes,
                           .ids                  = entries_[best_index].ids,
                           .committed_boundaries = entries_[best_index].committed_boundaries};
}

void EncodedHistoryCache::insert_committed(std::string bytes, std::vector<int> ids,
                                           std::vector<CommittedBoundary> committed_boundaries) {
    if (bytes.empty() || ids.empty() || bytes.size() > kHostEncodeCacheMaxBytes ||
        ids.size() > kHostEncodeCacheMaxIds) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& entry : entries_) {
        if (entry.bytes == bytes) {
            entry.ids                  = std::move(ids);
            entry.committed_boundaries = std::move(committed_boundaries);
            entry.stamp                = ++clock_;
            return;
        }
    }
    if (entries_.size() >= kHostEncodeCacheEntries) {
        std::size_t lru = 0;
        for (std::size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].stamp < entries_[lru].stamp) { lru = i; }
        }
        entries_[lru] = Entry{.bytes                = std::move(bytes),
                              .ids                  = std::move(ids),
                              .committed_boundaries = std::move(committed_boundaries),
                              .stamp                = ++clock_};
        return;
    }
    entries_.push_back(Entry{.bytes                = std::move(bytes),
                             .ids                  = std::move(ids),
                             .committed_boundaries = std::move(committed_boundaries),
                             .stamp                = ++clock_});
}

void EncodedHistoryCache::drop_committed(std::string_view bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase_if(entries_, [&](const Entry& entry) { return entry.bytes == bytes; });
}

void EncodedHistoryCache::poison_committed_ids(std::string_view bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& entry : entries_) {
        if (entry.bytes == bytes) {
            for (int& id : entry.ids) { id ^= 0x00ffffff; }
            return;
        }
    }
}

void EncodedHistoryCache::scramble_committed_bytes(std::string_view bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& entry : entries_) {
        if (entry.bytes == bytes && !entry.bytes.empty()) {
            entry.bytes.front() =
                static_cast<char>(static_cast<unsigned char>(entry.bytes.front()) ^ 0xff);
            return;
        }
    }
}

std::size_t EncodedHistoryCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

std::optional<EncodedChat>
try_splice_encoded_chat(const Tokenizer& tokenizer, std::span<const int> committed_ids,
                        const RenderedChat& rendered, std::size_t n,
                        std::span<const CommittedBoundary> committed_boundaries,
                        std::size_t maximum_tokens) {
    const std::string_view text = rendered.text;
    if (n > text.size() || !tokenizer.is_encode_loop_pos(text, n)) { return std::nullopt; }
    if (!rendered.literal_spans.empty() || !rendered.media_placeholders.empty() ||
        !rendered.media_token_runs.empty()) {
        return std::nullopt;
    }
    const std::size_t committed_count = committed_ids.size();
    if (committed_count >= maximum_tokens) { return std::nullopt; }
    const std::size_t suffix_limit = maximum_tokens - committed_count;

    std::optional<std::size_t> checkpoint_offset;
    if (rendered.rewrite_checkpoint) { checkpoint_offset = rendered.rewrite_checkpoint->offset; }
    if (checkpoint_offset &&
        (*checkpoint_offset < n || *checkpoint_offset > text.size())) {
        return std::nullopt;
    }

    std::vector<std::size_t> text_boundaries;
    append_rendered_text_boundaries(rendered, text_boundaries);
    std::vector<std::size_t> suffix_boundaries;
    suffix_boundaries.reserve(text_boundaries.size());
    for (const std::size_t boundary : text_boundaries) {
        if (boundary >= n) { suffix_boundaries.push_back(boundary - n); }
    }

    const BoundaryEncodedText marked = tokenizer.encode_with_boundaries(
        text.substr(n), suffix_boundaries, EncodeOptions{.max_tokens = suffix_limit});
    // The suffix encode hit its token limit, so the spliced prompt exceeds the budget; the
    // cold encode reports the overflow with the context-length diagnostic.
    if (marked.input_ids.size() == suffix_limit) { return std::nullopt; }

    EncodedChat encoded;
    encoded.input_ids.reserve(committed_count + marked.input_ids.size());
    encoded.input_ids.assign(committed_ids.begin(), committed_ids.end());
    encoded.input_ids.insert(encoded.input_ids.end(), marked.input_ids.begin(),
                             marked.input_ids.end());
    if (encoded.input_ids.empty()) { return std::nullopt; }

    if (checkpoint_offset) {
        std::optional<std::size_t> frontier;
        if (*checkpoint_offset == n) {
            frontier = committed_count;
        } else if (const auto index = suffix_index(suffix_boundaries,
                                                   *checkpoint_offset - n)) {
            const TokenBoundaryResult& boundary = marked.boundaries[*index];
            if (boundary.exact_frontier) { frontier = committed_count + *boundary.exact_frontier; }
        }
        // An empty or non-exact checkpoint prefix makes the cold encode throw; miss instead so
        // that diagnostic is preserved.
        if (!frontier || *frontier == 0) { return std::nullopt; }
        encoded.rewrite_checkpoint = RewriteCheckpointSpec{
            .kind     = rendered.rewrite_checkpoint->kind,
            .frontier = checked_frontier(*frontier, "rewrite checkpoint")};
    }

    encoded.rewrite_execution_frontiers.reserve(rendered.rewrite_execution_boundaries.size());
    for (const std::size_t boundary : rendered.rewrite_execution_boundaries) {
        std::optional<std::size_t> frontier;
        if (boundary < n) {
            if (const TokenBoundaryResult* result =
                    committed_lookup(committed_boundaries, boundary);
                result && result->exact_frontier) {
                frontier = *result->exact_frontier;
            }
        } else if (const auto index = suffix_index(suffix_boundaries, boundary - n)) {
            const TokenBoundaryResult& result = marked.boundaries[*index];
            if (result.exact_frontier) { frontier = committed_count + *result.exact_frontier; }
        }
        if (frontier && *frontier != 0 &&
            (encoded.rewrite_execution_frontiers.empty() ||
             encoded.rewrite_execution_frontiers.back() !=
                 checked_frontier(*frontier, "rewrite execution boundary"))) {
            encoded.rewrite_execution_frontiers.push_back(
                checked_frontier(*frontier, "rewrite execution boundary"));
        }
    }

    encoded.message_boundaries.resize(rendered.message_boundaries.size());
    for (std::size_t index = 0; index < rendered.message_boundaries.size(); ++index) {
        if (!rendered.message_boundaries[index]) { continue; }
        const std::size_t boundary = *rendered.message_boundaries[index];
        std::optional<std::size_t> frontier;
        if (boundary < n) {
            if (const TokenBoundaryResult* result =
                    committed_lookup(committed_boundaries, boundary);
                result && result->exact_frontier) {
                frontier = *result->exact_frontier;
            }
        } else if (const auto i = suffix_index(suffix_boundaries, boundary - n)) {
            const TokenBoundaryResult& result = marked.boundaries[*i];
            if (result.exact_frontier) { frontier = committed_count + *result.exact_frontier; }
        }
        if (frontier) {
            encoded.message_boundaries[index] = checked_frontier(*frontier, "message boundary");
        }
    }

    encoded.cache_boundaries.resize(rendered.cache_boundaries.size());
    for (std::size_t index = 0; index < rendered.cache_boundaries.size(); ++index) {
        if (!rendered.cache_boundaries[index]) { continue; }
        const std::size_t boundary = *rendered.cache_boundaries[index];
        std::size_t stable;
        if (boundary < n) {
            // A marker the insertion render did not carry has no stored result; fall back to
            // the conservative minimum instead of guessing a frontier.
            const TokenBoundaryResult* result = committed_lookup(committed_boundaries, boundary);
            stable = result ? result->stable_frontier : 0;
        } else {
            const auto i = suffix_index(suffix_boundaries, boundary - n);
            stable       = committed_count + (i ? marked.boundaries[*i].stable_frontier : 0);
        }
        encoded.cache_boundaries[index] = checked_frontier(stable, "cache boundary");
    }
    return encoded;
}

EncodedChat encode_chat_with_cache(const Tokenizer& tokenizer,
                                   const CompiledChatTemplate& chat_template,
                                   const std::vector<ChatMessage>& messages,
                                   const ChatRenderOptions& options, EncodedHistoryCache& cache,
                                   std::size_t maximum_tokens) {
    last_host_encode_observation = {};

    ChatRenderOptions committed_options              = options;
    committed_options.add_generation_prompt = false;
    const RenderedChat committed = chat_template.render(messages, committed_options);
    const RenderedChat full =
        options.add_generation_prompt ? chat_template.render(messages, options) : committed;

    // The cache only covers plain text renders: media and literal spans make the byte stream
    // non-re-encodable as a committed/suffix pair.
    if (!full.text.starts_with(committed.text) || !full.literal_spans.empty() ||
        !full.media_token_runs.empty() || !full.media_placeholders.empty() ||
        !committed.literal_spans.empty()) {
        return encode_rendered_chat(tokenizer, full, maximum_tokens);
    }

    std::optional<std::size_t> checkpoint_offset;
    if (full.rewrite_checkpoint) { checkpoint_offset = full.rewrite_checkpoint->offset; }

    const std::size_t committed_n = committed.text.size();
    std::optional<CopiedCommitted> hit =
        cache.copy_longest_prefix(full.text, tokenizer, checkpoint_offset);
    EncodedChat encoded;
    RenderedEncodeResult full_result;
    bool have_full_result = false;
    if (hit && hit->bytes.size() <= committed_n) {
        // An entry longer than the current committed prefix would splice bytes outside this
        // call's committed history; refuse it and fall back to the cold encode.
        last_host_encode_observation.attempted_prefix = true;
        last_host_encode_observation.prefix_bytes     = hit->bytes.size();
        if (auto spliced = try_splice_encoded_chat(tokenizer, hit->ids, full, hit->bytes.size(),
                                                   hit->committed_boundaries, maximum_tokens)) {
            encoded                              = std::move(*spliced);
            last_host_encode_observation.cache_hit = true;
        }
    }
    if (!last_host_encode_observation.cache_hit) {
        full_result = encode_rendered_chat_full(tokenizer, full, maximum_tokens);
        encoded     = std::move(full_result.chat);
        have_full_result = true;
    }
    if (last_host_encode_observation.cache_hit && host_encode_verify_enabled()) {
        full_result = encode_rendered_chat_full(tokenizer, full, maximum_tokens);
        if (!(full_result.chat == encoded)) {
            last_host_encode_observation.verified_mismatch = true;
            last_host_encode_observation.cache_hit         = false;
            if (hit) { cache.drop_committed(hit->bytes); }
            hit.reset();
            encoded = std::move(full_result.chat);
        }
        have_full_result = true;
    }

    // Insert this call's committed prefix so the next turn splices on it.
    if (committed_n > 0 && committed_n <= kHostEncodeCacheMaxBytes &&
        full.text.starts_with(committed.text) &&
        tokenizer.is_encode_loop_pos(full.text, committed_n)) {
        std::vector<int> committed_ids;
        std::vector<CommittedBoundary> entry_boundaries;
        bool have_entry = false;
        if (hit && hit->bytes.size() <= committed_n &&
            bytes_equal_prefix(hit->bytes, committed.text)) {
            // The hit ids cover the committed prefix; encode only the delta between the hit and
            // the current committed, and take its boundary results for the stored map.
            const std::size_t hit_n = hit->bytes.size();
            std::vector<std::size_t> delta_list;
            {
                std::vector<std::size_t> text_boundaries;
                append_rendered_text_boundaries(full, text_boundaries);
                for (const std::size_t boundary : text_boundaries) {
                    if (boundary >= hit_n && boundary < committed_n) {
                        delta_list.push_back(boundary - hit_n);
                    }
                }
            }
            const std::string_view delta = committed.text.substr(hit_n);
            const BoundaryEncodedText delta_encoded =
                tokenizer.encode_with_boundaries(delta, delta_list);
            const std::size_t committed_count = hit->ids.size();
            committed_ids.reserve(hit->ids.size() + delta_encoded.input_ids.size());
            committed_ids.insert(committed_ids.end(), hit->ids.begin(), hit->ids.end());
            committed_ids.insert(committed_ids.end(), delta_encoded.input_ids.begin(),
                                 delta_encoded.input_ids.end());
            entry_boundaries = hit->committed_boundaries;
            entry_boundaries.reserve(entry_boundaries.size() + delta_list.size());
            for (std::size_t i = 0; i < delta_list.size(); ++i) {
                const TokenBoundaryResult& result = delta_encoded.boundaries[i];
                entry_boundaries.push_back(CommittedBoundary{
                    .offset = hit_n + delta_list[i],
                    .result = TokenBoundaryResult{
                        .exact_frontier =
                            result.exact_frontier
                                ? std::optional<std::size_t>(
                                      committed_count + *result.exact_frontier)
                                : std::nullopt,
                        .stable_frontier = committed_count + result.stable_frontier}});
            }
            store_boundaries_sorted(entry_boundaries);
            have_entry = true;
        } else if (have_full_result) {
            const std::vector<int> tail = tokenizer.encode(full.text.substr(committed_n));
            if (!ids_end_with(encoded.input_ids, tail)) {
                return encoded;
            }
            committed_ids.assign(
                encoded.input_ids.begin(),
                encoded.input_ids.end() - static_cast<std::ptrdiff_t>(tail.size()));
            for (std::size_t i = 0; i < full_result.boundary_list.size(); ++i) {
                if (full_result.boundary_list[i] < committed_n) {
                    entry_boundaries.push_back(CommittedBoundary{
                        .offset = full_result.boundary_list[i],
                        .result = full_result.boundary_results[i]});
                }
            }
            store_boundaries_sorted(entry_boundaries);
            have_entry = true;
        }
        if (have_entry && !committed_ids.empty() &&
            committed_ids.size() <= kHostEncodeCacheMaxIds) {
            cache.insert_committed(committed.text, std::move(committed_ids),
                                   std::move(entry_boundaries));
            last_host_encode_observation.inserted = true;
        }
    }
    return encoded;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
