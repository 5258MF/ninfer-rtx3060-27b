#pragma once
// Incremental host encode: a small LRU cache of committed chat-history byte prefixes and
// their token ids, so a continuation prompt re-encodes only the appended suffix instead of
// the whole history. A splice is legal only at conservative encode-loop positions (added
// token match positions and run ends), where ordinary-run NFC normalization cannot change
// between the committed and suffix streams; every hit is cross-checked against a cold encode
// while verification is enabled.
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

inline constexpr std::size_t kHostEncodeCacheEntries = 16;
inline constexpr std::size_t kHostEncodeCacheMaxIds   = 262144;
inline constexpr std::size_t kHostEncodeCacheMaxBytes = 2 * 1024 * 1024;

struct HostEncodeObservation {
    bool cache_hit          = false;
    bool attempted_prefix   = false;
    bool verified_mismatch  = false;
    bool inserted           = false;
    std::size_t prefix_bytes = 0;
};

inline thread_local HostEncodeObservation last_host_encode_observation{};

[[nodiscard]] bool host_encode_verify_enabled() noexcept;

// One committed-region boundary result: the byte offset (within the stored committed bytes)
// and the token frontiers it had in the committed token stream. Exact/stable are properties
// of the committed bytes alone, so a later call with the same committed prefix can reuse
// them for the frontiers of message/execution/checkpoint boundaries that fall inside it.
struct CommittedBoundary {
    std::size_t offset = 0;
    TokenBoundaryResult result{};
};

struct CopiedCommitted {
    std::string bytes;
    std::vector<int> ids;
    std::vector<CommittedBoundary> committed_boundaries; // sorted by offset, unique offsets
};

class EncodedHistoryCache {
public:
    // Longest stored committed prefix of `full` that is a legal splice point. An entry whose
    // committed bytes extend past the current rewrite checkpoint is unusable: the checkpoint
    // frontier would lie inside the committed region and cannot be reproduced from a suffix.
    [[nodiscard]] std::optional<CopiedCommitted>
    copy_longest_prefix(std::string_view full, const Tokenizer& tokenizer,
                        std::optional<std::size_t> checkpoint_offset);

    void insert_committed(std::string bytes, std::vector<int> ids,
                          std::vector<CommittedBoundary> committed_boundaries);
    void drop_committed(std::string_view bytes);
    void poison_committed_ids(std::string_view bytes);
    void scramble_committed_bytes(std::string_view bytes);
    [[nodiscard]] std::size_t size() const;

private:
    struct Entry {
        std::string bytes;
        std::vector<int> ids;
        std::vector<CommittedBoundary> committed_boundaries;
        std::uint64_t stamp = 0;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    std::uint64_t clock_ = 0;
};

// Splice `committed_ids` in front of a fresh encode of rendered.text.substr(n). n must be a
// conservative encode-loop position of the full text; boundaries inside the committed region
// resolve from committed_boundaries, the rest from the suffix encode. Returns nullopt (the
// caller falls back to a cold encode) when any required frontier cannot be reproduced.
[[nodiscard]] std::optional<EncodedChat>
try_splice_encoded_chat(const Tokenizer& tokenizer, std::span<const int> committed_ids,
                        const RenderedChat& rendered, std::size_t n,
                        std::span<const CommittedBoundary> committed_boundaries,
                        std::size_t maximum_tokens);

// encode_rendered_chat with the incremental path: render committed (no generation prompt) and
// full, splice from the cache when possible, verify against the cold encode when enabled, and
// insert the current call's committed prefix for future calls.
EncodedChat encode_chat_with_cache(const Tokenizer& tokenizer,
                                   const CompiledChatTemplate& chat_template,
                                   const std::vector<ChatMessage>& messages,
                                   const ChatRenderOptions& options, EncodedHistoryCache& cache,
                                   std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max());

} // namespace ninfer::targets::qwen3_6::frontend_internal
