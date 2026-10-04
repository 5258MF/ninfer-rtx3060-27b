#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using qwen3_6::PreparedPromptData;
using qwen3_6::PromptModality;

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3_6::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
};

struct PrefillContext {
    ExecutionCore execution;
    qwen3_6::PagedKVCacheView text_kv;
    qwen3_6::PagedKVCacheView mtp_kv;
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t state_source_slot                          = 0;
    std::int32_t state_destination_slot                     = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    // KVMem 装配臂（prefill 期可见集有界化，标记 prefill-window/route-B）。
    // = 本 chunk 的**有效前沿** `sequence.kvmem_window.frontier(text_kv_base)`：载体未 arm 时
    // 等于 text_kv_base（恒等，OFF 臂逐位不变）；arm 过则是紧凑窗前沿 W。
    // 为什么必须从 program 侧传进来：载体 `kvmem_window` 是 per-lane 的 `SequenceState` 成员
    // （program.h:443），而 `TextContext` 只拿得到 `text_kv_base`（本结构 :50）。prefill 的
    // attention envelope 必须与 decode 期走**同一个** accessor（kvmem_window.h 的注释：
    // "The frontier every host path must use for the cache position, the envelope and the KV extent"），
    // 否则 `chunk_envelope{visible,visible}`（text_context_impl.h:1349）会用绝对位置顶穿
    // kCausalAttentionMaximumVisibleKeys ⇒ 灌到 262144 即 HTTP 500（坑表 §5.60）。
    // 默认值 0 + `has_kvmem_window_frontier=false` ⇒ 不设时行为与改前完全一致。
    std::uint32_t kvmem_window_frontier = 0;
    bool          has_kvmem_window_frontier = false;
    // KVMem P1 (no-rebake): text-prefill RoPE offset for a continuation written at window slots.
    std::int32_t  kvmem_rope_delta          = 0;
    bool          has_kvmem_rope_delta      = false;
    // KVMem P1b: cache slot = original position - shift for this chunk (0 = unshifted).
    std::uint32_t kvmem_cache_shift         = 0;
    // KVMem P1e: this chunk is a query replay (no raw-K harvest, no mean-K append, no scoring).
    bool          kvmem_replay              = false;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    qwen3_6::OrdinaryDecodeState& frame;
    const qwen3_6::OrdinaryDecodeIngress& host_ingress;
    qwen3_6::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    const qwen3_6::PagedKVCache& mtp_cache;
    qwen3_6::MtpDecodeState& frame;
    const qwen3_6::MtpDecodeIngress& host_ingress;
    qwen3_6::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3_6::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    qwen3_6::DFlashDecodeState& frame;
    const qwen3_6::DFlashDecodeIngress& host_ingress;
    qwen3_6::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct MtpCausalAttentionEnvelopes {
    ops::CausalAttentionExecutionEnvelope target_verify;
    ops::CausalAttentionExecutionEnvelope batch;
    std::array<ops::CausalAttentionExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    ops::SlidingWindowAttentionExecutionEnvelope local;
    ops::ContextAttentionExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(PrefillContext& state,
                                                    std::span<const TokenId> ids,
                                                    std::uint32_t nominal_length,
                                                    std::optional<std::uint32_t> split_frontier,
                                                    bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> split_frontier, bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
