#include "models/qwen3_5/program/speculative/lookup_draft.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/retrieval/block_retrieval.h"
#include "models/qwen3_5/program/retrieval/query_span.h"
#include "models/qwen3_5/program/retrieval/kv9_score.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/token_logprobs.h"
#include "core/device.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.state_destination_slots.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            // Decode and other lanes share this frame. Publish this chunk's actual
            // state slot and KV row every time, including query replay after a yield.
            ops::set_i32_scalar(lane, state.state_destination_slot, state.execution.device.stream);
            ops::set_i32_scalar(row, state.dflash_kv_table_row, state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            (void)rewrite_checkpoint;
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_kvmem_capture(execution.kvmem_q_sum, execution.kvmem_k_sum,
                           execution.kvmem_capture_slots, execution.kvmem_query_begin,
                           execution.kvmem_query_end, execution.kvmem_query_nseg,
                           execution.kvmem_query_seg, execution.kvmem_query_seg_stride,
                           execution.kvmem_q_rows, execution.kvmem_q_rows_cap, execution.kvmem_q_row0);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    card.set_fast_prefill_kernel(execution.fast_prefill_kernel);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    card.set_stage_runtime(state.execution.stages);
    card.set_rope_yarn(state.execution.rope_yarn);
    card.set_mtp_attention_window(state.execution.mtp_attention_window);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_first_token_logits(state.first_token_logits);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    card.set_layer_ready(state.layer_ready);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession* vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    card.set_stage_runtime(state.execution.stages);
    card.set_rope_yarn(state.execution.rope_yarn);
    card.set_mtp_attention_window(state.execution.mtp_attention_window);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_first_token_logits(state.first_token_logits);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    card.set_layer_ready(state.layer_ready);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision,
                                  finalize_at_end, sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            vision.bridge_column(chunk, static_cast<std::int32_t>(column - scatter.begin()));
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 ||
        hidden.ne[0] != dimension(state.execution.parameters.model.config().text.hidden_size) ||
        hidden.ne[1] != 1 || hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    project(hidden, state.execution.parameters.text.output_head, logits, state.execution.work,
            state.execution.device.stream);
    if (state.first_token_logits != nullptr) {
        CUDA_CHECK(
            cudaMemcpyAsync(state.first_token_logits, logits.data,
                            static_cast<std::size_t>(dimension(
                                state.execution.parameters.model.resources().public_token_count)) *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost, state.execution.device.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token,
                dimension(state.execution.parameters.model.resources().public_token_count),
                state.sampling, state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
}

} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

namespace {

// KV9_TIMING (diagnostic, off by default): device-synchronized wall time of each KVMem prefill
// phase, printed as "KV9T <phase> <ms>". The synchronizations only happen when it is set.
bool kv9_timing_on() {
    static const bool on = std::getenv("KV9_TIMING") != nullptr;
    return on;
}
void kv9_mark(DeviceContext& device, const char* tag, bool begin = false) {
    static std::chrono::steady_clock::time_point last;
    if (!kv9_timing_on()) { return; }
    device.synchronize();
    const auto now = std::chrono::steady_clock::now();
    if (!begin) {
        std::fprintf(stderr, "KV9T %s %.1f ms\n", tag,
                     std::chrono::duration<double, std::milli>(now - last).count());
    }
    last = now;
}

// Before an over-window conversation's retrieval index is discarded (another
// conversation took the lane), keep one copy so long reuse can restore it later.
void stash_long_kvmem_index(KvmemLaneState& sparse, std::uint32_t keep,
                            std::uint32_t min_tokens) {
    if (!kvmem_long_reuse_cfg) { return; }
    const std::uint32_t total = sparse.index.total_tokens();
    if (total <= min_tokens || keep >= total / 2) { return; }
    sparse.stash_index  = sparse.index;
    sparse.stash_tokens = sparse.index_tokens;
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token);

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

} // namespace

void ProgramImpl::start_sequence(std::uint32_t lane, SequenceState& sequence,
                                 MaterializationTransaction& transaction) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    auto& sparse = kvmem_lanes_.at(lane);
    RequestControl& request = requests[lane];
    if (!transaction.plan || transaction.plan->impl_ == nullptr || !transaction.prepared ||
        !request.prefill) {
        throw std::invalid_argument("materialization staging is incomplete");
    }
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }
    auto& staged                           = *request.prefill;
    const auto started                     = Clock::now();
    const std::uint32_t prompt_tokens      = staged.prompt_tokens;
    const std::uint32_t base               = staged.base;
    const std::uint32_t initial_mtp_extent = staged.initial_mtp_extent;
    request.lifecycle                      = Lifecycle::Empty;
    // The sequence's own token ceiling: the last frontier its Device KV lease may cover, so
    // on-demand growth never leases pages the request cannot reach.
    request.lease_ceiling =
        std::min(capacity, request_plan.summary.prompt_tokens +
                               (request_plan.summary.effective_output_tokens == 0
                                    ? 0U
                                    : request_plan.summary.effective_output_tokens - 1U));
    try {
        const std::uint32_t state_slots = request_plan.demand.active_entitlement.device.state_slots;
        const bool preserving_source =
            (transaction.has_source || transaction.has_shared_source) &&
            transaction.source_mode == runtime::PrivateSourceMode::Retain;
        const bool text_prefix_fork    = request_plan.text_prefix_fork_required;
        const bool backend_prefix_fork = request_plan.backend_prefix_fork_required;
        if (request_plan.reuse == ReusePath::Root) {
            if (transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_activation ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0) ||
                transaction.backend_activation.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("root materialization reservations are incomplete");
            }
            release_sequence_kv(sequence);
            release_sequence_state(sequence);
            sequence.state = ActiveStateBinding{.read  = transaction.reserved_states[0],
                                                .write = transaction.reserved_states[0]};
            transaction.reserved_states[0] = {};
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else if (preserving_source) {
            const bool private_source_ready = transaction.has_source &&
                                              transaction.source_index < continuation_capacity &&
                                              continuation_slots[transaction.source_index].role ==
                                                  ContinuationSlotRole::Catalogued;
            const bool shared_source_ready =
                transaction.has_shared_source &&
                transaction.shared_source_index < shared_prefix_capacity &&
                shared_prefix_slots[transaction.shared_source_index].role ==
                    SharedPrefixSlotRole::Catalogued;
            if (private_source_ready == shared_source_ready ||
                transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_prefix_fork ||
                !transaction.prefix_forks_ready ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("retained materialization is incomplete");
            }
            const StateImageHandle selected =
                private_source_ready
                    ? selected_state(continuation_states[transaction.source_index],
                                     request_plan.reuse, request_plan.selected_checkpoint)
                    : shared_prefix_states[transaction.shared_source_index].state;
            const StateImageHandle current = transaction.reserved_states[0];
            if (state_store->residency(selected) == StateReplicaResidency::HostOnly) {
                if (state_store->role(current) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("Host retained Fork destination was not published");
                }
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else if (transaction.split_state_identity) {
                if (!private_source_ready ||
                    state_store->residency(selected) != StateReplicaResidency::Both) {
                    throw std::logic_error("StateImage identity split source changed");
                }
                state_store->split_device_replica_identity(selected, current);
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else {
                const StateImageSelectors selectors = state_store->begin_fork(selected, current);
                if (is_masked_draft_backend(speculative_backend)) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = current,
                    .fork_pending   = true,
                    .read_ownership = StateReadOwnership::ExternalOwner,
                };
            }
            transaction.reserved_states[0]   = {};
            transaction.split_state_identity = false;
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;
            sequence.rewrite_state.reset();
            sequence.rewrite_checkpoint = {};

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else {
            if (request_plan.state_fork_required !=
                transaction.state_fork_destination.has_value()) {
                throw std::logic_error("private materialization StateImage Fork is incomplete");
            }
            if (transaction.reserved_state_count > 1 ||
                (transaction.reserved_state_count != 0 && sequence.reserved_state)) {
                throw std::logic_error("private materialization StateImage reservation is invalid");
            }
            if (transaction.reserved_state_count == 1) {
                sequence.reserved_state          = transaction.reserved_states[0];
                transaction.reserved_states[0]   = {};
                transaction.reserved_state_count = 0;
            }
        }

        if (!preserving_source) {
            std::array<HostKVPageReplicaRelease, 2> stale_tail_replicas{};
            std::size_t stale_tail_count           = 0;
            const auto preflight_inactive_truncate = [&](KVAddressSpaceStore& addresses,
                                                         LogicalKVPageStore& pages,
                                                         KVAddressSpaceHandle address,
                                                         std::optional<std::uint32_t> frontier) {
                if (!frontier ||
                    (addresses.committed_frontier(address) == *frontier &&
                     addresses.mapped_pages(address) == kv_pages_for_frontier(*frontier))) {
                    return;
                }
                bool releases_tail               = false;
                const std::uint32_t target_pages = kv_pages_for_frontier(*frontier);
                if (target_pages != 0) {
                    const LogicalKVPageHandle tail =
                        addresses.logical_page(address, target_pages - 1U);
                    const std::uint32_t columns =
                        *frontier -
                        (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
                    if (columns != pages.committed_columns(tail) && pages.host_resident(tail)) {
                        if (host_kv_extents == nullptr ||
                            stale_tail_count == stale_tail_replicas.size()) {
                            throw std::logic_error("stale Host KV tail replica is not releasable");
                        }
                        stale_tail_replicas[stale_tail_count++] =
                            HostKVPageReplicaRelease{.pages = &pages, .page = tail};
                        releases_tail = true;
                    }
                }
                if (!addresses.can_destructive_truncate_inactive(address, *frontier,
                                                                 releases_tail)) {
                    throw std::logic_error(
                        "selected private KV frontier is not destructively materializable");
                }
            };
            if (!sequence.kv) {
                throw std::logic_error("materialization destination has no KV address space");
            }
            if (!text_prefix_fork) {
                preflight_inactive_truncate(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                                            transaction.text_activation_frontier);
            }
            if (sequence.kv->backend && !backend_prefix_fork) {
                preflight_inactive_truncate(*backend_kv_addresses, *backend_kv_pages,
                                            *sequence.kv->backend,
                                            transaction.backend_activation_frontier);
            }
            if (stale_tail_count != 0) {
                const std::span<const HostKVPageReplicaRelease> releases(stale_tail_replicas.data(),
                                                                         stale_tail_count);
                if (!host_kv_extents->release_page_replicas(releases)) {
                    throw std::logic_error(
                        "stale Host KV tail replicas cannot be released atomically");
                }
            }
            if (!text_prefix_fork && transaction.text_activation_frontier &&
                (text_kv_addresses->committed_frontier(sequence.kv->text) !=
                     *transaction.text_activation_frontier ||
                 text_kv_addresses->mapped_pages(sequence.kv->text) !=
                     kv_pages_for_frontier(*transaction.text_activation_frontier))) {
                text_kv_addresses->destructive_truncate_inactive(
                    sequence.kv->text, *transaction.text_activation_frontier);
            }
            if (!backend_prefix_fork && transaction.backend_activation_frontier &&
                sequence.kv->backend &&
                (backend_kv_addresses->committed_frontier(*sequence.kv->backend) !=
                     *transaction.backend_activation_frontier ||
                 backend_kv_addresses->mapped_pages(*sequence.kv->backend) !=
                     kv_pages_for_frontier(*transaction.backend_activation_frontier))) {
                backend_kv_addresses->destructive_truncate_inactive(
                    *sequence.kv->backend, *transaction.backend_activation_frontier);
            }
            if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        }
        if ((text_prefix_fork || backend_prefix_fork) && !transaction.prefix_forks_ready) {
            throw std::logic_error("materialization prefix forks are incomplete");
        }
        if (text_prefix_fork) {
            text_kv_addresses->commit_prefix_fork(std::move(*transaction.text_prefix_fork),
                                                  compute_streams);
            transaction.text_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = sequence.kv->text;
                sequence.kv->text                         = *transaction.root_text_address;
                transaction.root_text_address.reset();
                if (!text_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Text KV source remained pinned after COW");
                }
            }
        } else {
            text_kv_addresses->commit_activation(std::move(*transaction.text_activation),
                                                 compute_streams);
            transaction.text_activation.reset();
        }
        if (backend_prefix_fork) {
            backend_kv_addresses->commit_prefix_fork(std::move(*transaction.backend_prefix_fork),
                                                     compute_streams);
            transaction.backend_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = *sequence.kv->backend;
                sequence.kv->backend                      = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
                if (!backend_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Backend KV source remained pinned after COW");
                }
            }
        } else if (transaction.backend_activation) {
            backend_kv_addresses->commit_activation(std::move(*transaction.backend_activation),
                                                    compute_streams);
            transaction.backend_activation.reset();
        }
        transaction.prefix_forks_ready = false;
        transaction.text_activation_frontier.reset();
        transaction.backend_activation_frontier.reset();
        transaction.prepared = false;

        const bool preserve_rewrite =
            request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting;
        const auto activate_consumed_state = [&](StateImageHandle selected) {
            if (!request_plan.state_fork_required) {
                if (transaction.state_fork_destination ||
                    state_store->checkpoint_references(selected) != 0) {
                    throw std::logic_error("planned StateImage Move is no longer valid");
                }
                state_store->move_checkpoint_to_active(selected);
                sequence.state = ActiveStateBinding{.read = selected, .write = selected};
                return;
            }
            if (!transaction.state_fork_destination ||
                state_store->checkpoint_references(selected) == 0) {
                throw std::logic_error("planned StateImage Fork is no longer valid");
            }
            const StateImageHandle destination = *transaction.state_fork_destination;
            if (transaction.state_restored) {
                if (state_store->role(destination) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("restored StateImage Fork destination is unavailable");
                }
                sequence.state = ActiveStateBinding{.read = destination, .write = destination};
            } else {
                const std::uint32_t references = state_store->checkpoint_references(selected);
                const std::uint32_t lineage_references =
                    owned_checkpoint_references(sequence, selected);
                if (lineage_references > references) {
                    throw std::logic_error("consumed StateImage Fork ownership is inconsistent");
                }
                const StateReadOwnership read_ownership =
                    lineage_references == references ? StateReadOwnership::LineageCheckpoint
                                                     : StateReadOwnership::ExternalOwner;
                const StateImageSelectors selectors =
                    state_store->begin_fork(selected, destination);
                if (is_masked_draft_backend(speculative_backend)) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = destination,
                    .fork_pending   = true,
                    .read_ownership = read_ownership,
                };
            }
            transaction.state_fork_destination.reset();
        };
        if (request_plan.reuse == ReusePath::Root) {
            sequence.rewrite_checkpoint = {};
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_digests.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
            if (kvmem_window_pages != 0) {
                // The retrieval index tracks the newest conversation; a Root start is a new
                // conversation, so stale block means from a finished one must not pollute
                // scoring (page numbering restarts at zero, so stale hits would be wrong).
                stash_long_kvmem_index(sparse, 0, kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize);
                sparse.index.truncate_to(0);
                sparse.index_tokens.clear();
                sparse.query.clear();
                sparse.query_count.clear();
            }
        } else if (preserving_source) {
            const SequenceState* private_source =
                transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
            SharedPrefixState* shared_source =
                transaction.has_shared_source
                    ? &shared_prefix_states[transaction.shared_source_index]
                    : nullptr;
            const std::uint32_t source_text_frontier =
                private_source != nullptr ? private_source->text_kv_valid : shared_source->frontier;
            if (!sequence.kv || source_text_frontier < base) {
                throw std::logic_error("retained prefix has incomplete Text KV");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base       = base == 0 ? 0 : base - 1U;
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->mtp_kv_valid
                                                         : shared_source->backend_frontier;
                if (!request_plan.prepare_mtp || source_backend < mtp_base) {
                    throw std::logic_error("retained prefix has incomplete MTP KV");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend)) {
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->dflash_context_frontier
                                                         : shared_source->frontier;
                if (source_backend < base) {
                    throw std::logic_error("retained prefix has incomplete DFlash KV");
                }
                sequence.dflash_context_frontier = base;
            }
            sequence.tail_hidden_valid =
                base == prompt_tokens &&
                (private_source != nullptr ? private_source->tail_hidden_valid
                                           : shared_source->tail_hidden_valid);
            if (shared_source != nullptr) {
                if (shared_source->active_references == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("shared-prefix active reference overflow");
                }
                ++shared_source->active_references;
                sequence.shared_prefix_references.push_back(transaction.shared_source_index);
            }
            refresh_state_views(sequence);
            bind_sequence_kv(sequence);
        } else if (request_plan.reuse == ReusePath::PrivateEndpoint) {
            if (!state_store->valid(sequence.state.read) ||
                sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable) {
                throw std::logic_error("resident endpoint StateImage is not movable");
            }
            if (!preserve_rewrite && sequence.rewrite_state) {
                const StateImageHandle dropped = *sequence.rewrite_state;
                state_store->release_checkpoint_reference(dropped);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
                if (dropped != sequence.state.read &&
                    state_store->checkpoint_references(dropped) == 0 &&
                    !state_store->release(dropped)) {
                    throw std::logic_error("dropped rewrite StateImage could not be released");
                }
            }
            activate_consumed_state(sequence.state.read);
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend) &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            const std::uint32_t text_entitlement =
                kvmem_window_pages != 0
                    ? std::max(request_plan.text_kv_page_entitlement,
                               text_kv_addresses->mapped_pages(sequence.kv->text) +
                                   (request_plan.text_kv_page_entitlement >
                                            text_kv_addresses->device_residency_floor_pages(
                                                sequence.kv->text)
                                        ? request_plan.text_kv_page_entitlement -
                                              text_kv_addresses->device_residency_floor_pages(
                                                  sequence.kv->text)
                                        : 0U))
                    : request_plan.text_kv_page_entitlement;
            resize_sequence_kv_entitlement(sequence, text_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            if (!sequence.rewrite_state || !state_store->valid(*sequence.rewrite_state) ||
                state_store->role(*sequence.rewrite_state) != StateImageRole::CheckpointImmutable ||
                (sequence.endpoint_valid &&
                 (!state_store->valid(sequence.state.read) ||
                  sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                  state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable))) {
                throw std::logic_error("resident rewrite StateImage is not movable");
            }
            const StateImageHandle checkpoint = *sequence.rewrite_state;
            if (sequence.endpoint_valid && sequence.state.read == checkpoint) {
                throw std::logic_error("resident endpoint aliases its rewrite StateImage");
            }
            if (sequence.endpoint_valid && !state_store->release(sequence.state.read)) {
                throw std::logic_error("superseded endpoint StateImage could not be released");
            }
            if (!preserve_rewrite) {
                state_store->release_checkpoint_reference(checkpoint);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
            }
            activate_consumed_state(checkpoint);
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend)) {
                if (!dflash || (backend_kv_cache() && !sequence.kv->backend) ||
                    sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                sequence.dflash_context_frontier = base;
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            const std::uint32_t text_entitlement =
                kvmem_window_pages != 0
                    ? std::max(request_plan.text_kv_page_entitlement,
                               text_kv_addresses->mapped_pages(sequence.kv->text) +
                                   (request_plan.text_kv_page_entitlement >
                                            text_kv_addresses->device_residency_floor_pages(
                                                sequence.kv->text)
                                        ? request_plan.text_kv_page_entitlement -
                                              text_kv_addresses->device_residency_floor_pages(
                                                  sequence.kv->text)
                                        : 0U))
                    : request_plan.text_kv_page_entitlement;
            resize_sequence_kv_entitlement(sequence, text_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.tail_hidden_valid = base == prompt_tokens;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        sequence.endpoint_valid = false;
        if (!preserving_source) { trim_sequence_kv(sequence, base, backend_kv_valid(sequence)); }
        bind_sequence_kv(sequence);
        std::uint32_t backend_materialized       =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prompt_tokens
                                                                : 0U;
        std::uint32_t main_materialized = prompt_tokens;
        if (kvmem_window_pages != 0) {
            // Features belong to the request being admitted, including cache-hit
            // branches. Never reuse another conversation's block IDs or aborted sums.
            // Long reuse keeps the whole blocks this lane already scored for the same token
            // prefix (the consumed continuation was built on this lane); anything else
            // restarts from zero. Blocks between the kept prefix and base stay unscored.
            std::uint32_t keep = 0;
            if (kvmem_long_reuse_cfg && base != 0) {
                const auto& ids = staged.prompt.token_ids;
                const auto matching = [&](const RetrievalIndex& index,
                                          const std::vector<TokenId>& tokens) {
                    const std::uint32_t limit = std::min<std::uint32_t>(
                        {base, index.total_tokens(), static_cast<std::uint32_t>(tokens.size()),
                         static_cast<std::uint32_t>(ids.size())});
                    std::uint32_t same = 0;
                    while (same < limit && tokens[same] == ids[same]) { ++same; }
                    return same;
                };
                std::uint32_t same = matching(sparse.index, sparse.index_tokens);
                if (sparse.stash_index.has_value()) {
                    const std::uint32_t stashed = matching(*sparse.stash_index, sparse.stash_tokens);
                    if (stashed > same) {
                        std::swap(sparse.index, *sparse.stash_index);
                        std::swap(sparse.index_tokens, sparse.stash_tokens);
                        same = stashed;
                        std::fprintf(stderr, "KVMem reuse | restored stashed index %u tokens\n",
                                     stashed);
                    }
                }
                keep = same / ops::kKvmemCaptureBlockTokens * ops::kKvmemCaptureBlockTokens;
            }
            stash_long_kvmem_index(sparse, keep, kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize);
            sparse.index.truncate_to(keep);
            (void)sparse.index.append(base - keep);
            sparse.index_tokens.assign(staged.prompt.token_ids.begin(),
                                       staged.prompt.token_ids.begin() + prompt_tokens);
            if (std::getenv("NINFER_KVMEM_TRACE") != nullptr || (kvmem_long_reuse_cfg && base != 0)) {
                std::fprintf(stderr, "KVMem reuse | base %u | kept index %u tokens | prompt %u\n",
                             base, keep, prompt_tokens);
            }
            sparse.query.clear();
            sparse.query_count.clear();
            sparse.retrieved_pages.clear();
            sparse.media_groups = media_page_groups(staged.prompt.vision_items);
            sparse.capture_begin = base;
            sparse.query_checkpoint_valid = false;
            // Long sparse requests currently restart from root, so their User span is
            // available to capture even when followed by an arbitrarily long tool tail.
            const std::uint32_t prompt_window_tokens =
                kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize;
            const auto query = kvmem_query_span(prompt_tokens, base, staged.prompt.retrieval_query,
                                               staged.prompt.vision_items, staged.prompt.token_ids,
                                               prompt_window_tokens);
            sparse.query_end = query.end;
            sparse.query_begin = query.begin;
            plan_kv9_query(sparse, staged.prompt, prompt_tokens, base);
            if (staged.vision && prompt_tokens > kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize && query.begin != 0) {
                staged.vision->retain_for_replay(query.begin);
            }
            if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
                std::fprintf(stderr, "KVMEM query source=%s begin=%u end=%u prompt=%u\n",
                             query.exact ? "last_user" : "suffix", sparse.query_begin,
                             sparse.query_end, prompt_tokens);
            }
            CUDA_CHECK(cudaMemsetAsync(sparse.query_sum.data, 0, sparse.query_sum.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(sparse.key_sums.data, 0, sparse.key_sums.bytes(), device.stream));
            // Sparse prefill maps only the first chunk of growth;
            // chunk boundaries demote the committed overflow to Host replicas afterwards.
            main_materialized = std::min(
                prompt_tokens, base + prefill_chunk);
            if (speculative_backend == SpeculativeBackend::Mtp &&
                backend_materialized > main_materialized) {
                const std::uint32_t lead = backend_materialized > prompt_tokens
                                               ? backend_materialized - prompt_tokens
                                               : 0U;
                backend_materialized = main_materialized + lead;
            }
        }
        ensure_sequence_kv_mapped(sequence, main_materialized, backend_materialized);
        request.grammar                  = request_plan.grammar;
        request.first_token_top_logprobs = request_plan.first_token_top_logprobs;
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = staged.prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        request.timings              = {};
        request.pending              = {};
        request.publish_continuation = request_plan.summary.publish_continuation;
        sequence.mtp_draft_count     = 0;
        sequence.tail_hidden_valid   = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.swap(materialization_ledger_);
        sequence.prefix_identity.swap(materialization_identity_);
        sequence.prefix_digests.swap(materialization_prefix_digests_);
        sequence.rebuild_work       = request_plan.root_rebuild_work;
        sequence.rebuild_tail_begin = request_plan.root_rebuild_tail_begin;

        if (staged.disk_restore_frontier != 0) {
            bool restored = false;
            try {
                restored =
                    restore_prefix_from_disk(sequence, staged, staged.disk_restore_frontier);
            } catch (const std::exception&) {
                // The tier is best effort: whatever failed, the prompt is recomputed.
            }
            if (!restored) { staged.hidden_replay_tokens = staged.disk_restore_frontier; }
        }

        if (is_masked_draft_backend(speculative_backend)) {
            if (!dflash || !io.dflash_decode || (backend_kv_cache() && !sequence.kv->backend)) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            upload_dflash_prefill_controls(sequence);
        }

        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle = Lifecycle::Prefilling;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane_best_effort(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult
ProgramImpl::advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(active_sequence(lane), requests[lane], failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_prefill_raw(std::uint32_t lane, bool terminal,
                                                          runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("prefill resolution requires a pending prefill token");
    }
    return resolve_non_speculative_pending(active_sequence(lane), requests[lane], 1, terminal,
                                           std::nullopt, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_pending_raw(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size() ||
        prefix_execution_splits.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (lanes.size() == 1 && lanes.front() < max_concurrency &&
        requests[lanes.front()].pending.kind == PendingKind::Begin) {
        const std::uint32_t lane = lanes.front();
        if (requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("prefill pending token no longer matches Program state");
        }
        if (cancelled.front()) {
            if (accepted_tokens.front() != 0 || !terminal.front()) {
                throw std::logic_error("cancelled prefill pending decision is invalid");
            }
            if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                throw std::logic_error("cancelled prefill lane is not strictly releasable");
            }
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(
                active_sequence(lane), requests[lane], accepted_tokens.front(),
                terminal.front() != 0, prefix_execution_splits.front(), failed_timing));
            timing.resume_post();
        }
        return timing.finish();
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                    throw std::logic_error("cancelled decode lane is not strictly releasable");
                }
            } else {
                timing.pause();
                timing.include(resolve_non_speculative_pending(
                    active_sequence(lane), requests[lane], accepted_tokens[row], terminal[row] != 0,
                    prefix_execution_splits[row], failed_timing));
                timing.resume_post();
            }
        }
        return timing.finish();
    }

    if (!replay_fold) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    std::uint32_t record_width   = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = active_sequence(lane);
        if (row == 0) { record_width = pending.record_width; }
        if (pending.record_width == 0 || pending.record_width != record_width) {
            throw std::logic_error("speculative pending rows disagree on their record width");
        }
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.prefix_digests.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (is_masked_draft_backend(speculative_backend) &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        const StateImageSelectors selectors = state_selectors(sequence);
        fold_rows[row] =
            ops::GdnReplayFoldRow{.source_state_slot      = selectors.source,
                                  .destination_state_slot = selectors.destination,
                                  .commit_columns         = static_cast<std::int32_t>(committed)};
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    // One speculative round produced every pending row, and its frame, egress and records are
    // laid out at the width it verified.
    const std::uint32_t verify_drafts = record_width - 1U;
    const auto tail_started           = Clock::now();
    try {
        timing.resume_submit();
        const std::span<const ops::GdnReplayFoldRow> fold_span(fold_rows.data(), lanes.size());
        {
            // Each state shard folds its own layers on its own device's stream.
            RankBinding bind(device, state_images->shard(0).rank);
            replay_fold->execute(fold_span, static_cast<std::int32_t>(record_width),
                                 compute_streams[state_images->shard(0).rank]);
        }
        for (std::size_t shard = 1; shard < state_images->shard_count(); ++shard) {
            RankBinding bind(device, state_images->shard(shard).rank);
            extra_replay_fold[shard - 1]->execute(fold_span,
                                                  static_cast<std::int32_t>(record_width),
                                                  compute_streams[state_images->shard(shard).rank]);
        }

        // Sparse acceptance reads counts. Publish only the prefix licensed by the Frontend.
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (cancelled[row] || !requests[lanes[row]].sampling_host.token_counts) {
                    continue;
                }
                const auto count = static_cast<std::int32_t>(accepted_tokens[row]);
                Tensor ids       = io.dflash_decode->narrowed(verify_drafts)
                                       .licensed_tokens.slice(1, static_cast<std::int32_t>(row), 1)
                                       .slice(0, 0, count)
                                       .view({count});
                Tensor counts =
                    token_counts.slice(1, static_cast<std::int32_t>(lanes[row]), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
                ops::increment_token_counts(ids, counts, device.stream);
            }
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                // The round's target hidden is laid out at the width it verified.
                const qwen3_5::MtpDecodeState frame =
                    io.mtp_decode->verification_view(verify_drafts);
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else if (is_masked_draft_backend(speculative_backend) && io.dflash_decode) {
                const qwen3_5::DFlashDecodeState frame = io.dflash_decode->narrowed(verify_drafts);
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, state_images->continuation_hidden_store(),
                         device.stream);
        }

        if (is_masked_draft_backend(speculative_backend)) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        work.reset();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        clear_execution_failure_lanes(lanes);
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = verify_drafts + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = active_sequence(lanes[row]);
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                if (!clear_lane_strict(sequence, request)) {
                    throw std::logic_error("cancelled speculative lane is not strictly releasable");
                }
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            settle_state_fork(sequence);
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            commit_generated_prefix_identity(sequence, pending.base_S,
                                             std::span<const TokenId>(token_base, committed),
                                             prefix_execution_splits[row]);
            advance_rebuild_work(sequence, pending.base_E + committed, prefill_chunk);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                    // Context lookup, preferred over the draft head's guess whenever the n-gram has
                    // been seen before. The two are complements: the head is weakest on output that
                    // repeats the input, which is exactly where a lookup is certain. The proposal is
                    // one-hot either way, so verify treats them identically and a wrong guess costs
                    // throughput rather than correctness.
                    if (lookup_ngram != 0) {
                        const auto found = ::ninfer::qwen3_5::lookup_draft(std::span<const TokenId>(sequence.ledger),
                                                                 lookup_ngram, draft_window,
                                                                 sequence.mtp_drafts.data());
                        if (found != 0) { sequence.mtp_draft_count = found; }
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                request.lifecycle = Lifecycle::Finishable;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        clear_execution_failure_lanes(lanes);
        throw;
    }
    return timing.finish();
}

runtime::PrefillStepResult ProgramImpl::advance_prefill(SequenceState& sequence,
                                                        RequestControl& request,
                                                        runtime::ExecutionTiming* failed_timing) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    if (kvmem_window_pages != 0) { kv9_mark(device, "begin", true); }
    if (staged.pending_capture_offer != 0) {
        throw std::logic_error("prefill cannot advance while a capture offer is pending");
    }
    const runtime::BeginSummary summary{
        .prompt_tokens        = staged.prompt_tokens,
        .reused_prompt_tokens = staged.disk_restore_frontier != 0 ? staged.disk_restore_frontier
                                                                  : staged.base,
        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    // After a failed disk restore the admitted prefix is recomputed without being reported.
    const auto reported_tokens = [&staged, &processed_prompt_tokens] {
        const std::uint32_t hidden = std::min(staged.hidden_replay_tokens, processed_prompt_tokens);
        staged.hidden_replay_tokens -= hidden;
        return processed_prompt_tokens - hidden;
    };
    const auto started                    = Clock::now();
    try {
        if (staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == staged.cursor) {
            if (staged.cursor != staged.base ||
                !staged.capture_groups[staged.next_capture].shared ||
                staged.capture_groups[staged.next_capture].rewrite ||
                staged.capture_groups[staged.next_capture].long_anchor) {
                throw std::logic_error("zero-prefill capture is not a shared base promotion");
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            staged.pending_capture_offer = next_capture_offer_id_;
            return runtime::PrefillStepResult{
                .summary = summary,
                .timing  = timing.finish(),
            };
        }
        // Prefill attention addresses its KV through the shared step table-row scalars. Another
        // lane's staging or capture can rebind them between this lane's steps, so every step
        // binds its own rows before any Prefill or MTP-bridge work.
        bind_sequence_kv(sequence);
        StateImageSelectors selectors = state_selectors(sequence);
        // Hybrid prefix cache taps (docs/maintainer/hybrid-prefix-cache-spec.md §7.1): an exact tap
        // splits the chunk at its frontier; any chunk boundary may realize a flexible tap, so the
        // continuation hidden is kept while the lane has taps left.
        const HybridLaneState* hybrid_lane =
            hybrid_ && sequence.lane < max_concurrency ? &hybrid_lanes_[sequence.lane] : nullptr;
        const auto hybrid_taps_left = [&]() {
            return hybrid_lane != nullptr && hybrid_lane->active && hybrid_lane->publish &&
                   hybrid_lane->next_tap < hybrid_lane->taps.size();
        };
        const auto next_hybrid_split = [&]() -> std::optional<std::uint32_t> {
            if (!hybrid_taps_left()) { return std::nullopt; }
            for (std::size_t tap = hybrid_lane->next_tap; tap < hybrid_lane->taps.size(); ++tap) {
                const runtime::prefix_cache::PlannedTap& planned = hybrid_lane->taps[tap];
                if (planned.placement == runtime::prefix_cache::TapPlacement::Exact &&
                    planned.position > staged.cursor) {
                    return planned.position;
                }
            }
            return std::nullopt;
        };
        Tensor rewrite_capture_hidden;
        Tensor* rewrite_capture_hidden_ptr = nullptr;
        if (staged.next_capture < staged.capture_groups.size() || hybrid_taps_left()) {
            rewrite_capture_hidden = state_images->continuation_hidden_slot(selectors.destination);
            rewrite_capture_hidden_ptr = &rewrite_capture_hidden;
        }
        // A capture (rewrite checkpoint) published during the first pass must stay immutable,
        // so long-reuse prompts that capture skip the replay; decode still reads the
        // retrieved working set.
        const bool needs_query_replay =
            kvmem_window_pages != 0 &&
            staged.prompt_tokens > kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize && sparse.query_begin != 0 &&
            staged.capture_groups.empty();
        execution::PrefillContext schedule_state{
            {device, parameters, work, state_images->linear(0),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
             mtp_attention_window,
             kvmem_window_pages != 0 ? static_cast<float*>(sparse.query_sum.data) : nullptr,
             kvmem_window_pages != 0 ? static_cast<float*>(sparse.key_sums.data) : nullptr,
             kvmem_capture_slots_, sparse.query_begin, sparse.query_end},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            // The probe's throwaway bridge token uses argmax. Sampling here would
            // publish a phantom token into penalty history before the real replay.
            needs_query_replay ? nullptr : static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            rewrite_capture_hidden_ptr,
            selectors.source,
            selectors.destination,
            staged.initial_mtp_extent,
            dflash_host_ingress};
        if (kvmem_window_pages != 0 && sparse.seg_n != 0) {
            // kv9: segments replace the legacy single query span.
            auto& ex = schedule_state.execution;
            ex.kvmem_query_end = ex.kvmem_query_begin;
            ex.kvmem_query_nseg = sparse.seg_accum;
            for (std::uint32_t i = 0; i < 2U * sparse.seg_accum; ++i) { ex.kvmem_query_seg[i] = sparse.acc_range[i]; }
            ex.kvmem_query_seg_stride = static_cast<std::uint32_t>(sparse.query_sum.bytes() / sizeof(float) / 4U);
            ex.kvmem_q_rows = sparse.q_rows.get();
            ex.kvmem_q_rows_cap = sparse.q_rows_cap;
            for (std::uint32_t i = 0; i < 4U; ++i) { ex.kvmem_q_row0[i] = sparse.slot_row0[i]; }
        }
        const auto public_tokens =
            static_cast<std::size_t>(dimension(parameters.model.resources().public_token_count));
        if (request.first_token_top_logprobs != 0) {
            if (!first_token_logits_host) {
                first_token_logits_host.emplace(public_tokens * sizeof(std::uint16_t));
            }
            schedule_state.first_token_logits = first_token_logits_host->data();
        }
        // The first pass after a Host restore waits for each layer's copies (hybrid spec §6.5).
        schedule_state.layer_ready = hybrid_take_restore_layers(sequence.lane);
        schedule_state.dflash_kv_table_row =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = sequence.tail_hidden;
            const execution::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                execution::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                 bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                execution::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                  bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
            staged.mtp_bridge = MtpBridgeMode::None;
        }

        if (staged.query_replay_cursor) {
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            const auto final_chunk_tokens = advance_kvmem_query_replay(sequence, staged, timing);
            kv9_mark(device, "replay-chunk");
            if (*staged.query_replay_cursor < staged.prompt_tokens) {
                timing.begin_wait();
                device.synchronize();
                timing.end_wait();
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary = summary,
                    .timing  = timing.finish(),
                };
            }
            timing.resume_submit();
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(final_chunk_tokens) - 1, 1));
        } else if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (is_masked_draft_backend(speculative_backend)) {
                mark_workspace_usage(workspace_plan.dflash_context);
                // Decode rounds and other prefills rewrite the shared DFlash frame controls
                // between steps; this step's feature sink reads its lane from row 0.
                upload_dflash_prefill_controls(sequence);
            }
            std::uint32_t remaining          = nominal;
            std::uint32_t final_chunk_tokens = 0;
            bool finalized                   = false;
            while (remaining != 0) {
                schedule_state.text_kv_base           = staged.cursor;
                selectors                             = state_selectors(sequence);
                schedule_state.state_source_slot      = selectors.source;
                schedule_state.state_destination_slot = selectors.destination;
                const std::optional<std::uint32_t> hybrid_split = next_hybrid_split();
                if (staged.next_capture < staged.capture_groups.size() || hybrid_taps_left()) {
                    rewrite_capture_hidden =
                        state_images->continuation_hidden_slot(selectors.destination);
                    schedule_state.rewrite_checkpoint_hidden = &rewrite_capture_hidden;
                } else {
                    schedule_state.rewrite_checkpoint_hidden = nullptr;
                }

                const bool final_candidate = staged.cursor + remaining == staged.prompt_tokens;
                const std::optional<std::uint32_t> capture_frontier =
                    staged.next_capture < staged.capture_groups.size()
                        ? std::optional<std::uint32_t>(
                              staged.capture_groups[staged.next_capture].frontier)
                        : std::nullopt;
                std::optional<std::uint32_t> split_frontier = capture_frontier;
                // Rewrite execution frontiers split prefill for the Legacy catalog's rewrite
                // checkpoints and execution provenance. The hybrid cache captures nothing there:
                // it keys resume points by content and splits only at its own exact taps, so
                // each split would be a whole extra pass over the model for nothing.
                const auto& rewrite_frontiers = staged.prompt.identity.rewrite_execution_frontiers;
                const auto rewrite_split =
                    hybrid_lane != nullptr && hybrid_lane->active
                        ? rewrite_frontiers.end()
                        : std::upper_bound(rewrite_frontiers.begin(), rewrite_frontiers.end(),
                                           staged.cursor);
                if (rewrite_split != rewrite_frontiers.end() &&
                    (!split_frontier || *rewrite_split < *split_frontier)) {
                    split_frontier = *rewrite_split;
                }
                if (hybrid_split && (!split_frontier || *hybrid_split < *split_frontier)) {
                    split_frontier = *hybrid_split;
                }
                if (kvmem_window_pages != 0 && !sparse.query_checkpoint_valid &&
                    staged.prompt_tokens > kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize &&
                    sparse.query_begin > staged.cursor &&
                    (!split_frontier || sparse.query_begin < *split_frontier)) {
                    split_frontier = sparse.query_begin;
                }
                execution::PrefillChunkResult result;
                if (kvmem_window_pages != 0) { kv9_mark(device, "pre-chunk"); }
                timing.pause();
                if (staged.prompt.has_media()) {
                    if (!workspace_plan.vision) {
                        throw std::logic_error("active Vision prefill lost its workspace plan");
                    }
                    // An overlay window borrows its encode workspace outside this allocation.
                    if (workspace_plan.vision_resident) {
                        mark_workspace_usage(workspace_plan.vision->capacity_bytes);
                    }
                    result = execution::prefill_multimodal_chunk(schedule_state, staged.prompt,
                                                                 staged.vision.get(), remaining,
                                                                 split_frontier, final_candidate);
                } else {
                    result = execution::prefill_text_chunk(
                        schedule_state, std::span<const TokenId>(staged.prompt.token_ids),
                        remaining, split_frontier, final_candidate);
                }
                timing.include(result.timing);
                timing.resume_post();
                if (kvmem_window_pages != 0) { kv9_mark(device, "chunk"); }
                schedule_state.layer_ready = {};
                if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
                    throw std::logic_error("ordinary prefill chunk made invalid progress");
                }
                if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
                staged.cursor += result.processed_tokens;
                processed_prompt_tokens += result.processed_tokens;
                remaining -= result.processed_tokens;
                final_chunk_tokens     = result.processed_tokens;
                sequence.text_kv_valid = staged.cursor;
                if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
                if (is_masked_draft_backend(speculative_backend)) {
                    sequence.dflash_context_frontier = staged.cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                if (kvmem_window_pages != 0) {
                    consume_kvmem_chunk_capture(sequence, staged.cursor - result.processed_tokens,
                                                staged.cursor);
                    kv9_mark(device, "capture");
                    if (staged.cursor < staged.prompt_tokens) {
                        roll_sparse_prefill_window(sequence, staged.prompt_tokens, staged.cursor,
                                                   backend_kv_valid(sequence));
                        kv9_mark(device, "roll");
                    }
                }

                // Prompt transitions are canonical immediately. If this was the first write after
                // an immutable source, close the Fork before potentially freezing a new rewrite.
                settle_state_fork(sequence);
                if (hybrid_lane != nullptr && !result.finalized) {
                    hybrid_after_prefill_chunk(sequence, staged.cursor, staged.prompt_tokens);
                }
                if (kvmem_window_pages != 0 && staged.cursor == sparse.query_begin &&
                    staged.prompt_tokens > kvmem_prompt_window_pages(kvmem_window_pages) * kPagedKVPageSize) {
                    copy_kvmem_query_state(sequence, false);
                    sparse.query_checkpoint_valid = true;
                }
                const bool reached_capture = capture_frontier && staged.cursor == *capture_frontier;
                if (reached_capture) {
                    if (result.finalized) {
                        // The prompt-frontier state becomes publishable only after the generated
                        // Begin token is committed. commit() emits the offer for this group.
                    } else {
                        staged.elapsed_seconds +=
                            std::chrono::duration<double>(Clock::now() - started).count();
                        if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
                        staged.pending_capture_offer = next_capture_offer_id_;
                        return runtime::PrefillStepResult{
                            .summary                 = summary,
                            .processed_prompt_tokens = reported_tokens(),
                            .timing                  = timing.finish(),
                        };
                    }
                }

                finalized = result.finalized;
                if (finalized || remaining == 0) { break; }
            }

            if (!finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary                 = summary,
                    .processed_prompt_tokens = reported_tokens(),
                    .timing                  = timing.finish(),
                };
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            if (kvmem_window_pages != 0) {
                kv9_mark(device, "pre-finalize");
                finalize_kvmem_query(sequence, staged.prompt_tokens);
                kv9_mark(device, "finalize");
                sparse.replay_planned = needs_query_replay;
                apply_kvmem_retrieval_placement(sequence);
                kv9_mark(device, "place");
                if (needs_query_replay) {
                    // Hand control back before replay. Each subsequent step executes at most
                    // one chunk so Engine can observe cancellation at stable GPU boundaries.
                    staged.query_replay_cursor = sparse.query_begin;
                    timing.begin_wait();
                    device.synchronize();
                    timing.end_wait();
                    staged.elapsed_seconds +=
                        std::chrono::duration<double>(Clock::now() - started).count();
                    return runtime::PrefillStepResult{
                        .summary                 = summary,
                        .processed_prompt_tokens = processed_prompt_tokens,
                        .timing                  = timing.finish(),
                    };
                }
            }
            timing.resume_submit();
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(final_chunk_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            execution::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                          checked_i32(staged.prompt_tokens, "sample position"),
                                          ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                execution::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
                staged.mtp_bridge = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds       = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.prefix_digests.append_generated(std::span<const TokenId>(host_tokens, 1),
                                                 sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (is_masked_draft_backend(speculative_backend) &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (staged.vision) {
            const execution::VisionOverlayWindowStats overlay = staged.vision->overlay_stats();
            request.timings.overlay_windows           = overlay.windows;
            request.timings.overlay_exclusive_windows = overlay.exclusive_windows;
            request.timings.overlay_window_seconds    = overlay.window_seconds;
            request.timings.overlay_evict_seconds     = overlay.evict_seconds;
            request.timings.overlay_restore_seconds   = overlay.restore_seconds;
            request.timings.overlay_evicted_bytes     = overlay.evicted_bytes;
            request.timings.overlay_staged_bytes      = overlay.staged_bytes;
        }
        staged.prompt.release_all_media_payloads();
        if (staged.vision) { staged.vision->retire_handoff(); }

        const bool prompt_frontier_capture =
            staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == prompt_tokens;
        if (!prompt_frontier_capture) { request.prefill.reset(); }
        std::optional<FirstTokenLogprobs> first_token_logprobs;
        if (request.first_token_top_logprobs != 0) {
            first_token_logprobs = token_logprobs_from_bf16(
                std::span<const std::uint16_t>(
                    static_cast<const std::uint16_t*>(first_token_logits_host->data()),
                    public_tokens),
                host_tokens[0], request.first_token_top_logprobs);
        }
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = reported_tokens(),
            .complete                = true,
            .timing                  = timing.finish(),
            .first_token_logprobs    = std::move(first_token_logprobs),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        const std::uint32_t lane = sequence.lane;
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}



// Rolls the sparse prefill working set one chunk forward: maps the next chunk's growth
// pages (inside the window entitlement), then demotes every mapped page outside the
// sink prefix plus the rolling window to Host replicas. Runs at the chunk GPU boundary
// with no in-flight unit on the lane.
void ProgramImpl::roll_sparse_prefill_window(SequenceState& sequence, std::uint32_t prompt_tokens,
                                             std::uint32_t cursor,
                                             std::uint32_t backend_valid, bool retrieved_history) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    const std::uint32_t sink_pages = kvmem_sink_pages_cfg;  // upstream: one 128-token block
    const std::uint32_t next_target =
        std::min(prompt_tokens, cursor + prefill_chunk);
    const std::uint32_t next_pages = (next_target + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    std::uint32_t next_backend = backend_valid;
    if (sequence.kv->backend && speculative_backend == SpeculativeBackend::Mtp) {
        // During prefill mtp_kv_valid tracks the cursor, so the lead here is the draft
        // window unless verify state already runs ahead. The backend membership must
        // follow the roll target unconditionally: prompts past the window keep writing
        // MTP KV past the one-shot mapping start_sequence did.
        const std::uint32_t lead =
            backend_valid > cursor ? backend_valid - cursor : draft_window;
        next_backend =
            std::min<std::uint32_t>(capacity, next_target + lead);
        const std::uint32_t next_backend_pages =
            (next_backend + kPagedKVPageSize - 1U) / kPagedKVPageSize;
        if (next_backend_pages > backend_kv_addresses->entitlement(*sequence.kv->backend)) {
            backend_kv_addresses->resize_entitlement(*sequence.kv->backend, next_backend_pages);
        }
    }
    // Demote first (shrinking the residency floor), then grow the entitlement, then
    // materialize the next chunk's pages inside it.
    const std::uint32_t mapped_pages = text_kv_addresses->mapped_pages(sequence.kv->text);
    const auto window = !sparse.media_groups.empty()
                            ? media_window_page_set(mapped_pages, kvmem_window_pages,
                                  retrieved_history ? std::span<const std::uint32_t>(sparse.retrieved_pages)
                                                    : std::span<const std::uint32_t>{}, sparse.media_groups)
                        : retrieved_history
                            ? decode_window_page_set(mapped_pages, kvmem_window_pages,
                                                     sparse.retrieved_pages)
                            : prefill_window_page_set(mapped_pages, sink_pages,
                                                      kvmem_window_pages - sink_pages);
    text_kv_addresses->apply_device_placement(sequence.kv->text, *host_kv_extents, window,
                                              device.transfer_stream,
                                              retrieved_history ? "replay" : "prefill");
    if (next_pages > text_kv_addresses->entitlement(sequence.kv->text)) {
        text_kv_addresses->resize_entitlement(sequence.kv->text, next_pages);
    }
    // Grow membership directly: the roll's own placement above just rang the window, so
    // routing through ensure_sequence_kv_mapped would re-run the decode ring and clamp
    // the growth reservation this resize just established.
    text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, next_target, device.stream);
    if (next_backend != 0 && sequence.kv->backend) {
        backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, next_backend,
                                                      device.stream);
    }
    if (sequence.kv->backend && speculative_backend == SpeculativeBackend::Mtp) {
        // The MTP follower rolls its own window: demote the overflow to Host replicas so
        // backend residency tracks the window plus draft lead, not the whole prefix.
        const std::uint32_t lead_pages = (draft_window + kPagedKVPageSize - 1U) / kPagedKVPageSize;
        const std::uint32_t backend_mapped =
            backend_kv_addresses->mapped_pages(*sequence.kv->backend);
        const auto backend_window = !sparse.media_groups.empty()
            ? media_window_page_set(backend_mapped, kvmem_window_pages + lead_pages,
                  retrieved_history ? std::span<const std::uint32_t>(sparse.retrieved_pages)
                                    : std::span<const std::uint32_t>{}, sparse.media_groups)
            : retrieved_history
            ? decode_window_page_set(backend_mapped, kvmem_window_pages + lead_pages,
                                     sparse.retrieved_pages)
            : prefill_window_page_set(backend_mapped, sink_pages,
                                      kvmem_window_pages - sink_pages + lead_pages);
        backend_kv_addresses->apply_device_placement(*sequence.kv->backend, *host_kv_extents,
                                                     backend_window, device.transfer_stream,
                                                     retrieved_history ? "replay" : "prefill");
    }
}

void ProgramImpl::copy_kvmem_query_state(SequenceState& sequence, bool restore) {
    auto& live = state_images->linear();
    auto& snapshot = *kvmem_query_checkpoint_;
    const auto slot = state_selectors(sequence).destination;
    if (is_masked_draft_backend(speculative_backend)) {
        const auto query_begin = kvmem_lanes_.at(sequence.lane).query_begin;
        if (!dflash || !kvmem_draft_checkpoint_ || dflash->full ||
            (!restore && sequence.dflash_context_frontier != query_begin)) {
            throw std::logic_error("sparse draft checkpoint does not match its query frontier");
        }
        if (restore) {
            dflash->local.copy_slot_from(*kvmem_draft_checkpoint_, sequence.lane, slot, device.stream);
            sequence.dflash_context_frontier = query_begin;
            // No pending decode feature is valid at a prefill checkpoint. Replay
            // rebuilds local KV from target features; the probe publishes no draft.
            const auto pending = dflash->pending_features.slice(2, sequence.lane, 1);
            CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
        } else {
            kvmem_draft_checkpoint_->copy_slot_from(dflash->local, slot, sequence.lane, device.stream);
        }
    }
    for (std::uint32_t layer = 0; layer < live.layer_count(); ++layer) {
        for (const bool recurrent : {false, true}) {
            const Tensor active = recurrent ? live.recurrent_slot(layer, slot) : live.conv_slot(layer, slot);
            const Tensor saved = recurrent ? snapshot.recurrent_slot(layer, sequence.lane) : snapshot.conv_slot(layer, sequence.lane);
            CUDA_CHECK(cudaMemcpyAsync(restore ? active.data : saved.data,
                                       restore ? saved.data : active.data, active.bytes(),
                                       cudaMemcpyDeviceToDevice, device.stream));
        }
    }
}

std::uint32_t ProgramImpl::advance_kvmem_query_replay(SequenceState& sequence,
                                                     RequestControl::Prefill& staged,
                                                     runtime::ExecutionTimingRecorder& timing) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    if (!sparse.query_checkpoint_valid || !staged.query_replay_cursor ||
        !staged.capture_groups.empty()) {
        throw std::logic_error("sparse query replay has no private checkpoint");
    }
    auto& cursor = *staged.query_replay_cursor;
    if (cursor < sparse.query_begin || cursor >= staged.prompt_tokens) {
        throw std::logic_error("sparse query replay cursor is outside the prompt");
    }
    if (cursor == sparse.query_begin) {
        const auto* counts = requests[sequence.lane].sampling_host.token_counts;
        if (counts != nullptr && std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            // Admission resets generated-token counts. The unpublished probe must not
            // change penalty history before replay samples the actual Begin token.
            TokenId probe_token = 0;
            std::int32_t probe_count = 0;
            CUDA_CHECK(cudaMemcpy(&probe_token, io.token.data, sizeof(probe_token),
                                   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&probe_count, counts + probe_token, sizeof(probe_count),
                                   cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "KVMEM probe sample_history=%d\n", probe_count);
        }
        // Rewind the unpublished probe's complete causal state exactly once.
        copy_kvmem_query_state(sequence, true);
        if (staged.vision) {
            staged.vision->begin_replay(cursor);
            if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
                std::fprintf(stderr, "KVMEM vision replay begin=%u end=%u lane=%u\n",
                             cursor, staged.prompt_tokens, sequence.lane);
            }
        }
        text_kv_addresses->truncate_for_replay(sequence.kv->text, cursor, *host_kv_extents);
        if (sequence.kv->backend) {
            backend_kv_addresses->truncate_for_replay(*sequence.kv->backend, cursor, *host_kv_extents);
        }
        const auto historical_pages = (cursor + kPagedKVPageSize - 1U) / kPagedKVPageSize;
        sparse.retrieved_pages.erase(
            std::lower_bound(sparse.retrieved_pages.begin(), sparse.retrieved_pages.end(), historical_pages),
            sparse.retrieved_pages.end());
        sequence.text_kv_valid = cursor;
        if (staged.prepare_mtp) sequence.mtp_kv_valid = cursor;
    }

    // One bounded execution unit. Placement republishes the truncated execution row
    // before this chunk; the caller synchronizes before returning control to Engine.
    {
        roll_sparse_prefill_window(sequence, staged.prompt_tokens, cursor,
                                   backend_kv_valid(sequence), true);
        const auto slots = state_selectors(sequence);
        execution::PrefillContext replay{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
             mtp_attention_window}, // No Q/K accumulation: the probe owns the retrieval features.
            text_kv_view(sequence), mtp_kv_view(sequence), decoder->text_kv, decoder->mtp_cache(),
            dflash ? &*dflash : nullptr, cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            nullptr, slots.source, slots.destination, staged.initial_mtp_extent, dflash_host_ingress};
        replay.dflash_kv_table_row =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        if (dflash) { mark_workspace_usage(workspace_plan.dflash_context); }
        const auto count = std::min(prefill_chunk, staged.prompt_tokens - cursor);
        const auto split = std::upper_bound(staged.prompt.identity.rewrite_execution_frontiers.begin(),
                                            staged.prompt.identity.rewrite_execution_frontiers.end(), cursor);
        const auto frontier = split == staged.prompt.identity.rewrite_execution_frontiers.end()
                                  ? std::optional<std::uint32_t>{} : std::optional<std::uint32_t>{*split};
        timing.pause();
        const auto result = staged.prompt.has_media()
            ? execution::prefill_multimodal_chunk(replay, staged.prompt, staged.vision.get(),
                                                  count, frontier, cursor + count == staged.prompt_tokens)
            : execution::prefill_text_chunk(replay, staged.prompt.token_ids, count, frontier,
                                             cursor + count == staged.prompt_tokens);
        timing.include(result.timing);
        timing.resume_post();
        if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
        if (result.processed_tokens == 0 || result.processed_tokens > count) {
            throw std::logic_error("sparse query replay made invalid progress");
        }
        cursor += result.processed_tokens;
        sequence.text_kv_valid = cursor;
        if (staged.prepare_mtp) sequence.mtp_kv_valid = cursor;
        if (is_masked_draft_backend(speculative_backend)) {
            sequence.dflash_context_frontier = cursor;
        }
        commit_sequence_kv(sequence, cursor, backend_kv_valid(sequence));
        if (cursor == staged.prompt_tokens && !result.finalized) {
            throw std::logic_error("sparse query replay did not replace the probe sample");
        }
        if (cursor == staged.prompt_tokens && std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
            std::fprintf(stderr, "KVMEM replay begin=%u end=%u tokens=%u\n", sparse.query_begin,
                         cursor, cursor - sparse.query_begin);
        }
        return result.processed_tokens;
    }
}

// Publishes the completed-block key sums of one prefill chunk into the retrieval index.
// Global block IDs select ring slots. An incomplete block retains its partial sum
// across chunk boundaries; only completed slots are cleared after publication.
void ProgramImpl::consume_kvmem_chunk_capture(SequenceState& sequence, std::uint32_t chunk_begin,
                                              std::uint32_t chunk_end) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    const std::uint32_t kLayers = sparse.index.layers();
    const std::uint32_t kKvWidth = sparse.index.kv_heads() * sparse.index.head_dim();
    const std::uint32_t first_block =
        chunk_begin / ops::kKvmemCaptureBlockTokens;
    const std::uint32_t last_block = chunk_end / ops::kKvmemCaptureBlockTokens;
    if (last_block <= first_block) {
        // No block completed; still advance the index so block numbering tracks tokens.
        while (sparse.index.total_tokens() < chunk_end) {
            const std::uint32_t step = std::min(
                ops::kKvmemCaptureBlockTokens,
                chunk_end - sparse.index.total_tokens());
            (void)sparse.index.append(step);
        }
        return;
    }
    while (sparse.index.total_tokens() < chunk_end) {
        const std::uint32_t step =
            std::min(ops::kKvmemCaptureBlockTokens, chunk_end - sparse.index.total_tokens());
        (void)sparse.index.append(step);
    }
    const std::uint32_t block_count = last_block - first_block;
    std::vector<float> sums(static_cast<std::size_t>(kLayers) *
                            kvmem_capture_slots_ * kKvWidth);
    CUDA_CHECK(cudaMemcpyAsync(sums.data(), sparse.key_sums.data,
                               sums.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<float> mean(kKvWidth);
    for (std::uint32_t offset = 0; offset < block_count; ++offset) {
        const auto block = first_block + offset;
        const auto slot = block % kvmem_capture_slots_;
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            const float* source =
                sums.data() +
                (static_cast<std::size_t>(layer) * kvmem_capture_slots_ + slot) * kKvWidth;
            for (std::uint32_t element = 0; element < kKvWidth; ++element) {
                mean[element] = source[element] /
                                static_cast<float>(ops::kKvmemCaptureBlockTokens);
            }
            if (block * ops::kKvmemCaptureBlockTokens >= sparse.capture_begin) {
                sparse.index.write_block_mean(block, layer, mean);
            }
            auto* device_sum = static_cast<float*>(sparse.key_sums.data) +
                (static_cast<std::size_t>(layer) * kvmem_capture_slots_ + slot) * kKvWidth;
            CUDA_CHECK(cudaMemsetAsync(device_sum, 0, kKvWidth * sizeof(float), device.stream));
        }
    }
}

void ProgramImpl::plan_kv9_query(KvmemLaneState& sparse, const PreparedPromptData& prompt,
                                 std::uint32_t prompt_tokens, std::uint32_t base) {
    sparse.kv9_plan = false;
    sparse.kv9_tool = false;
    sparse.replay_planned = false;
    sparse.seg_n = 0;
    sparse.seg_accum = 0;
    sparse.seg_injected = 0;
    for (std::uint32_t k = 0; k < kKv9MaxSeg; ++k) {
        sparse.seg_slot[k] = -1;
        sparse.seg_full[k] = false;
        sparse.seg_count[k] = 0;
        sparse.seg_query[k].clear();
        sparse.seg_rows[k].clear();
        sparse.seg_rows_n[k] = 0;
        sparse.seg_msg[k] = -1;
        sparse.slot_row0[k] = UINT32_MAX;
    }
    for (int m = 0; m < 2; ++m) { sparse.msg_b[m] = -1; sparse.msg_e[m] = -1; sparse.msg_hash[m] = 0; sparse.msg_stash[m] = false; }
    sparse.stash_n = 0;
    sparse.protect_b.clear();
    sparse.protect_e.clear();
    sparse.protect_used = 0;
    sparse.protect_cap = 0;
    sparse.protect_units = 0;
    const auto& attention = *parameters.model.config().text.attention;
    const std::uint32_t kLayers = sparse.index.layers();
    const std::size_t qw = static_cast<std::size_t>(attention.query_width());
    const std::uint32_t prompt_pages = kvmem_prompt_window_pages(kvmem_window_pages);
    const std::uint32_t window_tokens = prompt_pages * kPagedKVPageSize;
    const bool have_ids = prompt.token_ids.size() >= prompt_tokens;
    const std::span<const TokenId> ids =
        have_ids ? std::span<const TokenId>(prompt.token_ids.data(), prompt_tokens) : std::span<const TokenId>{};
    std::uint32_t rows_used = 0;
    const auto add_accum = [&](std::uint32_t k, std::uint32_t b, std::uint32_t e, bool rows) {
        const auto slot = sparse.seg_accum++;
        sparse.seg_slot[k] = static_cast<std::int32_t>(slot);
        sparse.acc_range[2U * slot] = b;
        sparse.acc_range[2U * slot + 1U] = e;
        if (rows && sparse.q_rows && rows_used + (e - b) <= sparse.q_rows_cap) {
            sparse.slot_row0[slot] = rows_used;
            sparse.seg_rows_n[k] = e - b;
            rows_used += e - b;
        }
    };
    if (kv9_enabled() && have_ids) {
        const Kv9Plan plan = kv9_make_plan(ids);
        sparse.kv9_shadow.clear();
        if (const char* sh = std::getenv("KV9_QSHADOW"); sh != nullptr && *sh != '\0') {
            for (const char* cur = sh; cur != nullptr && *cur != '\0';) {
                char* next = nullptr;
                const auto q = static_cast<std::int32_t>(std::strtol(cur, &next, 10));
                cur = (next != nullptr && *next == ',') ? next + 1 : nullptr;
                if (q < 8 || q >= kv9_query_max()) { continue; }
                const Kv9Plan sp = kv9_make_plan(ids, q);
                std::vector<std::uint32_t> segs;
                for (std::uint32_t i = 0; i < sp.nseg; ++i) {
                    segs.push_back(sp.begin[i]);
                    segs.push_back(std::min(sp.end[i], prompt_tokens));
                }
                sparse.kv9_shadow.emplace_back(static_cast<std::uint32_t>(q), std::move(segs));
            }
        }
        sparse.kv9_tool = plan.tool;
        sparse.msg_b[0] = plan.msg_begin;
        sparse.msg_e[0] = plan.msg_end;
        sparse.msg_stash[0] = !plan.tool;
        sparse.msg_b[1] = plan.real_begin;
        sparse.msg_e[1] = plan.real_end;
        sparse.msg_stash[1] = plan.tool && plan.real_begin >= 0;
        sparse.stash_n = plan.stash_n;
        for (std::uint32_t i = 0; i < 2U; ++i) { sparse.stash_b[i] = plan.stash_b[i]; sparse.stash_e[i] = plan.stash_e[i]; }
        for (int m = 0; m < 2; ++m) {
            if (sparse.msg_b[m] >= 0 && sparse.msg_e[m] > sparse.msg_b[m]) {
                sparse.msg_hash[m] = kv9_hash(ids, static_cast<std::uint32_t>(sparse.msg_b[m]),
                                              static_cast<std::uint32_t>(sparse.msg_e[m]));
            }
        }
        for (std::uint32_t i = 0; i < plan.nseg; ++i) {
            const std::uint32_t b = plan.begin[i], e = std::min(plan.end[i], prompt_tokens);
            if (e <= b) { continue; }
            const std::uint32_t k = sparse.seg_n;
            const int m = plan.msg[i];
            sparse.seg_range[2U * k] = b;
            sparse.seg_range[2U * k + 1U] = e;
            sparse.seg_hash[k] = kv9_hash(ids, b, e);
            sparse.seg_msg[k] = m;
            if (b >= base) {
                add_accum(k, b, e, true);
                sparse.seg_full[k] = true;
                ++sparse.seg_n;
                continue;
            }
            // Before the reuse base: take the rows from the stash of the same message.
            bool injected = false;
            for (auto& st : sparse.row_stash) {
                if (m < 0 || st.b != sparse.msg_b[m] || st.e != sparse.msg_e[m] || st.hash != sparse.msg_hash[m]) { continue; }
                for (std::size_t r = 0; r < st.rb.size() && !injected; ++r) {
                    if (st.rb[r] > b || e > st.re[r]) { continue; }
                    const std::uint32_t n = e - b, first = st.r0[r] + (b - st.rb[r]);
                    std::vector<std::uint16_t>& dst = sparse.seg_rows[k];
                    dst.resize(static_cast<std::size_t>(kLayers) * n * qw);
                    for (std::uint32_t l = 0; l < kLayers; ++l) {
                        std::memcpy(dst.data() + static_cast<std::size_t>(l) * n * qw,
                                    st.rows.data() + (static_cast<std::size_t>(l) * st.nrows + first) * qw,
                                    static_cast<std::size_t>(n) * qw * sizeof(std::uint16_t));
                    }
                    sparse.seg_rows_n[k] = n;
                    st.tick = ++sparse.stash_tick;
                    injected = true;
                }
                if (injected) { break; }
            }
            if (injected) {
                ++sparse.seg_injected;
                ++sparse.seg_n;
            } else if (e > base) {
                add_accum(k, base, e, true);
                ++sparse.seg_n;
            }
        }
        sparse.kv9_plan = sparse.seg_n != 0;
    }
    if (sparse.seg_n == 0) {
        // Legacy query (last 512 tokens of the user message / prompt suffix) as one segment.
        const auto legacy = kvmem_query_span(prompt_tokens, base, prompt.retrieval_query, prompt.vision_items);
        if (legacy.end > legacy.begin) {
            sparse.seg_range[0] = legacy.begin;
            sparse.seg_range[1] = legacy.end;
            add_accum(0, legacy.begin, legacy.end, kv9_enabled());
            sparse.seg_n = 1;
        }
    }
    // kv8b protect-new v2: only over-window text prompts (media keep the qz group rules).
    if (sparse.kv9_plan && kv9_protect_enabled() && prompt.vision_items.empty() &&
        prompt_tokens > window_tokens) {
        const std::uint32_t sink_pages = kvmem_sink_pages_cfg;
        const std::uint32_t rest = prompt_pages > sink_pages ? prompt_pages - sink_pages : 0U;
        // kv8b: 12288 of a 32768 retrieval share (3/8); scaled to this window.
        const std::int64_t cap = std::min<std::int64_t>(
            kv9_env_i32("NINFER_KVMEM_PROTECT_MAX", 12288),
            static_cast<std::int64_t>(rest) * kPagedKVPageSize * 3 / 8);
        Kv9Protect pr = kv9_make_protect(ids, cap, static_cast<std::int64_t>(sink_pages) * kPagedKVPageSize);
        sparse.protect_b = std::move(pr.b);
        sparse.protect_e = std::move(pr.e);
        sparse.protect_used = pr.used;
        sparse.protect_cap = cap;
        sparse.protect_units = pr.units;
    }
}

// Folds the per-segment query sums into GQA-summed query means (cosine fallback / gate) and
// stashes the real user message's query rows for later tool rounds.
void ProgramImpl::finalize_kvmem_query(SequenceState& sequence, std::uint32_t prompt_tokens) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    const auto& attention = *parameters.model.config().text.attention;
    const std::uint32_t kLayers = sparse.index.layers();
    const std::uint32_t kQHeads = attention.num_attention_heads;
    const std::uint32_t kKvHeads = attention.num_key_value_heads;
    const std::uint32_t kHeadDim = attention.head_dim;
    const std::size_t qw = static_cast<std::size_t>(kQHeads) * kHeadDim;
    const std::size_t stride = static_cast<std::size_t>(kLayers) * qw;
    std::vector<float> sums(stride * std::max(1U, sparse.seg_accum));
    if (sparse.seg_accum != 0) {
        CUDA_CHECK(cudaMemcpyAsync(sums.data(), sparse.query_sum.data,
                                   stride * sparse.seg_accum * sizeof(float), cudaMemcpyDeviceToHost,
                                   device.stream));
    }
    // The query rows were copied on the layer streams; wait for all of them.
    device.synchronize();
    const auto fold = [&](const float* src, std::uint32_t count) {
        std::vector<float> q(static_cast<std::size_t>(kLayers) * kKvHeads * kHeadDim, 0.0F);
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            for (std::uint32_t head = 0; head < kQHeads; ++head) {
                const std::size_t source = (static_cast<std::size_t>(layer) * kQHeads + head) * kHeadDim;
                const std::size_t target =
                    (static_cast<std::size_t>(layer) * kKvHeads + head / (kQHeads / kKvHeads)) * kHeadDim;
                for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) { q[target + dim] += src[source + dim]; }
            }
        }
        const float scale = 1.0F / static_cast<float>(count);
        double norm = 0;
        for (float& value : q) { value *= scale; norm += static_cast<double>(value) * value; }
        if (!std::isfinite(norm) || norm == 0) { q.clear(); }
        return q;
    };
    sparse.query.clear();
    sparse.query_count.clear();
    std::uint32_t active = 0;
    for (std::uint32_t k = 0; k < sparse.seg_n; ++k) {
        sparse.seg_query[k].clear();
        sparse.seg_count[k] = 0;
        if (sparse.seg_slot[k] < 0) {
            // injected: the mean comes from the stashed rows
            const std::uint32_t n = sparse.seg_rows_n[k];
            if (n == 0 || sparse.seg_rows[k].empty()) { continue; }
            std::vector<float> acc(stride, 0.0F);
            for (std::uint32_t l = 0; l < kLayers; ++l) {
                for (std::uint32_t r = 0; r < n; ++r) {
                    const std::uint16_t* row = sparse.seg_rows[k].data() + (static_cast<std::size_t>(l) * n + r) * qw;
                    float* dst = acc.data() + static_cast<std::size_t>(l) * qw;
                    for (std::size_t i = 0; i < qw; ++i) { dst[i] += kv9_bf16(row[i]); }
                }
            }
            sparse.seg_query[k] = fold(acc.data(), n);
            if (!sparse.seg_query[k].empty()) { sparse.seg_count[k] = n; ++active; }
            continue;
        }
        const auto slot = static_cast<std::uint32_t>(sparse.seg_slot[k]);
        const std::uint32_t ab = sparse.acc_range[2U * slot];
        const std::uint32_t ae = std::min(prompt_tokens, sparse.acc_range[2U * slot + 1U]);
        const std::uint32_t count = ae > ab ? ae - ab : 0U;
        if (count == 0) { sparse.seg_rows_n[k] = 0; continue; }
        if (sparse.seg_rows_n[k] != count) { sparse.seg_rows_n[k] = std::min(sparse.seg_rows_n[k], count); }
        sparse.seg_query[k] = fold(sums.data() + stride * slot, count);
        if (!sparse.seg_query[k].empty()) { sparse.seg_count[k] = count; ++active; }
    }
    // kv9 stash: the rows of the real user message that a later tool round asks for (head/tail of
    // the half budget), copied out of the live segments that cover them.
    for (int m = 0; m < 2; ++m) {
        if (!sparse.msg_stash[m] || sparse.msg_b[m] < 0 || !sparse.q_rows || sparse.stash_n == 0) { continue; }
        KvmemLaneState::RowStash st;
        st.b = sparse.msg_b[m];
        st.e = sparse.msg_e[m];
        st.hash = sparse.msg_hash[m];
        std::vector<std::uint32_t> src_row;   // first pinned row of each stash range
        for (std::uint32_t i = 0; i < sparse.stash_n; ++i) {
            const std::uint32_t sb = sparse.stash_b[i], se = sparse.stash_e[i];
            for (std::uint32_t k = 0; k < sparse.seg_n; ++k) {
                if (sparse.seg_msg[k] != m || !sparse.seg_full[k] || sparse.seg_slot[k] < 0) { continue; }
                const auto slot = static_cast<std::uint32_t>(sparse.seg_slot[k]);
                const std::uint32_t rb = sparse.acc_range[2U * slot], re = sparse.acc_range[2U * slot + 1U];
                if (sparse.slot_row0[slot] == UINT32_MAX || sparse.seg_rows_n[k] != re - rb) { continue; }
                if (rb > sb || se > re) { continue; }
                st.rb.push_back(sb);
                st.re.push_back(se);
                st.r0.push_back(st.nrows);
                src_row.push_back(sparse.slot_row0[slot] + (sb - rb));
                st.nrows += se - sb;
                break;
            }
        }
        if (st.nrows == 0) { continue; }
        st.rows.resize(static_cast<std::size_t>(kLayers) * st.nrows * qw);
        for (std::size_t r = 0; r < st.rb.size(); ++r) {
            const std::uint32_t n = st.re[r] - st.rb[r];
            for (std::uint32_t l = 0; l < kLayers; ++l) {
                std::memcpy(st.rows.data() + (static_cast<std::size_t>(l) * st.nrows + st.r0[r]) * qw,
                            sparse.q_rows.get() + (static_cast<std::size_t>(l) * sparse.q_rows_cap + src_row[r]) * qw,
                            static_cast<std::size_t>(n) * qw * sizeof(std::uint16_t));
            }
        }
        st.tick = ++sparse.stash_tick;
        auto it = std::find_if(sparse.row_stash.begin(), sparse.row_stash.end(), [&](const auto& x) {
            return x.b == st.b && x.e == st.e && x.hash == st.hash;
        });
        if (it != sparse.row_stash.end()) {
            *it = std::move(st);
        } else if (sparse.row_stash.size() < 2) {
            sparse.row_stash.push_back(std::move(st));
        } else {
            *std::min_element(sparse.row_stash.begin(), sparse.row_stash.end(),
                              [](const auto& x, const auto& y) { return x.tick < y.tick; }) = std::move(st);
        }
    }
    for (std::uint32_t k = 0; k < sparse.seg_n; ++k) {
        if (!sparse.seg_query[k].empty()) {
            sparse.query = sparse.seg_query[k];
            sparse.query_count.assign(kLayers, sparse.seg_count[k]);
            break;
        }
    }
    if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
        std::fprintf(stderr, "KVMEM capture segs=%u active=%u blocks=%u lane=%u\n", sparse.seg_n, active,
                     sparse.index.block_count(), sequence.lane);
    }
    CUDA_CHECK(cudaMemsetAsync(sparse.query_sum.data, 0, sparse.query_sum.bytes(), device.stream));
}

// Replaces the rolling recency window with the retrieval-scored window once a turn's
// query means exist. Retrieval, protected new content and recent pages share one window.
void ProgramImpl::apply_kvmem_retrieval_placement(SequenceState& sequence) {
    auto& sparse = kvmem_lanes_.at(sequence.lane);
    constexpr std::uint32_t kBlockTokens = 128U;
    const std::uint32_t mapped = text_kv_addresses->mapped_pages(sequence.kv->text);
    const std::uint32_t prompt_pages = kvmem_prompt_window_pages(kvmem_window_pages);
    if (mapped <= prompt_pages) { return; }
    if (sparse.query_count.empty() || sparse.query_count[0] == 0) { return; }
    const std::uint32_t blocks = sparse.index.block_count();
    if (blocks == 0) { return; }
    detail::BlockSelectionConfig config;
    config.block_tokens  = kBlockTokens;
    // The device budget is one window shared by sink, recency, and retrieval; the
    // selection must leave the recency and sink share inside the window or the
    // placement's promote side overflows the pool.
    // Old-style allocation: retrieval compresses the prompt into window - gen reserve; the
    // sink prefix is kept whole and the recency share comes out of what remains after it.
    const std::uint32_t sink_pages   = kvmem_sink_pages_cfg;
    std::uint32_t recent_pages =
        sink_pages > 2U ? (prompt_pages - sink_pages) / 4U : prompt_pages / 4U;
    config.sink_blocks   = std::max(1U, sink_pages / 2U);
    config.recent_blocks = 2U;
    // kv8b: protected new content (newest messages / tool results, previous answer tail) is
    // mandatory and replaces most of the fixed recency share; decode still fills the rest of
    // the window (generation reserve) with the newest pages.
    std::vector<std::uint32_t> mandatory;
    std::vector<std::uint8_t> include(blocks, 1U);
    if (!sparse.protect_b.empty() && sparse.media_groups.empty()) {
        const std::uint32_t cap_blocks =
            static_cast<std::uint32_t>(sparse.protect_cap / kBlockTokens) + 8U;
        for (std::size_t r = 0; r < sparse.protect_b.size() && mandatory.size() < cap_blocks; ++r) {
            const std::uint32_t first = sparse.protect_b[r] / kBlockTokens;
            const std::uint32_t last = (sparse.protect_e[r] - 1U) / kBlockTokens;
            for (std::uint32_t block = first; block <= last && block < blocks; ++block) {
                if (block < config.sink_blocks || !include[block]) { continue; }
                include[block] = 0U;
                mandatory.push_back(block);
                if (mandatory.size() >= cap_blocks) { break; }
            }
        }
        if (!mandatory.empty()) { recent_pages = 4U; }
    }
    if (sparse.replay_planned && sparse.query_begin != 0 && sparse.query_begin / kPagedKVPageSize < mapped) {
        // kv9: the replayed new input stays in the window as the recent share.
        const std::uint32_t limit = prompt_pages > sink_pages ? (prompt_pages - sink_pages) / 2U : prompt_pages / 2U;
        recent_pages = std::max(recent_pages,
                                std::min(limit, mapped - sparse.query_begin / kPagedKVPageSize + 1U));
    }
    config.budget_blocks = (prompt_pages - recent_pages - 2U) / 2U;
    // Blocks kept anyway are not candidates (paper: C excludes sink and recent blocks).
    for (std::uint32_t block = 0; block < std::min(config.sink_blocks, blocks); ++block) { include[block] = 0U; }
    const std::uint32_t recent_blocks_all = config.recent_blocks + (recent_pages + 1U) / 2U;
    for (std::uint32_t block = blocks > recent_blocks_all ? blocks - recent_blocks_all : 0U; block < blocks; ++block) {
        include[block] = 0U;
    }

    std::vector<std::uint32_t> active;
    for (std::uint32_t k = 0; k < sparse.seg_n; ++k) {
        if (!sparse.seg_query[k].empty()) { active.push_back(k); }
    }
    // Per-segment scores: Eq.10 where the query rows exist, else the qz cosine.
    const auto& attention = *parameters.model.config().text.attention;
    const auto t_score = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> seg_scores(active.size());
    std::vector<Kv9RowView> views;
    std::vector<std::size_t> view_of;
    for (std::size_t i = 0; i < active.size(); ++i) {
        const std::uint32_t k = active[i];
        const std::uint32_t n = sparse.seg_rows_n[k];
        if (n == 0 || n != sparse.seg_count[k]) { continue; }
        Kv9RowView v;
        if (sparse.seg_slot[k] < 0) {
            if (sparse.seg_rows[k].empty()) { continue; }
            v = {sparse.seg_rows[k].data(), n, n};
        } else {
            const auto slot = static_cast<std::uint32_t>(sparse.seg_slot[k]);
            if (sparse.slot_row0[slot] == UINT32_MAX || !sparse.q_rows) { continue; }
            v = {sparse.q_rows.get() + static_cast<std::size_t>(sparse.slot_row0[slot]) * attention.query_width(),
                 n, sparse.q_rows_cap};
        }
        views.push_back(v);
        view_of.push_back(i);
    }
    std::uint32_t eq10_segs = 0;
    if (!views.empty()) {
        // Paper Eq.10: the candidate set C excludes the blocks kept unconditionally (sink, recent,
        // and here also the protected new content), so their mass does not dilute the middle.
        std::vector<std::vector<float>> out;
        kv9_eq10_scores(sparse.index, views, attention.num_attention_heads, include, out);
        for (std::size_t j = 0; j < views.size(); ++j) {
            std::vector<float>& sc = seg_scores[view_of[j]];
            sc.assign(blocks, std::numeric_limits<float>::quiet_NaN());
            for (std::uint32_t block = 0; block < blocks; ++block) {
                // ranked among the blocks not kept anyway (sink / mandatory / recent pages)
                if (include[block] && sparse.index.block(block).full) { sc[block] = out[j][block]; }
            }
            ++eq10_segs;
        }
    }
    for (std::size_t i = 0; i < active.size(); ++i) {
        if (!seg_scores[i].empty()) { continue; }
        const std::uint32_t k = active[i];
        const std::vector<std::uint32_t> counts(sparse.index.layers(), sparse.seg_count[k]);
        seg_scores[i].assign(blocks, std::numeric_limits<float>::quiet_NaN());
        for (std::uint32_t block = 0; block < blocks; ++block) {
            if (!include[block] || !sparse.index.block(block).full) { continue; }
            (void)sparse.index.score(block, sparse.seg_query[k], counts, seg_scores[i][block]);
        }
    }
    const double score_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_score).count();
    // Segment merge. "sum" (paper Eq.10): the raw Eq.10 masses of all query tokens are
    // added (R_b = sum over every query token); "nsum" (kv9 QUERY_MERGE=sum): per-segment
    // normalised scores averaged; "split" (kv9 default, default here): segments take turns (best rank).
    std::vector<float> scores(blocks, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> score_split(blocks, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> score_nsum(blocks, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> score_sum(blocks, std::numeric_limits<float>::quiet_NaN());
    std::string seg_log;
    // 10-02 test (19万, window 648): needle block rank D1 sum 194 / nsum 93 / split 91, tool round
    // sum 5 / nsum 10 / split 3 -> split stays the default (a long head segment of document text
    // outweighs a short question under the raw sum).
    int merge_mode = 2;   // 0 sum, 1 nsum, 2 split
    {
        const char* m = std::getenv("KV9_MERGE");
        if (m == nullptr || *m == '\0') { m = std::getenv("NINFER_KVMEM_QUERY_MERGE"); }
        if (m != nullptr && std::strcmp(m, "sum") == 0) { merge_mode = 0; }
        if (m != nullptr && std::strcmp(m, "nsum") == 0) { merge_mode = 1; }
    }
    if (merge_mode == 0 && eq10_segs != active.size()) { merge_mode = 1; }   // cosine scores are not masses
    {
        std::vector<std::uint32_t> best(blocks, blocks);
        std::vector<double> mean(blocks, 0.0);
        std::vector<std::uint32_t> order;
        order.reserve(blocks);
        for (std::size_t i = 0; i < active.size(); ++i) {
            const std::vector<float>& sc = seg_scores[i];
            order.clear();
            double total = 0.0;
            for (std::uint32_t block = 0; block < blocks; ++block) {
                if (!std::isnan(sc[block])) { order.push_back(block); total += std::fabs(sc[block]); }
            }
            std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) { return sc[x] > sc[y]; });
            for (std::uint32_t r = 0; r < order.size(); ++r) {
                best[order[r]] = std::min(best[order[r]], r);
                if (total > 0.0) { mean[order[r]] += sc[order[r]] / total / static_cast<double>(active.size()); }
            }
            char line[200];
            std::snprintf(line, sizeof(line), " [%u,%u)%s top %u,%u,%u;", sparse.seg_range[2U * active[i]],
                          sparse.seg_range[2U * active[i] + 1U], sparse.seg_slot[active[i]] < 0 ? "S" : "",
                          order.size() > 0 ? order[0] * kBlockTokens : 0U,
                          order.size() > 1 ? order[1] * kBlockTokens : 0U,
                          order.size() > 2 ? order[2] * kBlockTokens : 0U);
            seg_log += line;
        }
        for (std::uint32_t block = 0; block < blocks; ++block) {
            if (best[block] >= blocks) { continue; }
            score_nsum[block] = static_cast<float>(mean[block]);
            score_split[block] = active.size() == 1U ? static_cast<float>(mean[block])
                                                     : static_cast<float>(blocks - best[block]) + 0.5F * static_cast<float>(mean[block]);
            double raw = 0.0;
            for (std::size_t i = 0; i < active.size(); ++i) {
                if (!std::isnan(seg_scores[i][block])) { raw += seg_scores[i][block]; }
            }
            score_sum[block] = static_cast<float>(raw);
        }
        scores = merge_mode == 0 ? score_sum : merge_mode == 1 ? score_nsum : score_split;
    }
    const detail::BlockSelection selection =
        detail::select_blocks(sparse.index, scores, config, mandatory);
    if (const char* probe = std::getenv("KV9_PROBE_POS"); probe != nullptr && *probe != '\0') {
        // diagnostics: rank of the block holding known positions (e.g. needles), comma separated
        for (const char* cur = probe; cur != nullptr && *cur != '\0';) {
        char* next = nullptr;
        const std::uint32_t pb = static_cast<std::uint32_t>(std::strtoul(cur, &next, 10)) / kBlockTokens;
        cur = (next != nullptr && *next == ',') ? next + 1 : nullptr;
        if (pb < blocks) {
            const auto rank_of = [&](const std::vector<float>& sc) {
                std::uint32_t rank = 0;
                if (std::isnan(sc[pb])) { return blocks; }
                for (std::uint32_t block = 0; block < blocks; ++block) {
                    if (!std::isnan(sc[block]) && sc[block] > sc[pb]) { ++rank; }
                }
                return rank;
            };
            const bool kept = std::find(selection.selected.begin(), selection.selected.end(), pb) != selection.selected.end();
            std::uint32_t scored = 0;
            for (const std::uint32_t block : selection.selected) {
                if (block < blocks && include[block]) { ++scored; }
            }
            char line[200];
            std::snprintf(line, sizeof(line),
                          " probe block %u rank sum %u / nsum %u / split %u, retrieval slots %u, kept %d;", pb,
                          rank_of(score_sum), rank_of(score_nsum), rank_of(score_split), scored, kept ? 1 : 0);
            seg_log += line;
        }
        }
        // KV9_QSHADOW: the same probe ranks for smaller query sizes (split merge), on the same
        // window, candidates and captured rows.
        for (const auto& [qsize, segs] : sparse.kv9_shadow) {
            std::vector<Kv9RowView> sv;
            for (std::size_t s2 = 0; s2 + 1 < segs.size(); s2 += 2) {
                const std::uint32_t b = segs[s2], e = segs[s2 + 1];
                for (const std::uint32_t k : active) {
                    const std::uint32_t n = sparse.seg_rows_n[k];
                    if (n == 0 || n != sparse.seg_count[k]) { continue; }
                    std::uint32_t B = 0;
                    const std::uint16_t* base = nullptr;
                    std::size_t stride = 0;
                    if (sparse.seg_slot[k] < 0) {
                        if (sparse.seg_rows[k].empty()) { continue; }
                        B = sparse.seg_range[2U * k];
                        base = sparse.seg_rows[k].data();
                        stride = n;
                    } else {
                        const auto slot = static_cast<std::uint32_t>(sparse.seg_slot[k]);
                        if (sparse.slot_row0[slot] == UINT32_MAX || !sparse.q_rows) { continue; }
                        B = sparse.acc_range[2U * slot];
                        base = sparse.q_rows.get() + static_cast<std::size_t>(sparse.slot_row0[slot]) * attention.query_width();
                        stride = sparse.q_rows_cap;
                    }
                    if (b < B || e > B + n || e <= b) { continue; }
                    sv.push_back({base + static_cast<std::size_t>(b - B) * attention.query_width(), e - b, stride});
                    break;
                }
            }
            char line[200];
            if (sv.size() * 2 != segs.size() || sv.empty()) {
                std::snprintf(line, sizeof(line), " shadow q%u: %zu/%zu segs not nested;", qsize, sv.size(), segs.size() / 2);
                seg_log += line;
                continue;
            }
            std::vector<std::vector<float>> out;
            kv9_eq10_scores(sparse.index, sv, attention.num_attention_heads, include, out);
            std::vector<std::uint32_t> best(blocks, blocks);
            std::vector<double> mean(blocks, 0.0);
            std::vector<std::uint32_t> order;
            for (std::size_t j = 0; j < sv.size(); ++j) {
                order.clear();
                double total = 0.0;
                for (std::uint32_t block = 0; block < blocks; ++block) {
                    if (include[block] && sparse.index.block(block).full) { order.push_back(block); total += std::fabs(out[j][block]); }
                }
                std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) { return out[j][x] > out[j][y]; });
                for (std::uint32_t r = 0; r < order.size(); ++r) {
                    best[order[r]] = std::min(best[order[r]], r);
                    if (total > 0.0) { mean[order[r]] += out[j][order[r]] / total / static_cast<double>(sv.size()); }
                }
            }
            std::vector<float> sc(blocks, std::numeric_limits<float>::quiet_NaN());
            for (std::uint32_t block = 0; block < blocks; ++block) {
                if (best[block] >= blocks) { continue; }
                sc[block] = sv.size() == 1U ? static_cast<float>(mean[block])
                                            : static_cast<float>(blocks - best[block]) + 0.5F * static_cast<float>(mean[block]);
            }
            std::snprintf(line, sizeof(line), " shadow q%u (%zu segs):", qsize, sv.size());
            seg_log += line;
            for (const char* cur = probe; cur != nullptr && *cur != '\0';) {
                char* next = nullptr;
                const std::uint32_t pb = static_cast<std::uint32_t>(std::strtoul(cur, &next, 10)) / kBlockTokens;
                cur = (next != nullptr && *next == ',') ? next + 1 : nullptr;
                if (pb >= blocks) { continue; }
                std::uint32_t rank = blocks;
                if (!std::isnan(sc[pb])) {
                    rank = 0;
                    for (std::uint32_t block = 0; block < blocks; ++block) {
                        if (!std::isnan(sc[block]) && sc[block] > sc[pb]) { ++rank; }
                    }
                }
                std::snprintf(line, sizeof(line), " b%u %u", pb, rank);
                seg_log += line;
            }
            seg_log += ";";
        }
    }
    if (sparse.kv9_plan || kv9_enabled()) {
        std::fprintf(stderr,
                     "KVMem kv9 | kvmem_score: SELECT | segs %u%s (stash %u, eq10 %u, %.0f ms) | protect %lld/%lld tok, %zu ranges, "
                     "%u blocks | budget %u blocks, recent %u pages | replay %s | merge %s |%s\n",
                     static_cast<unsigned>(active.size()), sparse.kv9_tool ? " tool" : "", sparse.seg_injected,
                     eq10_segs, score_ms, static_cast<long long>(sparse.protect_used),
                     static_cast<long long>(sparse.protect_cap), sparse.protect_b.size(), selection.mandatory_kept,
                     config.budget_blocks, recent_pages, sparse.replay_planned ? "yes" : "no",
                     merge_mode == 0 ? "sum" : merge_mode == 1 ? "nsum" : "split", seg_log.c_str());
        std::fflush(stderr);
    }
    std::vector<std::uint32_t> pages =
        detail::block_pages(selection.selected, kBlockTokens);
    pages.erase(std::remove_if(pages.begin(), pages.end(),
                               [mapped](std::uint32_t page) { return page >= mapped; }),
                pages.end());
    if (!sparse.media_groups.empty()) {
        // Normalize scored candidates into atomic media groups. Large latest media
        // may consume the usual recent share, but two boundary pages stay reserved.
        std::uint32_t media_budget = config.budget_blocks * 2U;
        for (const auto& group : sparse.media_groups) {
            if (group.begin < 2) { media_budget = std::max(media_budget, group.end); }
        }
        const auto& latest = sparse.media_groups.back();
        std::uint32_t sink_extent = 2;
        for (const auto& group : sparse.media_groups) {
            if (group.begin < 2) { sink_extent = std::max(sink_extent, group.end); }
        }
        media_budget = std::max(media_budget,
            sink_extent + (latest.begin < 2 ? 0 : latest.end - latest.begin));
        pages = media_window_page_set(mapped, media_budget, pages, sparse.media_groups, false);
    }
    sparse.retrieved_pages = pages;
    const std::uint32_t recent_begin = mapped > recent_pages ? mapped - recent_pages : 0U;
    for (std::uint32_t page = recent_begin; page < mapped; ++page) { pages.push_back(page); }
    // Rewind may shorten a Host-backed partial page. Restore its preserved prefix
    // before dropping the probe suffix; later replay writes into that same page.
    if (sparse.query_begin != 0) {
        pages.push_back((sparse.query_begin - 1U) / kPagedKVPageSize);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    if (!sparse.media_groups.empty()) {
        pages = media_window_page_set(mapped, kvmem_window_pages - 1U, pages, sparse.media_groups);
        // This transient page preserves the prefix before truncate_for_replay. No
        // attention runs on this placement; replay normalizes complete groups again.
        if (sparse.query_begin != 0) { pages.push_back((sparse.query_begin - 1U) / kPagedKVPageSize); }
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    }
    const auto placement = text_kv_addresses->apply_device_placement(
        sequence.kv->text, *host_kv_extents, pages, device.transfer_stream, "retrieval");
    if (std::getenv("NINFER_KVMEM_TRACE") != nullptr) {
        std::fprintf(stderr, "KVMEM retrieval scored=%u selected=%zu promoted=%u demoted=%u lane=%u\n",
                     selection.scored_blocks, pages.size(), placement.promoted, placement.demoted,
                     sequence.lane);
    }
    if (sequence.kv->backend && speculative_backend == SpeculativeBackend::Mtp) {
        const auto backend_mapped = backend_kv_addresses->mapped_pages(*sequence.kv->backend);
        pages.erase(std::lower_bound(pages.begin(), pages.end(), backend_mapped), pages.end());
        for (std::uint32_t page = mapped; page < backend_mapped; ++page) { pages.push_back(page); }
        backend_kv_addresses->apply_device_placement(*sequence.kv->backend, *host_kv_extents,
                                                     pages, device.transfer_stream, "retrieval");
    }
}

} // namespace ninfer::models::qwen3_5::detail
