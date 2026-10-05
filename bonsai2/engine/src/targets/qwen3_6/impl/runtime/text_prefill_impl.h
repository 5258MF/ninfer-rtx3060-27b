#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ops/kvmem/kvmem_shadow.h"
#include "ops/kvmem/mean_k_index.h"
#include "ops/kvmem/raw_k_harvest.h"
#include "ops/kvmem/kvmem_score.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
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
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            (void)rewrite_checkpoint;
        });
}

} // namespace

// KVMem raw-K harvest: installation and the per-chunk drive.
//
// WHY THE HARVEST OUTLIVES THE CARD
//   TextContext is constructed on the stack for one schedule recording and is destroyed again
//   (text_context.h:29), while the harvest owns a device arena and a pinned host allocation that
//   must NOT be recycled per call -- so it outlives every TextContext and is handed to each one by
//   pointer (text_context.h:338-343). The accessor lives in the op (raw_k_harvest.h) because SIX
//   card construction sites need the SAME object; while it was file-local here, the other five
//   could not reach it and the whole harvest was dead code.
//
// WHY IT IS SIZED FROM prefill_chunk
//   attn_mix() asserts T <= prefill_chunk_ on every call (text_context_impl.h:455 and :593 are the
//   same guard on the two prefill entry points), so `prefill_chunk` is a PROVEN upper bound on the
//   width any staging slot must accept. Sizing from it is therefore safe, and sizing from
//   kCausalAttentionMaximumVisibleKeys instead would reserve 16 x 512 MiB = 8 GiB.
//
// INSTALLED WHENEVER THE SWITCH IS ON -- NOT ONLY WHEN DUMPING
//   Installing the harvest is what lets a later stage consume the raw K, so it must not be tied to
//   the L3 dump. The accessor returns nullptr when NINFER_TERNARY_KVMEM is off (the default), and
//   the mixer treats a null harvest as "nothing to copy into", so the OFF arm still allocates
//   nothing and stays bit-for-bit identical to the engine without this rung.
namespace {

// Opens a fresh round so no slot can be mistaken for one written by an earlier chunk. Returns the
// installed harvest, or nullptr when the switch is off.
ops::RawKShadowHarvest* begin_raw_k_shadow_round(std::uint32_t prefill_chunk) {
    ops::RawKShadowHarvest* harvest =
        ops::raw_k_shadow_harvest_for(prefill_chunk, static_cast<std::int32_t>(TextConfig::kv_size),
                                      static_cast<std::int32_t>(
                                          TextConfig::full_attention_layers()));
    if (harvest != nullptr) { harvest->begin_round(); }
    return harvest;
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    // KVMem 装配臂（prefill 期可见集有界化，标记 prefill-window/route-B）：把本 chunk 的**有效
    // 前沿**交给 card，供它算 attention envelope 上界。program 侧给的是
    // `sequence.kvmem_window.frontier(text_kv_base)`（未 arm 时 == text_kv_base ⇒ 恒等）。
    // 不设该字段时 card 的 has_ 保持 false ⇒ envelope 与改前逐位一致（OFF 臂不受影响）。
    // 文本入口与多模态入口**都要**这一句：两者各自建 card、各自走 prefill attention。
    if (state.has_kvmem_window_frontier) {
        card.set_kvmem_window_frontier(state.kvmem_window_frontier);
    }
    if (state.kvmem_cache_shift != 0) { card.set_kvmem_cache_shift(state.kvmem_cache_shift); }
    if (state.has_kvmem_rope_delta) { card.set_kvmem_rope_delta(state.kvmem_rope_delta); }

    // Install the harvest whenever the switch is on, and start a fresh round: installing is what
    // lets a later stage consume the raw K, while the dump is only one possible consumer of it.
    ops::RawKShadowHarvest* harvest =
        state.kvmem_replay ? nullptr : begin_raw_k_shadow_round(state.execution.prefill_chunk);
    card.set_raw_k_shadow_harvest(harvest);
    // 装配臂：记下这一轮 raw-K 覆盖的首个绝对位置（重烘要按块定位，见 raw_k_harvest.h）。
    // 只写 harvest 自己的两个标量，不碰任何张量 ⇒ 不改变 OFF 臂的数值。
    if (harvest != nullptr) {
        harvest->set_round_origin(static_cast<std::int32_t>(state.text_kv_base));
    }

    // ---- KVMem P1d: a sequence that starts again at position 0 gets a clean mean-K index ----------
    if (harvest != nullptr && state.text_kv_base == 0) {
        ops::detail::kvmem_score_forget_carry("fresh sequence");   // KVMem P3a-h
        ops::MeanKIndex* fresh_index = ops::mean_k_index_for(
            static_cast<std::int32_t>(TextConfig::full_attention_layers()),
            static_cast<std::int32_t>(TextConfig::kv_heads),
            static_cast<std::int32_t>(TextConfig::head_dim),
            static_cast<std::int32_t>(state.text_kv.max_context() / kPagedKVPageSize));
        if (fresh_index != nullptr && fresh_index->blocks_written() > 0) {
            const std::int32_t stale = fresh_index->blocks_written();
            const cudaError_t zst = ops::kvmem_meank_stash_and_zero(*fresh_index, state.execution.device.stream);  // KVMem P3a-g
            std::fprintf(stderr, "[kvmem-p1d] mean-K index reset for a fresh sequence (stale_blocks=%d%s)\n",
                         stale, zst == cudaSuccess ? "" : " ZERO FAILED");
        }
    }

    // ---- KVMem 选择探针：本 chunk 开始（最小切片，默认关）-----------------------------------
    // (KVMem P1e: never during a query replay.)
    // 打分要对着**历史**：本 chunk 自己还没进索引（索引在本函数尾部才 append_round），所以此刻的
    // blocks_written() 天然就是"历史块数"——与官方实现"用当前 query 打分历史块"一致。
    // 块大小按 house rule **传进来**（不许写死；kPagedKVPageSize = 64）。
    if (ops::detail::kvmem_score_enabled() && !state.kvmem_replay) {
        ops::MeanKIndex* score_index = ops::mean_k_index_for(
            static_cast<std::int32_t>(TextConfig::full_attention_layers()),
            static_cast<std::int32_t>(TextConfig::kv_heads),
            static_cast<std::int32_t>(TextConfig::head_dim),
            static_cast<std::int32_t>(state.text_kv.max_context() / kPagedKVPageSize));
        if (score_index != nullptr) {
            ops::detail::kvmem_score_begin(
                static_cast<std::int32_t>(state.text_kv.max_context() / kPagedKVPageSize),
                score_index->blocks_written(), score_index->tail_fill(),
                static_cast<std::int32_t>(kPagedKVPageSize),
                static_cast<std::int32_t>(TextConfig::full_attention_layers()),
                static_cast<std::int32_t>(TextConfig::query_heads),
                static_cast<std::int32_t>(TextConfig::kv_heads),
                static_cast<std::int32_t>(TextConfig::head_dim),
                state.execution.device.stream);
            // KVMem P4 (kv8): msg-mode query plan / stash bookkeeping (no-op unless enabled).
            static_assert(sizeof(TokenId) == sizeof(std::int32_t), "KVMem P4 reads prompt ids as int32");
            ops::detail::kvmem_q4_begin_chunk(
                reinterpret_cast<const std::int32_t*>(ids.data()), static_cast<std::int64_t>(ids.size()),
                static_cast<std::int32_t>(state.text_kv_base), score_index->blocks_written(),
                static_cast<std::int32_t>(state.text_kv.max_context() / kPagedKVPageSize),
                static_cast<std::int32_t>(TextConfig::full_attention_layers()),
                static_cast<std::int32_t>(TextConfig::query_heads),
                static_cast<std::int32_t>(TextConfig::head_dim), state.execution.device.stream);
        }
    }
    const bool dump_armed =
        harvest != nullptr && ops::detail::kvmem_shadow_dump_path() != nullptr;

    const std::span<const int> prompt(ids.data(), ids.size());
    PrefillChunkResult result;
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        result = card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                    sink);
    } else {
        result = card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
    }

    // ---- KVMem mean-K index: append THIS chunk, while the raw K is still in hand ----------------
    //
    // The whole point of the index is that it describes history the paged KV can no longer be
    // scanned for (mean_k_index.h:19-25), so it must be built here -- per chunk, on the same stream,
    // from the object the harvest just filled -- and never reconstructed from the cache afterwards.
    //
    // WHAT IS CONSUMED: staging_for_round(), i.e. the harvest's DEVICE slots, straight into the
    // index's device arena. Same stream, so this is a device-to-device accumulate with no host round
    // trip and NO drain_all() -- draining first would copy the whole chunk to pinned memory and
    // synchronise, which is the prefill-throughput mistake the harvest exists to avoid.
    //
    // WHY NO DE-ROPE: the harvest holds the PRE-RoPE key (kvmem_shadow.h:29), and the index is
    // defined over exactly that content frame (mean_k_index.h:9). The official executor captures
    // from the POST-RoPE batch instead and therefore has to de-RoPE it back
    // (qwen_executor.cpp:19004-19007 -- "Archived raw K is stored position-free, so the builders
    // must not de-RoPE it again"). Our design removes that step rather than performing it.
    //
    // FIRST BLOCK: this chunk's first token lands at state.text_kv_base, and a logical block IS one
    // KV page, so the block number is that position divided by the page size. Deriving it from the
    // same base the chunk was given is what keeps the index aligned with the sequence after a
    // re-prefill or a prefix reuse; a running counter here would silently drift.
    if (harvest != nullptr) {
        // ★ 装配臂（路 1）：把这一轮的**完整块**的 raw-K 留存下来。arena 开关关时它是一句 return：
        // 不分配、不拷贝 ⇒ OFF 臂逐位不变。放在这里与 mean-K append 同一个位置、同一条 stream，
        // 因为此刻 staging 活着、这一轮的起点已知、且在任何 CUDA graph 捕获之外（捕获里不能同步、
        // 不能分配）。半块不收（staging 里没有其余 token 的字节，收了就是错料）。
        ops::detail::raw_k_block_arena_store_round(*harvest, static_cast<std::int32_t>(kPagedKVPageSize),
                                           static_cast<std::int32_t>(state.text_kv.max_context()),
                                           state.execution.device.stream);
        ops::MeanKIndex* kvmem_index = ops::mean_k_index_for(
            static_cast<std::int32_t>(TextConfig::full_attention_layers()),
            static_cast<std::int32_t>(TextConfig::kv_heads),
            static_cast<std::int32_t>(TextConfig::head_dim),
            static_cast<std::int32_t>(state.text_kv.max_context() / kPagedKVPageSize));
        if (kvmem_index != nullptr) {
            const std::int32_t round_first_block =
                static_cast<std::int32_t>(state.text_kv_base / kPagedKVPageSize);
            // Retrieval observes only the pre-chunk history. Host scoring must
            // finish before this append can mutate its partial tail block.
            // ---- KVMem 选择探针：本 chunk 收尾（D2H + 选块 + 一行证据日志）--------------------
            //
            // 放在这里与下面那两段既有探针同一个理由：chunk 尾巴、任何 CUDA graph 捕获之外，
            // 用**一次同步**换一个外部可核的数。本探针不改可见集 ⇒ 只回答"选择器选得对不对"。
            if (ops::detail::kvmem_score_enabled()) {
                ops::detail::kvmem_score_finish("prefill_chunk",
                                                static_cast<std::int32_t>(ids.size()),
                                                state.execution.device.stream);
                ops::detail::kvmem_score_dump_if_requested("prefill_chunk");
            }

            kvmem_index->append_round_at(*harvest, static_cast<std::int32_t>(state.text_kv_base),
                                         state.execution.device.stream);   // kvmem_kv6_aligned_append

            // ---- Self-proof that the index is really being FED, not merely that it compiled -------
            //
            // The criterion for wiring this rung is "layer_counts[b] != 0", and it has to be produced
            // here: the index is a process-lifetime singleton owned by the op, so this driver is the
            // only place that can see it. Reading it back needs the bytes on the host, so this block is
            //   * bounded to the four blocks a prefill chunk can have appended LAST (all of them in one
            //     copy, and none per layer) -- a per-layer readback would sync the stream 16 times on
            //     every prefill chunk for no extra information;
            //   * issued as an INDEPENDENT cudaMemcpyAsync rather than by reading the index's own
            //     tensors, so the diagnostic cannot perturb the accumulate path it is measuring;
            //   * synchronised once, OUTSIDE any capture, at the tail of a prefill chunk -- the same
            //     place the L3 dump below already syncs.
            //
            // WHY FOUR AND NOT ONE: a single block's count cannot distinguish "the index holds this
            // chunk" from "it holds an earlier one". With four consecutive counts in hand, the sum
            // over blocks is a number an external oracle can check against the prompt length -- and it
            // is what caught the first version of this probe reporting a warmup block's 53 as if it
            // were the request's data.
            constexpr std::int32_t kProbeLayer   = 0;
            constexpr std::int32_t kProbeBlocks  = 4;
            if (kvmem_index->blocks_written() > 0) {
                const Tensor& counts      = kvmem_index->layer_counts(kProbeLayer);
                const cudaStream_t stream = state.execution.device.stream;
                const std::int32_t total  = kvmem_index->blocks_written();
                const std::int32_t span   = std::min(kProbeBlocks, total);
                const std::int32_t from   = total - span;
                if (counts.data != nullptr && from >= 0) {
                    float block_counts[kProbeBlocks] = {0.0F, 0.0F, 0.0F, 0.0F};
                    const cudaError_t read_status = kvmem_index->read_counts(
                        kProbeLayer, from, span, block_counts, stream);
                    if (read_status == cudaSuccess) {
                        const cudaError_t sync_status = cudaStreamSynchronize(stream);
                        if (sync_status == cudaSuccess) {
                            double prior_sum = 0.0;
                            for (std::int32_t i = 0; i < span; ++i) {
                                prior_sum += static_cast<double>(block_counts[i]);
                            }
                            std::fprintf(stderr,
                                         "kvmem_index: APPENDED layers=%d blocks_written=%d "
                                         "tail_fill=%d blocks=[%d..%d) counts=[%.0f,%.0f,%.0f,%.0f] "
                                         "tail_sum=%.0f\n",
                                         static_cast<int>(kvmem_index->layers()),
                                         static_cast<int>(total),
                                         static_cast<int>(kvmem_index->tail_fill()),
                                         static_cast<int>(from), static_cast<int>(total),
                                         static_cast<double>(block_counts[0]),
                                         static_cast<double>(block_counts[1]),
                                         static_cast<double>(block_counts[2]),
                                         static_cast<double>(block_counts[3]), prior_sum);

                            // ---- ORACLE SNAPSHOT (the reason this diagnostic exists at all) --------
                            //
                            // Everything above proves the index is FED. None of it proves the NUMBERS
                            // are right, and "fed with the wrong values" is the failure that would
                            // survive every check so far: the counts would still be plausible, the
                            // cursor would still advance, and the retrieval ranking would just be
                            // quietly wrong.
                            //
                            // The values can only be checked against an INDEPENDENT source, and we
                            // already have one: the L3 dump written a few lines below holds this very
                            // chunk's PRE-RoPE K, which is exactly what the index accumulates
                            // (raw_k_harvest.h:131-143 for the blob layout). What is missing is a
                            // way to line the two up, because the index ACCUMULATES across chunks
                            // while the dump covers one -- so a diff would compare different sets of
                            // tokens (the "two sides reading different inputs" defect).
                            //
                            // So the index is rebuilt from scratch out of the SAME round that the
                            // dump is about to capture: zero() clears the sums, the counts AND the
                            // cursor, and the re-append puts this round back at its own first_block.
                            // After this the index holds exactly this chunk, and the dump holds
                            // exactly this chunk -- one set of tokens, two independent computations.
                            //
                            // Safe because: the switch is on, the stream is not capturing, and this
                            // runs once per chunk at the tail of a prefill where the L3 dump below
                            // already synchronises. When the dump path is off nothing is captured and
                            // the index keeps its accumulated history (the reset is worth nothing
                            // without the blob, so it is gated on the blob).
                            if (ops::detail::kvmem_shadow_dump_path() != nullptr) {
                                const cudaError_t zero_status = kvmem_index->zero(stream);
                                if (zero_status == cudaSuccess) {
                                    kvmem_index->append_round_at(   // kvmem_kv6_aligned_append
                                        *harvest, static_cast<std::int32_t>(state.text_kv_base), stream);
                                    std::fprintf(stderr,
                                                 "kvmem_index: ORACLE SNAPSHOT layer=%d "
                                                 "first_block=%d blocks=%d tail_fill=%d "
                                                 "(index rebuilt from this round; compare against the "
                                                 "L3 dump)\n",
                                                 static_cast<int>(kProbeLayer),
                                                 static_cast<int>(round_first_block),
                                                 static_cast<int>(kvmem_index->blocks_written()),
                                                 static_cast<int>(kvmem_index->tail_fill()));

                                    // Export layer 0's SUMS so an external oracle can check the values.
                                    // The L3 dump carries the raw PRE-RoPE K of this same round, so the
                                    // oracle recomputes the mean from that blob and compares it here --
                                    // the engine's own numbers on one side, an independent derivation
                                    // on the other. Without this the only checkable quantity would be
                                    // the count, which the append kernel could get right while the
                                    // sums were entirely wrong.
                                    //
                                    // Bounded to ONE layer and to the blocks this round produced: 4
                                    // heads x 256 dims x 4 B = 4 KiB per block, so a chunk of 512
                                    // tokens exports 32 KiB. A failure here is reported and ignored,
                                    // because a diagnostic must never take the engine down.
                                    const Tensor& sums_out = kvmem_index->layer_sums(kProbeLayer);
                                    const std::int32_t have_blocks = kvmem_index->blocks_written();
                                    const std::int32_t row_elems   = TextConfig::kv_heads * TextConfig::head_dim;
                                    if (sums_out.data != nullptr && have_blocks > 0 && row_elems > 0) {
                                        const std::size_t want = static_cast<std::size_t>(have_blocks) *
                                                                 static_cast<std::size_t>(row_elems);
                                        std::vector<float> host_sums(want);
                                        const cudaError_t sums_status = cudaMemcpyAsync(
                                            host_sums.data(), sums_out.data, want * sizeof(float),
                                            cudaMemcpyDeviceToHost, stream);
                                        if (sums_status == cudaSuccess) {
                                            const cudaError_t sums_sync = cudaStreamSynchronize(stream);
                                            if (sums_sync == cudaSuccess) {
                                                // The file carries the identity of the round that produced it
                                                // (first_block, round width) so the oracle can confirm
                                                // mechanically that BOTH sides describe the same set of tokens
                                                // instead of assuming it. The last dump on disk is whatever
                                                // dump_to_file wrote most recently, and this diagnostic runs
                                                // more than once per process -- so without this the oracle
                                                // would silently compare two different rounds, which is the
                                                // "two sides reading different inputs" defect.
                                                const char* path =
                                                    "E:\\infer-build\\exp\\kvmem-index-oracle\\engine_layer0_sums.bin";
                                                std::FILE* out = std::fopen(path, "wb");
                                                if (out != nullptr) {
                                                    // Also export the first few elements and this block's COUNT.
                                                    // Purpose: when the oracle and the engine disagree, the
                                                    // single worst element says nothing about WHY, whereas
                                                    // eight consecutive engine values next to eight oracle
                                                    // values plus the token count localise the defect
                                                    // immediately -- a scaling, an offset, or a permutation.
                                                    // Human-readable on purpose: this is the line a person
                                                    // reads when the numeric diff is red.
                                                    std::fprintf(stderr,
                                                                 "kvmem_index: ORACLE ELEMS[0..8)=[");
                                                    for (std::int32_t i = 0; i < 8 && i < row_elems; ++i) {
                                                        std::fprintf(stderr, "%s%.6f", i ? "," : "",
                                                                     static_cast<double>(host_sums[
                                                                         static_cast<std::size_t>(i)]));
                                                    }
                                                    std::fprintf(stderr, "] blocks=%d row_elems=%d\n",
                                                                 static_cast<int>(have_blocks),
                                                                 static_cast<int>(row_elems));
                                                    const std::int32_t header[4] = {
                                                        round_first_block, have_blocks, row_elems,
                                                        harvest->round_width()};
                                                    (void)std::fwrite(header, sizeof(std::int32_t), 4, out);
                                                    (void)std::fwrite(host_sums.data(), sizeof(float),
                                                                      want, out);
                                                    std::fclose(out);
                                                    std::fprintf(stderr,
                                                                 "kvmem_index: ORACLE EXPORT %s "
                                                                 "first_block=%d blocks=%d row_elems=%d "
                                                                 "round_width=%d floats=%zu\n",
                                                                 path,
                                                                 static_cast<int>(round_first_block),
                                                                 static_cast<int>(have_blocks),
                                                                 static_cast<int>(row_elems),
                                                                 static_cast<int>(harvest->round_width()),
                                                                 want);

                                                    // ---- The K side, written HERE ------------------
                                                    //
                                                    // The engine's own L3 dump CANNOT be the other side of
                                                    // this comparison. Two measured reasons:
                                                    //   1. it is one path overwritten by every round, so what
                                                    //      is on disk at the end belongs to whichever round
                                                    //      wrote last (observed: round_width=4, a decode
                                                    //      step, while the export was a 486-token chunk);
                                                    //   2. it stages the FULL `tokens` slots while a round
                                                    //      writes only `round_width` of them, so an oracle
                                                    //      reading it would average unwritten slots together
                                                    //      with real ones.
                                                    // Either alone makes a numeric diff meaningless, which
                                                    // is the "two sides reading different inputs" defect.
                                                    //
                                                    // So this diagnostic writes BOTH sides, from the same
                                                    // harvest, in the same call: layer 0's pre-RoPE K, raw
                                                    // BF16, EXACTLY round_width tokens and no padding slots.
                                                    // Layout: int32[3] = {round_width, row_elems, layer},
                                                    // then the payload, one contiguous row per token.
                                                    const Tensor kview =
                                                        harvest->staging_for_round(kProbeLayer);
                                                    const std::int32_t k_width = harvest->round_width();
                                                    const std::size_t k_bytes =
                                                        static_cast<std::size_t>(k_width) *
                                                        static_cast<std::size_t>(row_elems) * 2U;
                                                    if (k_width > 0 && kview.data != nullptr) {
                                                        std::vector<unsigned char> kbytes(k_bytes);
                                                        const cudaError_t kcopy = cudaMemcpyAsync(
                                                            kbytes.data(), kview.data, k_bytes,
                                                            cudaMemcpyDeviceToHost, stream);
                                                        if (kcopy == cudaSuccess &&
                                                            cudaStreamSynchronize(stream) == cudaSuccess) {
                                                            const char* kpath =
                                                                "E:\\infer-build\\exp\\kvmem-index-oracle\\k0.bin";
                                                            std::FILE* kout = std::fopen(kpath, "wb");
                                                            if (kout != nullptr) {
                                                                const std::int32_t khdr[3] = {
                                                                    k_width, row_elems, kProbeLayer};
                                                                (void)std::fwrite(khdr, sizeof(std::int32_t),
                                                                                  3, kout);
                                                                (void)std::fwrite(kbytes.data(), 1,
                                                                                  k_bytes, kout);
                                                                std::fclose(kout);
                                                                std::fprintf(stderr,
                                                                             "kvmem_index: ORACLE K %s "
                                                                             "round_width=%d row_elems=%d "
                                                                             "bytes=%zu\n",
                                                                             kpath,
                                                                             static_cast<int>(k_width),
                                                                             static_cast<int>(row_elems),
                                                                             k_bytes);
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    std::fprintf(stderr,
                                                                 "kvmem_index: ORACLE EXPORT cannot open %s\n",
                                                                 path);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    std::fprintf(stderr, "kvmem_index: ORACLE SNAPSHOT zero failed %s\n",
                                                 cudaGetErrorName(zero_status));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write the L3 blob AFTER the chunk, when the harvest holds both halves: attn_mix drained the
    // pre-RoPE values and then refilled every slot with the post-rope key it copied after ops::rope.
    //
    // The post-RoPE half is drained to its own PINNED HOST mirror first, and the dump then reads both
    // halves from host memory. Writing straight from the device pointer inside std::fwrite faulted
    // with c0000005 inside ucrtbase.dll on every attempt (4/4 measured), so the writer is never
    // given device memory to read.
    if (dump_armed && harvest != nullptr) {
        (void)harvest->drain_post_rope(state.execution.device.stream);
        const std::int32_t width = static_cast<std::int32_t>(ids.size());
        std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
        for (std::int32_t i = 0; i < width; ++i) {
            positions[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>(state.text_kv_base) + i;
        }
        (void)harvest->dump_to_file(ops::detail::kvmem_shadow_dump_path(),
                                    harvest->post_rope_host_layer(0),
                                    harvest->staging_bytes_per_layer(), positions.data(),
                                    TextConfig::rotary_dim);
        std::fprintf(stderr, "raw_k_shadow: round harvested %d of %d layers\n",
                     harvest->layers_harvested(),
                     static_cast<int>(TextConfig::full_attention_layers()));
    }
    return result;
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession& vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    // KVMem 装配臂（prefill 期可见集有界化，标记 prefill-window/route-B）：把本 chunk 的**有效
    // 前沿**交给 card，供它算 attention envelope 上界。program 侧给的是
    // `sequence.kvmem_window.frontier(text_kv_base)`（未 arm 时 == text_kv_base ⇒ 恒等）。
    // 不设该字段时 card 的 has_ 保持 false ⇒ envelope 与改前逐位一致（OFF 臂不受影响）。
    // 文本入口与多模态入口**都要**这一句：两者各自建 card、各自走 prefill attention。
    if (state.has_kvmem_window_frontier) {
        card.set_kvmem_window_frontier(state.kvmem_window_frontier);
    }
    if (state.kvmem_cache_shift != 0) { card.set_kvmem_cache_shift(state.kvmem_cache_shift); }
    // Same installation as prefill_text_chunk: this entry point builds its own card and would
    // otherwise silently harvest nothing (the failure mode raw_k_harvest.h documents).
    ops::RawKShadowHarvest* multimodal_harvest =
        begin_raw_k_shadow_round(state.execution.prefill_chunk);
    card.set_raw_k_shadow_harvest(multimodal_harvest);
    if (multimodal_harvest != nullptr) {
        multimodal_harvest->set_round_origin(static_cast<std::int32_t>(state.text_kv_base));
    }
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
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    // The workspace overload: the folded (rotated-basis) ternary output head needs a [hidden, 1]
    // activation rotation buffer, and this arena is the only one in scope here.
    ops::linear(hidden, state.execution.model.output_head, logits, ops::LinearPolicy::A16Only,
                state.execution.work, state.execution.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token, TextConfig::token_domain, state.sampling,
                state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
