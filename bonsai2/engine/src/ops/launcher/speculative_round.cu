// Implements: include/ninfer/ops/speculative_round.h
// Match: validated speculative state and BF16 verification logits.
// Algorithm assumptions: the shared sampling layout selects either one block
// or a two-launch partial/group pipeline without host reads of device config.
#include "ops/launcher/speculative_round.h"

#include "ops/common/math.h"
#include "ops/kernel/speculative_round.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

// ASD (approximate speculative decoding) knobs, read ONCE per process -- the value decides how the
// captured graph behaves at replay, so it must not change between capture and replay.
//
//   NINFER_SPEC_ASD_BUDGET  > 0 enables ASD; 0 (default) = the shipping strict accept.
//   NINFER_SPEC_ASD_M       per-round cap on accepted disagreements (default 1).
//
// Returns 0 when ASD is off, which makes the kernel take the strict first-divergence loop and
// therefore reproduce the pre-ASD behaviour bit for bit (the rollback floor).
//
// NOT implemented yet: the per-request accumulating ledger `B` and the regret gate `g`. The
// regret needs the draft token's TARGET logit, which the greedy path does not retain -- the
// partial top-k kernel stores only rank 1 when `cap = greedy ? 1 : ...` (see
// speculative_sampling_partial_topk_kernel), so `g` needs one extra scalar per (row, column)
// plumbed through the sampling workspace. Today BUDGET only sets the per-round cap.
std::int32_t asd_max_per_round() {
    static const std::int32_t value = [] {
        const char* budget = std::getenv("NINFER_SPEC_ASD_BUDGET");
        if (budget == nullptr) { return std::int32_t{0}; }
        const std::int32_t b = std::atoi(budget);
        if (b <= 0) { return std::int32_t{0}; }
        const char* cap_env = std::getenv("NINFER_SPEC_ASD_M");
        std::int32_t cap    = cap_env == nullptr ? 1 : std::atoi(cap_env);
        if (cap < 1) { cap = 1; }
        return cap < b ? cap : b;
    }();
    return value;
}

// The regret gate `g`, in NATS: a disagreement may be accepted only when the target's own
// distribution was nearly indifferent there -- logit(argmax) - logit(draft) <= g, i.e. the draft
// token kept at least exp(-g) of the argmax probability AT THAT COLUMN. It reads the verify logits
// we already hold (one bf16 load per decision, no extra kernel, no extra pass).
//
// Why it is not optional: with only the "recall" condition (a later column still agrees), ANY
// disagreement is admissible with no bound on its size. Measured at 32768 on 2026-09-20:
// 19/20 -> 16/20 (3 new wrong answers) and +51% generated tokens / +34% wall clock on reasoning
// prompts -- a real deviation sends the model down a different and longer path. 0 (default, unset)
// = no regret gate = the v1 behaviour (kept so v1 can be reproduced and compared).
float asd_regret_gate() {
    static const float value = [] {
        const char* raw = std::getenv("NINFER_SPEC_ASD_G");
        if (raw == nullptr) { return 0.0f; }
        const float g = static_cast<float>(std::atof(raw));
        return g > 0.0f ? g : 0.0f;
    }();
    return value;
}

} // namespace

void speculative_prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                              const Tensor& base_positions,
                                              const Tensor& current_extents, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(base_positions.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), static_cast<std::int32_t*>(positions.data), k);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                                           const Tensor& current_extents, Tensor& verify_ids,
                                           cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), nullptr, k);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, const Tensor& current_extents,
                                             Tensor& lengths, Tensor& anchors,
                                             Tensor& licensed_tokens, Tensor& licensed_counts,
                                             Tensor& accepted, std::int32_t token_domain,
                                             const SamplingConfig* configs, DeviceSpan workspace,
                                             cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t cols              = drafts.ne[0] + 1;
    const std::int32_t batch             = drafts.ne[1];
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    if (!layout.multiblock) {
        speculative_accept_greedy_drafts_kernel<<<batch, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, physical_rows,
            drafts.ne[0]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
        physical_rows, cols, drafts.ne[0], scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
    const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                  static_cast<unsigned int>(cols),
                                  static_cast<unsigned int>(batch));
    speculative_sampling_group_finalize_kernel<false>
        <<<batched_group_grid, kSamplerGroupBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const std::int32_t*>(drafts.data), nullptr, nullptr,
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, cols, partial_blocks,
            groups, scratch, layout.bytes, asd_max_per_round(),
            static_cast<const __nv_bfloat16*>(logits.data), physical_rows, asd_regret_gate());
    CUDA_CHECK(cudaGetLastError());
}

void speculative_accept_sparse_drafts_launch(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
    Tensor& round_lengths, Tensor& round_anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted_drafts, std::int32_t token_domain, const SamplingConfig* configs,
    bool raw_greedy, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t batch = drafts.ne[1];
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t cols  = k + 1;
    if (raw_greedy) {
        speculative_accept_sparse_warp_greedy_kernel<<<1, 32 * batch, 0, stream>>>(
            static_cast<const int*>(target_tokens.data), static_cast<const int*>(drafts.data),
            static_cast<const int*>(current_extents.data), static_cast<int*>(round_lengths.data),
            static_cast<int*>(round_anchors.data), static_cast<int*>(licensed_tokens.data),
            static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted_drafts.data), k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    const std::int32_t partial_blocks    = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups            = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch      = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(cols), static_cast<unsigned int>(batch));
    speculative_sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain, logits.ne[0],
        cols, k, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());

    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols),
                          static_cast<unsigned int>(batch));

    speculative_sampling_group_finalize_kernel<true><<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(proposal_q.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(round_lengths.data),
        static_cast<std::int32_t*>(round_anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data),
        static_cast<std::int32_t*>(accepted_drafts.data), configs, token_domain, cols,
        partial_blocks, groups, scratch, layout.bytes, asd_max_per_round(),
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], asd_regret_gate());

    CUDA_CHECK(cudaGetLastError());
}

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& selectors,
                                               Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int batch      = hidden.ne[2];
    const dim3 grid(static_cast<unsigned int>(std::max(1, div_up(rows, kBlock))),
                    static_cast<unsigned int>(batch));
    speculative_select_accepted_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(selectors.data), static_cast<__nv_bfloat16*>(out.data),
        rows, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = proposal_tokens.ne[0];
    const int grid       = std::max(1, div_up(count, kBlock));
    proposal_remap_token_ids_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<std::int32_t*>(proposal_tokens.data), count, id_map, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
