#include "ops/kvmem/kvmem_select.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>

// See kvmem_select.h for the policy this implements and for why it is host-only.

namespace ninfer::ops {
namespace {

constexpr const char* kSubject = "kvmem_select";

// Descending score, and on ties the HIGHER block id first -- i.e. the NEWER block wins.
//
// The tie DIRECTION is not a taste decision: it is the reference's (`kvmem_store.cpp:403-408`). It
// matters because equal scores are a REACHABLE state, not a corner case -- the reference degrades
// Retrieval -> H2O -> Recency when the query embedding is flat or absent (`kvmem_store.hpp:140-143`),
// and the first re-selection after a prefill has no query at all. With the direction backwards we
// would keep the OLDEST candidates and evict the NEWEST, which is the opposite of what the recency
// fallback exists for.
//
// The order is TOTAL either way: block ids are unique, so no two candidates compare equal and the
// selected SET cannot depend on the order std::nth_element happened to shuffle equal elements into.
// (An earlier revision of this comment claimed only an id-ASCENDING rule is deterministic. That
// argument was wrong -- id-descending is just as total. Determinism comes from totality, not from its
// direction.)
bool score_before(std::int32_t a, std::int32_t b, const float* scores, std::int32_t first_block) {
    const float sa = scores[a - first_block];
    const float sb = scores[b - first_block];
    if (sa > sb) { return true; }
    if (sa < sb) { return false; }
    return a > b;
}

// Guard: the scorer writes "no signal" as NaN or infinity rather than a very negative number, so a
// comparison that let those through would let a block with no score outrank every real candidate.
// Treating them as the lowest possible score is the only choice that cannot starve the budget.
bool finite_score(const float* scores, std::int32_t block, std::int32_t first_block) {
    return std::isfinite(scores[block - first_block]);
}

} // namespace

KvMemSelectResult kvmem_select_blocks(const KvMemSelectConfig& config, const float* scores,
                                      std::int32_t first_block) {
    if (config.total_blocks < 0 || config.budget_blocks < 0 || config.sink_blocks < 0 ||
        config.recent_blocks < 0 || first_block < 0) {
        throw std::invalid_argument(std::string(kSubject) + ": negative count");
    }
    if (config.total_blocks > 0 && first_block >= config.total_blocks) {
        throw std::invalid_argument(std::string(kSubject) + ": first_block is past the history");
    }
    if (config.budget_blocks > 0 && scores == nullptr) {
        throw std::invalid_argument(std::string(kSubject) + ": scores is null with a nonzero budget");
    }

    KvMemSelectResult out;
    if (config.total_blocks == 0) { return out; }
    // A zero budget means UNLIMITED, not "keep nothing" (reference: `kvmem_store.cpp:362` treats
    // `budget == 0 || n <= budget` as "select everything", ignoring both the scores and the sink /
    // recent regions). That is reachable in practice, not hypothetical: the caller derives the budget
    // as `select_budget / block_tokens` with INTEGER division (`kvmem_store.hpp:378-380`), so any
    // select_budget below one block truncates to 0. Returning nothing there would silently empty the
    // window on a configuration the reference answers with the entire history.
    if (config.budget_blocks == 0) {
        const std::int32_t total = config.total_blocks;
        out.kept.resize(static_cast<std::size_t>(total));
        std::iota(out.kept.begin(), out.kept.end(), 0);
        // The counters still add up to the emitted list, so the "list length == counters sum"
        // invariant holds on EVERY path. Note the third counter is named `scored_kept` while in this
        // branch nothing was scored: everything fits, so the remainder is filled without consulting
        // `scores` at all.
        out.sink_kept   = std::min(config.sink_blocks, total);
        out.recent_kept = std::min(config.recent_blocks, total - out.sink_kept);
        out.scored_kept = total - out.sink_kept - out.recent_kept;
        out.candidates  = out.scored_kept;
        return out;
    }

    // ---- 1. Always-kept regions, clamped so they can never overlap and never exceed the history.
    //
    // Sink is clamped first and recent takes what is left: a configuration whose regions together
    // exceed the history must still produce a valid, duplicate-free set, and giving the prefix
    // priority is what preserves the stable-prefix anchor the sink exists to be.
    std::int32_t sink = std::min(config.sink_blocks, config.total_blocks);
    sink              = std::min(sink, config.budget_blocks);
    std::int32_t recent = std::min(config.recent_blocks, config.total_blocks - sink);
    recent              = std::min(recent, config.budget_blocks - sink);

    std::vector<std::int32_t> kept;
    kept.reserve(static_cast<std::size_t>(config.budget_blocks));
    for (std::int32_t b = 0; b < sink; ++b) { kept.push_back(b); }
    out.sink_kept = sink;
    for (std::int32_t i = 0; i < recent; ++i) {
        kept.push_back(config.total_blocks - 1 - i);
    }
    out.recent_kept = recent;

    const std::int32_t middle_begin = sink;
    const std::int32_t middle_end   = config.total_blocks - recent;  // exclusive
    const std::int32_t remaining    = config.budget_blocks - sink - recent;
    if (remaining <= 0 || middle_end <= middle_begin) {
        // ---- 1b. Nothing scored fits: the always-kept regions ARE the answer.
        //
        // THIS BRANCH MUST HAND `kept` TO `out`. It used to `return out;` with `out.kept` never
        // assigned -- only the scored path at the bottom assigned it -- so whenever sink + recent
        // consumed the whole budget, or the middle region was empty, the function returned an EMPTY
        // window while `out.sink_kept` / `out.recent_kept` still reported those blocks as kept. The
        // caller cannot see the contradiction: an empty window is not "nothing to do", it is a
        // silently wrong answer, and the counters actively argue the blocks are there.
        //
        // Caught by cross-checking against the reference implementation's own test suite (3675 cases,
        // `E:\infer-build\exp\kvmem-select-xcheck\`): the reference returns [0..budget-1] for
        // `total_blocks=40, budget_blocks=8, sink=8, recent=0`, we returned []. Our own oracle could not
        // see it -- it was a Python mirror of this very logic AND asserted only structural invariants
        // (<= budget, ordered, unique, in range), every one of which an empty list satisfies happily.
        //
        // The sort+unique mirror the scored path below, so the emitted list stays ascending and
        // duplicate-free when the two always-kept regions overlap (sink + recent > total_blocks).
        std::sort(kept.begin(), kept.end());
        kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
        out.kept = std::move(kept);
        return out;
    }

    // ---- 2. Middle blocks by score.
    //
    // Only blocks at or after first_block have a score. Anything earlier cannot be scored by this
    // caller and is left to the always-kept regions rather than being selected on a stale or absent
    // score.
    const std::int32_t cand_begin = std::max(middle_begin, first_block);
    const std::int32_t cand_end   = middle_end;
    if (cand_end > cand_begin) {
        std::vector<std::int32_t> cand(static_cast<std::size_t>(cand_end - cand_begin));
        std::iota(cand.begin(), cand.end(), cand_begin);
        out.candidates = static_cast<std::int32_t>(cand.size());

        const auto before = [scores, first_block](std::int32_t a, std::int32_t b) {
            return score_before(a, b, scores, first_block);
        };
        // Full stable sort of the candidate ids, with unscored (NaN/inf) blocks demoted to the end so
        // they can never displace a scored block. The sort is TOTAL -- descending score with ascending
        // id on ties -- so the selected set is independent of element order, which is what makes the
        // plan reproducible across runs and across captured-graph replays.
        //
        // The candidate set is the middle region minus the sink and recent regions, so it is bounded
        // by the history length, and unlike the reference's std::nth_element this is O(n log n) on the
        // middle. That cost is accepted deliberately at this stage: the selection runs ONCE per view
        // update (reference section 5 update modes are `step` or `interval`, not per token), and a
        // deterministic total order is worth more here than a partial one, because a plan that is not
        // bit-reproducible cannot be compared between the two arms of an A/B run.
        const auto take = std::min<std::size_t>(static_cast<std::size_t>(remaining), cand.size());
        std::stable_sort(cand.begin(), cand.end(), [&](std::int32_t a, std::int32_t b) {
            const bool fa = finite_score(scores, a, first_block);
            const bool fb = finite_score(scores, b, first_block);
            if (fa != fb) { return fa; }
            return before(a, b);
        });
        static const int kvmem_neighbors = [] {
            const char* v = std::getenv("NINFER_TERNARY_KVMEM_NEIGHBORS");
            return v ? std::max(0, std::atoi(v)) : 0;
        }();
        if (kvmem_neighbors == 0) {
            for (std::size_t i = 0; i < take; ++i) {
                kept.push_back(cand[i]);
                ++out.scored_kept;
            }
        } else {
            // KVMem P1g: seed + neighbours, seeds in score order, until the budget is spent.
            std::vector<char> chosen(static_cast<std::size_t>(cand_end - cand_begin), 0);
            std::size_t used = 0;
            const auto add = [&](std::int32_t id) {
                if (used >= take || id < cand_begin || id >= cand_end) { return; }
                char& c = chosen[static_cast<std::size_t>(id - cand_begin)];
                if (c) { return; }
                c = 1;
                kept.push_back(id);
                ++out.scored_kept;
                ++used;
            };
            for (std::size_t i = 0; i < cand.size() && used < take; ++i) {
                if (!finite_score(scores, cand[i], first_block)) { break; }
                add(cand[i]);
                for (int r = 1; r <= kvmem_neighbors; ++r) {
                    add(cand[i] - r);
                    add(cand[i] + r);
                }
            }
            for (std::size_t i = 0; i < cand.size() && used < take; ++i) { add(cand[i]); }
        }
    }

    // ---- 3. Ascending block id: the window's inside-window positions are chronological.
    std::sort(kept.begin(), kept.end());
    kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
    out.kept = std::move(kept);
    return out;
}

} // namespace ninfer::ops
