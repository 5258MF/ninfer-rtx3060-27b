#pragma once

// KVMem P4b (kv8b) -- 2026-09-30.
// protect-new v2: message-aware protection of THIS ROUND's new content in the boundary window.
//   v1 (kv8) protected the blocks this request prefilled as one lump: <= cap => all, else first
//   cap/2 + last cap/2 blocks, cut at arbitrary 64-token block edges, and the re-rendered previous
//   answer took the head share.
//   v2: "new content" = the trailing non-assistant messages before the generation prompt (user
//   message / tool results; parallel tool results are split at each <tool_response>). The cap is
//   water-filled over them (short ones whole, long ones share the rest equally); a long unit keeps
//   head + tail whose cut points are pulled INWARD to the nearest paragraph end ("\n\n"-token) or
//   else line end ("\n"-token) within SCORE_PROTECT_NEW_ALIGN tokens (never grows past the cap).
//   The previous assistant answer only gets what is left (its tail). Independent of the prefix
//   cache reuse point => first send and regenerate protect the same ranges.
// Switch: SCORE_PROTECT_NEW=1 (as before) now means v2; SCORE_PROTECT_NEW_V1=1 restores v1.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>

#include "ops/kvmem/kvmem_q4.h"

namespace ninfer::ops::detail {

inline bool kvmem_q4_protect_v1() noexcept {
    static const bool on = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW_V1", 0) != 0;
    return on;
}

struct KvMemQ4Protect {
    static constexpr int kMax = 24;
    std::int32_t n = 0;
    std::int32_t b[kMax] = {};   // absolute prompt positions [b, e), priority order
    std::int32_t e[kMax] = {};
};

inline KvMemQ4Protect& kvmem_q4_protect() {
    static KvMemQ4Protect prot;
    return prot;
}

// Token ids (Qwen3.6 vocab, tokenizer.json) whose text contains "\n" / "\n\n". Only used to place
// cut points; a different vocab is detected (the generation prompt's "\n" is not in the table) and
// alignment is then skipped.
inline bool kvmem_q4b_is_nl(std::int32_t id) noexcept {
    static constexpr std::int32_t kNl[] = {
        198, 271, 317, 695, 845, 987, 1031, 1358, 1517, 1726, 2228, 2317, 2449, 3265, 4326, 4408, 4558, 4969, 5690, 5744,
        5773, 6010, 6159, 6219, 6710, 6987, 7388, 8301, 8675, 8726, 9122, 10109, 10367, 10628, 10797, 11463, 11953, 12213, 13488, 14200,
        14305, 14379, 15216, 15334, 15398, 15602, 15684, 16391, 16750, 16957, 17116, 17694, 17817, 18015, 18050, 18438, 19075, 19721, 21996, 22616,
        22727, 23574, 23745, 23840, 24630, 25263, 25401, 25445, 25698, 25952, 26467, 26480, 26771, 27330, 29644, 29773, 30858, 31472, 32077, 32527,
        32811, 33006, 33019, 33339, 33438, 33496, 33954, 34127, 35369, 35412, 35624, 35973, 36444, 36826, 37050, 37528, 38336, 38544, 39000, 39598,
        40260, 40315, 41210, 41585, 43622, 43916, 44058, 44281, 44879, 44902, 46813, 47458, 47947, 48305, 49404, 49420, 49746, 50946, 52245, 53585,
        54701, 56137, 56580, 56955, 56963, 57103, 57633, 58943, 59366, 59776, 60121, 61332, 62386, 62591, 63063, 63111, 63454, 64137, 64576, 65117,
        65382, 66436, 67511, 68830, 69229, 69340, 69896, 70273, 70707, 70866, 70929, 71356, 71848, 71990, 72666, 73714, 75128, 75321, 75668, 75981,
        76294, 76374, 76421, 76507, 76909, 77670, 77933, 78035, 78254, 79519, 81018, 81927, 82471, 82946, 83774, 83829, 84075, 85709, 85748, 85809,
        85844, 85935, 86092, 86182, 87199, 87251, 87328, 88065, 88574, 88927, 88985, 89803, 91421, 91668, 91804, 92699, 92766, 93957, 94820, 94861};
    return std::binary_search(std::begin(kNl), std::end(kNl), id);
}
inline bool kvmem_q4b_is_para(std::int32_t id) noexcept {
    static constexpr std::int32_t kPara[] = {
        271, 987, 1358, 4558, 4969, 5773, 14200, 14305, 14379, 15334, 18050, 21996, 22727, 23840, 26467, 32811, 33019, 33438, 38544, 43622,
        52245, 58943, 59366, 63063, 63454, 64137, 68830, 71848, 71990, 73714, 76374, 76507, 78035, 79519, 84075, 86182, 88574};
    return std::binary_search(std::begin(kPara), std::end(kPara), id);
}

inline void kvmem_q4b_make_protect(const std::int32_t* ids, std::int64_t n) noexcept {
    KvMemQ4Protect& pr = kvmem_q4_protect();
    pr.n = 0;
    if (ids == nullptr || n < 8) { return; }
    const std::int32_t start = ids[0];
    std::int64_t g = -1;   // generation prompt = last <|im_start|> within the final 32 tokens
    for (std::int64_t p = n - 1; p >= 1 && p + 32 >= n; --p) {
        if (ids[p] == start) { g = p; break; }
    }
    if (g < 3 || g + 2 >= n) {
        std::fprintf(stderr, "[kvmem-q4] protect2 prompt=%lld no generation prompt => nothing protected\n",
                     static_cast<long long>(n));
        return;
    }
    const std::int32_t role_asst = ids[g + 1];
    const bool align_ok = kvmem_q4b_is_nl(ids[g + 2]);
    const std::int32_t tool_id = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_TOOL_RESPONSE_ID", 248066);
    const std::int64_t cap =
        std::max<std::int32_t>(128, kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW_MAX", 12288));
    const std::int64_t sink = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);
    const std::int64_t slack_env = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW_ALIGN", 256);

    // ---- units: trailing non-assistant messages (newest first), split at <tool_response>
    constexpr int kMaxUnits = 16;
    std::int64_t ub[kMaxUnits] = {}, ue[kMaxUnits] = {};
    int nu = 0, nmsg = 0;
    std::int64_t msg_end = g, asst_b = -1, asst_e = -1;
    for (std::int64_t p = g - 1; p >= 0 && nu < kMaxUnits; --p) {
        if (ids[p] != start) { continue; }
        if (ids[p + 1] == role_asst) { asst_b = p; asst_e = msg_end; break; }
        ++nmsg;
        std::int64_t seg_end = msg_end;
        for (std::int64_t q = msg_end - 1; q > p && nu < kMaxUnits; --q) {
            if (ids[q] == tool_id) { ub[nu] = q; ue[nu] = seg_end; ++nu; seg_end = q; }
        }
        if (seg_end > p) {
            if (seg_end - p < 16 && nu > 0 && ub[nu - 1] == seg_end) { ub[nu - 1] = p; }   // role header
            else if (nu < kMaxUnits) { ub[nu] = p; ue[nu] = seg_end; ++nu; }
        }
        msg_end = p;
    }
    // the sink is always in the window: clip, drop what is left empty
    std::int64_t len[kMaxUnits] = {};
    int k = 0;
    for (int i = 0; i < nu; ++i) {
        const std::int64_t b = std::max(ub[i], sink);
        if (ue[i] > b) { ub[k] = b; ue[k] = ue[i]; len[k] = ue[i] - b; ++k; }
    }
    nu = k;

    // ---- water-fill the cap: short units whole, long units share the rest equally
    std::int64_t share[kMaxUnits] = {};
    int order[kMaxUnits];
    for (int i = 0; i < nu; ++i) { order[i] = i; }
    std::sort(order, order + nu, [&](int a, int b) { return len[a] != len[b] ? len[a] < len[b] : a < b; });
    std::int64_t rem = cap;
    for (int j = 0; j < nu; ++j) {
        const int i = order[j];
        share[i] = std::min<std::int64_t>(len[i], rem / (nu - j));
        rem -= share[i];
    }

    int al_p = 0, al_n = 0, al_x = 0;
    // end of a head: last boundary token in [lo, t) => cut right after it (never past t)
    const auto head_end = [&](std::int64_t b, std::int64_t t, std::int64_t h) -> std::int64_t {
        if (!align_ok) { ++al_x; return t; }
        const std::int64_t lo = std::max(b, t - std::min(slack_env, h / 2));
        for (std::int64_t q = t - 1; q >= lo; --q) { if (kvmem_q4b_is_para(ids[q])) { ++al_p; return q + 1; } }
        for (std::int64_t q = t - 1; q >= lo; --q) { if (kvmem_q4b_is_nl(ids[q]))   { ++al_n; return q + 1; } }
        ++al_x; return t;
    };
    // start of a tail: first boundary token in [u-1, hi) => start right after it (never before u)
    const auto tail_begin = [&](std::int64_t u, std::int64_t e, std::int64_t t) -> std::int64_t {
        if (!align_ok) { ++al_x; return u; }
        const std::int64_t hi = std::min(e - 1, u + std::min(slack_env, t / 2));
        for (std::int64_t q = u - 1; q < hi; ++q) { if (kvmem_q4b_is_para(ids[q])) { ++al_p; return q + 1; } }
        for (std::int64_t q = u - 1; q < hi; ++q) { if (kvmem_q4b_is_nl(ids[q]))   { ++al_n; return q + 1; } }
        ++al_x; return u;
    };
    std::int64_t used = 0;
    const auto add = [&](std::int64_t b, std::int64_t e) {
        if (e > b && pr.n < KvMemQ4Protect::kMax) {
            pr.b[pr.n] = static_cast<std::int32_t>(b);
            pr.e[pr.n] = static_cast<std::int32_t>(e);
            ++pr.n;
            used += e - b;
        }
    };
    int split = 0;
    for (int i = 0; i < nu; ++i) {   // newest unit first
        if (share[i] <= 0) { continue; }
        if (share[i] >= len[i]) { add(ub[i], ue[i]); continue; }
        ++split;
        const std::int64_t h = share[i] / 2, t = share[i] - h;
        add(ub[i], head_end(ub[i], ub[i] + h, h));
        add(tail_begin(ue[i] - t, ue[i], t), ue[i]);
    }
    // previous assistant answer: only what is left, its tail (tool call / conclusion)
    std::int64_t asst_used = 0;
    const std::int64_t left = cap - used;
    if (asst_b >= 0 && left >= 64) {
        const std::int64_t b = std::max(asst_b, sink);
        if (asst_e > b) {
            const std::int64_t before = used;
            if (asst_e - b <= left) { add(b, asst_e); }
            else { add(tail_begin(asst_e - left, asst_e, left), asst_e); }
            asst_used = used - before;
        }
    }
    std::fprintf(stderr,
                 "[kvmem-q4] protect2 prompt=%lld msgs=%d units=%d split=%d asst=[%lld,%lld) asst_used=%lld "
                 "used=%lld cap=%lld align=%s para=%d nl=%d none=%d ranges:",
                 static_cast<long long>(n), nmsg, nu, split, static_cast<long long>(asst_b),
                 static_cast<long long>(asst_e), static_cast<long long>(asst_used), static_cast<long long>(used),
                 static_cast<long long>(cap), align_ok ? "on" : "off(vocab)", al_p, al_n, al_x);
    for (int r = 0; r < pr.n; ++r) { std::fprintf(stderr, " [%d,%d)", pr.b[r], pr.e[r]); }
    std::fprintf(stderr, "\n");
}

} // namespace ninfer::ops::detail
