#pragma once

// KVMem kv9 port (3060, 2026-10-02): message-aware retrieval query, new-content protection
// and new-input replay range, ported from the old-engine kv8/kv8b/kv9 (kvmem_score.h,
// kvmem_q4b.h). Everything works on the rendered prompt token ids; no tokenizer is needed:
//   <|im_start|> = ids[0], assistant role / newline come from the generation prompt
//   (the last <|im_start|> within the final 32 tokens), <tool_response> id is configurable
//   (NINFER_KVMEM_TOOL_RESPONSE_ID, default 248066; Swift 1.5 / Qwen3.6 vocab).
// Switch: NINFER_KVMEM_KV9=0 restores the qz behaviour (last 512 tokens of the user message,
// fixed recency share, replay from the query span).

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

inline std::int32_t kv9_env_i32(const char* name, std::int32_t fallback) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') { return fallback; }
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    return end == v ? fallback : static_cast<std::int32_t>(x);
}

inline bool kv9_enabled() noexcept {
    static const bool on = kv9_env_i32("NINFER_KVMEM_KV9", 1) != 0;
    return on;
}

inline bool kv9_protect_enabled() noexcept {
    static const bool on = kv9_env_i32("NINFER_KVMEM_PROTECT_NEW", 1) != 0;
    return on;
}

inline constexpr std::uint32_t kKv9MaxSeg = 4;

// Generation prompt position (last <|im_start|> within the final 32 tokens), -1 if absent.
inline std::int64_t kv9_generation_prompt(std::span<const TokenId> ids) noexcept {
    const auto n = static_cast<std::int64_t>(ids.size());
    if (n < 8) { return -1; }
    const TokenId start = ids[0];
    for (std::int64_t p = n - 1; p >= 1 && p + 32 >= n; --p) {
        if (ids[static_cast<std::size_t>(p)] == start) { return p; }
    }
    return -1;
}

// Token ids whose text contains "\n" / "\n\n" (Swift 1.5 = Qwen3.6 vocab, verified 10-02).

inline bool kv9_is_nl(TokenId id) noexcept {
    static constexpr TokenId kNl[] = {
        198, 271, 317, 695, 845, 987, 1031, 1358, 1517, 1726, 2228, 2317,
        2449, 3265, 4326, 4408, 4558, 4969, 5690, 5744, 5773, 6010, 6159, 6219,
        6710, 6987, 7388, 8301, 8675, 8726, 9122, 10109, 10367, 10628, 10797, 11463,
        11953, 12213, 13488, 14200, 14305, 14379, 15216, 15334, 15398, 15602, 15684, 16391,
        16750, 16957, 17116, 17694, 17817, 18015, 18050, 18438, 19075, 19721, 21996, 22616,
        22727, 23574, 23745, 23840, 24630, 25263, 25401, 25445, 25698, 25952, 26467, 26480,
        26771, 27330, 29644, 29773, 30858, 31472, 32077, 32527, 32811, 33006, 33019, 33339,
        33438, 33496, 33954, 34127, 35369, 35412, 35624, 35973, 36444, 36826, 37050, 37528,
        38336, 38544, 39000, 39598, 40260, 40315, 41210, 41585, 43622, 43916, 44058, 44281,
        44879, 44902, 46813, 47458, 47947, 48305, 49404, 49420, 49746, 50946, 52245, 53585,
        54701, 56137, 56580, 56955, 56963, 57103, 57633, 58943, 59366, 59776, 60121, 61332,
        62386, 62591, 63063, 63111, 63454, 64137, 64576, 65117, 65382, 66436, 67511, 68830,
        69229, 69340, 69896, 70273, 70707, 70866, 70929, 71356, 71848, 71990, 72666, 73714,
        75128, 75321, 75668, 75981, 76294, 76374, 76421, 76507, 76909, 77670, 77933, 78035,
        78254, 79519, 81018, 81927, 82471, 82946, 83774, 83829, 84075, 85709, 85748, 85809,
        85844, 85935, 86092, 86182, 87199, 87251, 87328, 88065, 88574, 88927, 88985, 89803,
        91421, 91668, 91804, 92699, 92766, 93957, 94820, 94861};
    return std::binary_search(std::begin(kNl), std::end(kNl), id);
}
inline bool kv9_is_para(TokenId id) noexcept {
    static constexpr TokenId kPara[] = {
        271, 987, 1358, 4558, 4969, 5773, 14200, 14305, 14379, 15334, 18050, 21996,
        22727, 23840, 26467, 32811, 33019, 33438, 38544, 43622, 52245, 58943, 59366, 63063,
        63454, 64137, 68830, 71848, 71990, 73714, 76374, 76507, 78035, 79519, 84075, 86182,
        88574};
    return std::binary_search(std::begin(kPara), std::end(kPara), id);
}

inline std::uint64_t kv9_hash(std::span<const TokenId> ids, std::uint32_t b, std::uint32_t e) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::uint32_t i = b; i < e && i < ids.size(); ++i) {
        h ^= static_cast<std::uint32_t>(ids[i]);
        h *= 1099511628211ULL;
    }
    return h ^ (static_cast<std::uint64_t>(b) << 32) ^ e;
}

// Query rows captured per request (pinned host buffer per lane: rows x 16 layers x 6144 bf16).
// Hard cap 2048; the effective size is kv9_query_max() (default 1024, 192 MiB pinned; 512/1024/2048
// tied on the 190K needle ranks, 1024 chosen 10-02).
inline constexpr std::uint32_t kKv9QueryRowsMax = 2048;

// QUERY_MAX: KV9_QMAX (diagnostic, survives the launcher) or NINFER_KVMEM_QUERY_MAX, default 1024.
inline std::int32_t kv9_query_max() noexcept {
    std::int32_t q = kv9_env_i32("KV9_QMAX", 0);
    if (q <= 0) { q = kv9_env_i32("NINFER_KVMEM_QUERY_MAX", 1024); }
    return std::clamp(q, 8, static_cast<std::int32_t>(kKv9QueryRowsMax));
}

struct Kv9Plan {
    std::uint32_t nseg = 0;
    std::uint32_t begin[kKv9MaxSeg] = {};
    std::uint32_t end[kKv9MaxSeg]   = {};
    std::int32_t msg[kKv9MaxSeg]    = {};   // 0 = newest message (msg_*), 1 = real user message (real_*)
    bool tool = false;
    std::int64_t msg_begin = -1, msg_end = -1;    // last non-assistant message content
    std::int64_t real_begin = -1, real_end = -1;  // last real user message (tool case)
    std::int64_t new_begin = -1;                  // <|im_start|> of the first new message
    // Ranges of the real user message that a later tool round asks for (stash these rows).
    std::uint32_t stash_b[2] = {}, stash_e[2] = {};
    std::uint32_t stash_n = 0;
};

// kv8 msg plan + kv9 tool rounds (3060: QUERY_MAX 512 instead of kv9's 256, tail aligned):
//   normal round : last user message, whole if <= 512 tokens else head 256 + tail 256;
//   tool round   : the newest tool result (whole if <= 256 else head 128 + tail 128) plus the
//                  last real user message (whole if <= 256 else head 128 + tail 128; these rows
//                  are a subset of the normal round's, so they come from the stash).
inline Kv9Plan kv9_make_plan(std::span<const TokenId> ids, std::int32_t qmax_override = 0) noexcept {
    Kv9Plan plan;
    const auto n = static_cast<std::int64_t>(ids.size());
    const std::int64_t g = kv9_generation_prompt(ids);
    if (g < 3 || g + 1 >= n) { return plan; }
    const TokenId start     = ids[0];
    const TokenId role_asst = ids[static_cast<std::size_t>(g + 1)];
    const TokenId nl        = (g + 2 < n) ? ids[static_cast<std::size_t>(g + 2)] : -1;
    const TokenId im_end    = (ids[static_cast<std::size_t>(g - 1)] == nl)
                                  ? ids[static_cast<std::size_t>(g - 2)]
                                  : ids[static_cast<std::size_t>(g - 1)];
    const TokenId tool_id   = kv9_env_i32("NINFER_KVMEM_TOOL_RESPONSE_ID", 248066);
    const auto at = [&](std::int64_t p) { return ids[static_cast<std::size_t>(p)]; };

    // new input: the trailing non-assistant messages (kv9 fix: position 0 is included).
    for (std::int64_t p = g - 1; p >= 0; --p) {
        if (at(p) != start) { continue; }
        if (p + 1 < n && at(p + 1) == role_asst) { break; }
        plan.new_begin = p;
    }

    const auto content = [&](std::int64_t s, std::int64_t e, std::int64_t& cb, std::int64_t& ce) {
        cb = s + 2;
        if (cb < e && at(cb) == nl) { ++cb; }
        ce = e;
        while (ce > cb && (at(ce - 1) == nl || at(ce - 1) == im_end)) { --ce; }
    };
    std::int64_t msg_end = g;
    TokenId user_role = -1;
    bool need_real = false;
    std::int64_t first_cb = -1, first_ce = -1, real_cb = -1, real_ce = -1;
    for (std::int64_t p = g - 1; p >= 0; --p) {
        if (at(p) != start) { continue; }
        const TokenId role = at(p + 1);
        if (role != role_asst) {
            std::int64_t cb = 0, ce = 0;
            content(p, msg_end, cb, ce);
            const bool is_tool = cb < ce && at(cb) == tool_id;
            if (first_cb < 0) {
                first_cb = cb; first_ce = ce; user_role = role; plan.tool = is_tool;
                if (!is_tool) { break; }
                need_real = true;
            } else if (need_real && role == user_role && !is_tool && ce > cb) {
                real_cb = cb; real_ce = ce;
                break;
            }
        }
        msg_end = p;
    }
    if (first_cb < 0 || first_ce <= first_cb) { return plan; }
    plan.msg_begin = first_cb;
    plan.msg_end   = first_ce;
    plan.real_begin = real_cb;
    plan.real_end   = real_ce;

    std::int32_t cur_msg = 0;
    bool to_stash = false;
    const auto push = [&](std::int64_t b, std::int64_t e) {
        if (to_stash) {
            if (e > b && plan.stash_n < 2) {
                plan.stash_b[plan.stash_n] = static_cast<std::uint32_t>(b);
                plan.stash_e[plan.stash_n] = static_cast<std::uint32_t>(e);
                ++plan.stash_n;
            }
            return;
        }
        if (e > b && plan.nseg < kKv9MaxSeg) {
            plan.begin[plan.nseg] = static_cast<std::uint32_t>(b);
            plan.end[plan.nseg]   = static_cast<std::uint32_t>(e);
            plan.msg[plan.nseg]   = cur_msg;
            ++plan.nseg;
        }
    };
    const bool align_ok = g + 2 < n && kv9_is_nl(at(g + 2));
    const auto add_msg = [&](std::int64_t cb, std::int64_t ce, std::int64_t budget) {
        const std::int64_t len = ce - cb;
        if (len <= 0 || budget <= 0) { return; }
        if (len <= budget) { push(cb, ce); return; }
        const std::int64_t h = budget / 2;
        push(cb, cb + h);
        // 3060 (user-approved, not in kv9): a long message's tail usually is "document ...
        // \n\n question"; start the tail right after the last paragraph (else line) break inside
        // it so the query is the question, not the document's last lines (kept >= 8 tokens).
        std::int64_t tb = ce - (budget - h);
        if (align_ok && kv9_env_i32("NINFER_KVMEM_QUERY_TAIL_ALIGN", 1) != 0) {
            std::int64_t cut = -1;
            for (std::int64_t q = ce - 9; q >= tb && cut < 0; --q) { if (kv9_is_para(at(q))) { cut = q + 1; } }
            for (std::int64_t q = ce - 9; q >= tb && cut < 0; --q) { if (kv9_is_nl(at(q))) { cut = q + 1; } }
            if (cut > tb) { tb = cut; }
        }
        push(tb, ce);
    };
    const std::int64_t qmax = qmax_override > 0
        ? std::clamp(qmax_override, 8, static_cast<std::int32_t>(kKv9QueryRowsMax))
        : kv9_query_max();
    if (plan.tool && real_cb >= 0) {
        // kv9: tool result head/tail + the real user message's head/tail (half budget each).
        cur_msg = 0;
        add_msg(first_cb, first_ce, qmax / 2);
        cur_msg = 1;
        add_msg(real_cb, real_ce, qmax - qmax / 2);
        to_stash = true;
        add_msg(real_cb, real_ce, qmax - qmax / 2);
    } else {
        cur_msg = 0;
        add_msg(first_cb, first_ce, qmax);
        if (!plan.tool) {
            to_stash = true;
            add_msg(first_cb, first_ce, qmax - qmax / 2);
        }
    }
    return plan;
}

// kv9 replay range: from the new input's <|im_start|> to the end, at most
// min(NINFER_KVMEM_REPLAY_MAX=6144, window/4) tokens (only the tail when longer). 0 = unknown.
inline std::uint32_t kv9_replay_begin(std::span<const TokenId> ids, std::uint32_t prompt_tokens,
                                      std::uint32_t window_tokens) noexcept {
    if (ids.size() < prompt_tokens || prompt_tokens < 8) { return 0; }
    const Kv9Plan plan = kv9_make_plan(ids.first(prompt_tokens));
    if (plan.new_begin < 0) { return 0; }
    std::uint32_t cap = static_cast<std::uint32_t>(std::max(64, kv9_env_i32("NINFER_KVMEM_REPLAY_MAX", 6144)));
    if (window_tokens != 0) { cap = std::min(cap, std::max(64U, window_tokens / 4U)); }
    std::uint32_t begin = static_cast<std::uint32_t>(plan.new_begin);
    if (prompt_tokens - begin > cap) { begin = prompt_tokens - cap; }
    return begin;
}

struct Kv9Protect {
    std::vector<std::uint32_t> b, e;   // [b, e) absolute positions, priority order
    std::int64_t used = 0, cap = 0, asst_used = 0;
    int msgs = 0, units = 0, split = 0;
};

// "New content" = trailing non-assistant messages before the generation prompt (parallel tool
// results split at each <tool_response>). The cap is water-filled over them; a long unit keeps
// head + tail with cut points pulled inward to a paragraph / line end within `slack` tokens.
// The previous assistant answer only gets what is left (its tail).
inline Kv9Protect kv9_make_protect(std::span<const TokenId> ids, std::int64_t cap, std::int64_t sink) {
    Kv9Protect pr;
    pr.cap = cap;
    const auto n = static_cast<std::int64_t>(ids.size());
    const std::int64_t g = kv9_generation_prompt(ids);
    if (g < 3 || g + 2 >= n || cap <= 0) { return pr; }
    const auto at = [&](std::int64_t p) { return ids[static_cast<std::size_t>(p)]; };
    const TokenId start     = ids[0];
    const TokenId role_asst = at(g + 1);
    const bool align_ok     = kv9_is_nl(at(g + 2));
    const TokenId tool_id   = kv9_env_i32("NINFER_KVMEM_TOOL_RESPONSE_ID", 248066);
    const std::int64_t slack = kv9_env_i32("NINFER_KVMEM_PROTECT_ALIGN", 256);

    constexpr int kMaxUnits = 16;
    std::int64_t ub[kMaxUnits] = {}, ue[kMaxUnits] = {};
    int nu = 0;
    std::int64_t msg_end = g, asst_b = -1, asst_e = -1;
    for (std::int64_t p = g - 1; p >= 0 && nu < kMaxUnits; --p) {
        if (at(p) != start) { continue; }
        if (at(p + 1) == role_asst) { asst_b = p; asst_e = msg_end; break; }
        ++pr.msgs;
        std::int64_t seg_end = msg_end;
        for (std::int64_t q = msg_end - 1; q > p && nu < kMaxUnits; --q) {
            if (at(q) == tool_id) { ub[nu] = q; ue[nu] = seg_end; ++nu; seg_end = q; }
        }
        if (seg_end > p) {
            if (seg_end - p < 16 && nu > 0 && ub[nu - 1] == seg_end) { ub[nu - 1] = p; }
            else if (nu < kMaxUnits) { ub[nu] = p; ue[nu] = seg_end; ++nu; }
        }
        msg_end = p;
    }
    std::int64_t len[kMaxUnits] = {};
    int k = 0;
    for (int i = 0; i < nu; ++i) {
        const std::int64_t b = std::max(ub[i], sink);
        if (ue[i] > b) { ub[k] = b; ue[k] = ue[i]; len[k] = ue[i] - b; ++k; }
    }
    nu = k;
    pr.units = nu;
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
    const auto head_end = [&](std::int64_t b, std::int64_t t, std::int64_t h) -> std::int64_t {
        if (!align_ok) { return t; }
        const std::int64_t lo = std::max(b, t - std::min(slack, h / 2));
        for (std::int64_t q = t - 1; q >= lo; --q) { if (kv9_is_para(at(q))) { return q + 1; } }
        for (std::int64_t q = t - 1; q >= lo; --q) { if (kv9_is_nl(at(q))) { return q + 1; } }
        return t;
    };
    const auto tail_begin = [&](std::int64_t u, std::int64_t e, std::int64_t t) -> std::int64_t {
        if (!align_ok) { return u; }
        const std::int64_t hi = std::min(e - 1, u + std::min(slack, t / 2));
        for (std::int64_t q = std::max<std::int64_t>(0, u - 1); q < hi; ++q) { if (kv9_is_para(at(q))) { return q + 1; } }
        for (std::int64_t q = std::max<std::int64_t>(0, u - 1); q < hi; ++q) { if (kv9_is_nl(at(q))) { return q + 1; } }
        return u;
    };
    const auto add = [&](std::int64_t b, std::int64_t e) {
        if (e > b) {
            pr.b.push_back(static_cast<std::uint32_t>(b));
            pr.e.push_back(static_cast<std::uint32_t>(e));
            pr.used += e - b;
        }
    };
    for (int i = 0; i < nu; ++i) {   // newest unit first
        if (share[i] <= 0) { continue; }
        if (share[i] >= len[i]) { add(ub[i], ue[i]); continue; }
        ++pr.split;
        const std::int64_t h = share[i] / 2, t = share[i] - h;
        add(ub[i], head_end(ub[i], ub[i] + h, h));
        add(tail_begin(ue[i] - t, ue[i], t), ue[i]);
    }
    const std::int64_t left = cap - pr.used;
    if (asst_b >= 0 && left >= 64) {
        const std::int64_t b = std::max(asst_b, sink);
        if (asst_e > b) {
            const std::int64_t before = pr.used;
            if (asst_e - b <= left) { add(b, asst_e); }
            else { add(tail_begin(asst_e - left, asst_e, left), asst_e); }
            pr.asst_used = pr.used - before;
        }
    }
    // the generation prompt itself (a few tokens) is always the newest content
    add(g, n);
    return pr;
}

} // namespace ninfer::models::qwen3_5::detail
