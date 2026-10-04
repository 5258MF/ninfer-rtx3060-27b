#pragma once

// KVMem 选择探针（**最小切片**）· 只做三件事：按 query 打分 → 选块 → 打一行证据日志
//
// 【它是什么】把已经在库里的三件套接上主路径的**第一刀**：
//     kvmem_retrieve（官方 KVMem 论文 Eq.10 的 query-conditioned 打分器）
//   → kvmem_select  （host：sink/recent 恒留 + 中段按分降序）
//   → 一行 fprintf（对齐官方实现的三口径：selected 非连续 / skip>0 / window≈budget）
//
// 【它不是什么】**不碰页表、不碰 cache position、不碰 attention envelope、不 arm 窗口**。
//   ⇒ 可见集不变、OFF 臂逐位不变、**不需要 re-RoPE**（那属于第二刀：见 kvmem_window.h 的 arming 前置）。
//   ⇒ 因此本探针**不会让任何答案变对或变错**，它只回答"选择器选得对不对"。
//
// 【开关】NINFER_TERNARY_KVMEM_SCORE=1（默认关）。OFF 臂：不分配、不启动、不读索引、不写日志。
//   ⚠️ 开关与 env 只在这个头文件里读（与 kvmem_shadow.h 同一条纪律：开关的读取点必须唯一）。
//
// 【为什么只有打分器需要新代码】索引（mean_k_index，prefill 时按 chunk 喂）、raw-K 收割
//   （raw_k_harvest）、窗口载体（kvmem_window）**都已经接在引擎里**；缺的正是中间那句"按分选块"。
//
// 【两处必须记住的接缝】
//   1. 索引存的是每块 key 的 **和**（不是 mean）⇒ 打分器要用 block_tokens 自己除；而索引里的计数是
//      **F32**，打分器要 **int32** ⇒ 这里在 host 侧按 blocks_written/tail_fill 造 int32 数组
//      （尾部不满的块必须带**真实填充**，不能带名义块大小 —— mean_k_index.h:127-131）。
//   2. query 必须是**内容帧**（de-RoPE 之后的位置无关帧），与索引同帧 ⇒ 引擎里那个可用的点只有
//      `qn`（RMS 归一化后、`ops::rope` 覆盖之前）。这就是为什么 accumulate() 的调用点在 rope 之前。
//
// 【已知边界（写死在代码里，别当没看见）】
//   * **多 lane 不安全**：mean-K 索引是**全进程一份**（mean_k_index.h:248-251 自陈），本探针的
//     score 缓冲同样是全进程一份 ⇒ 只在单路（--max-concurrency 1）下有意义。多路时必须 per-lane。
//   * 只对"短 query span"打分（≤ NINFER_TERNARY_KVMEM_SCORE_MAXQ，默认 256 个 token）：文档
//     ingest 那种几万 token 的 chunk 打分代价是 O(块数×query长)，会拖死 prefill；而且那种 chunk
//     的"query"本来也不是检索 query。
//   * ★ 2026-09-26（语义档）：NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=N ⇒ 只把 chunk 的最后 N 个
//     token 当检索 query 打分（官方 kvmem_set_query_span 的等价物：query 是"问"，不是整段
//     ingest），MAXQ 的门限看 **span 长度**、不看 chunk 长度 —— 解掉"探针只在短 query chunk
//     上打分（skip long chunk tokens=1024 > MAXQ=256）"那块拦路石。默认 0 = 旧行为（逐位不变）。
//   * 本探针**不改可见集** ⇒ 它与"答案对不对"无关；要验证"被选中 ⇒ 真的进了 softmax"必须等第二刀。

#include "ops/kvmem/kvmem_retrieve_launch.h"
#include "ops/kvmem/kvmem_cpu_score.h"
#include <chrono>
#include <optional>
#include "ops/kvmem/kvmem_select.h"
#include "ops/kvmem/mean_k_index.h"
#include "ops/kvmem/kvmem_q4.h"   // KVMem P4 (kv8)
#include "ops/kvmem/kvmem_q4b.h"  // KVMem P4b (kv8b)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// ---- 开关（唯一读取点；env 只读一次，与 kvmem_shadow_enabled() 同形）---------------------------
inline bool kvmem_score_enabled() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_SCORE");
        if (v != nullptr && v[0] != '\0') { return v[0] != '0'; }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return kv != nullptr && kv[0] != '\0' && kv[0] != '0';
    }();
    return on;
}

inline std::int32_t kvmem_score_env_i32(const char* name, std::int32_t fallback) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') { return fallback; }
    const long parsed = std::strtol(v, nullptr, 10);
    return parsed > 0 ? static_cast<std::int32_t>(parsed) : fallback;
}

// ---- 探针状态（进程级单例，与 mean-K 索引同寿命）---------------------------------------------
struct KvMemScoreProbe {
    // device 缓冲
    float*        score            = nullptr;  // [capacity_blocks]，每 chunk 前清零（打分器只 ADD）
    std::int32_t* block_tokens_dev = nullptr;  // [capacity_blocks]，尾块带真实填充（device 侧）
    // host staging
    std::vector<float>        score_host;
    std::vector<std::int32_t> tokens_host;

    std::int32_t capacity_blocks = 0;
    std::int32_t block_tokens    = 0;      // == kPagedKVPageSize，由调用方传入（house rule：不许写死）
    std::int32_t chunk_tokens    = 0;      // live 平面的 token 行数（= q_layer_stride 口径）
    std::int32_t n_blocks        = 0;      // 本 chunk 打分的"历史"块数（= 打分开始时的 blocks_written）
    // ---- ★ 分数的**出处**（2026-09-26 语义档要用：静默用错查询/错时刻 = 静默错答案）------------
    std::int32_t query_span_begin  = 0;    // 本次 query span 在 live 平面里的起点（QUERY_TAIL 档）
    std::int32_t query_span_tokens = 0;    // 本次真正打分的 query token 数（= span 长度）
    std::int32_t scored_n_blocks   = 0;    // 这份分数描述块 [0, scored_n_blocks)（finish 时登记）
    bool scored_last_chunk = false;        // 最近一个 prefill chunk 真打过分（语义档的准入闸）
    std::int32_t layers_total    = 0;
    std::int32_t n_heads         = 0;
    std::int32_t n_kv_heads      = 0;
    std::int32_t head_dim        = 0;
    bool armed    = false;                 // 本 chunk 正在打分
    bool scored_this_chunk = false;        // 本 chunk 真的累加过（否则 finish 打出来的是全零假日志）
    bool failed   = false;                 // 出过致命几何/分配错 ⇒ 之后一律不打分（但引擎继续跑）
    bool logged_once = false;
    // ★ P1c 跨 chunk 累加（QUERY_TAIL 档）：短尾 chunk 把分数加在上一个 chunk 的分数上。
    bool         pending_clear      = false;
    bool         carry_ok           = false;
    std::int32_t carry_prev_blocks  = 0;
    std::int32_t carry_prev_tokens  = 0;
    std::int32_t carry_tokens       = 0;   // 本次分数里来自前面 chunk 的 query token 数
    std::int32_t last_total_tokens  = 0;   // 上次 finish 时分数里的 query token 总数
    // ---- KVMem P4 (kv8): msg-mode query (head/tail of the last user message) -----------------------
    static constexpr int kQ4MaxSeg = 4;
    bool          q4_in_chunk       = false;  // a prefill chunk is between q4_begin_chunk and finish
    bool          q4_active         = false;  // this request has a valid msg-mode plan
    bool          q4_scored         = false;  // this request's msg-mode scores are in score_host
    bool          q4_complete_chunk = false;  // the current chunk ran the msg-mode scoring
    bool          q4_cleared        = false;
    bool          q4_tool           = false;  // last user message is a tool result
    bool          q4_owns_scores    = false;  // score_host came from msg mode (device p.score is stale)
    std::int32_t  q4_nseg           = 0;
    std::int32_t  q4_seg_begin[kQ4MaxSeg] = {};   // absolute prompt positions [begin, end)
    std::int32_t  q4_seg_end[kQ4MaxSeg]   = {};
    std::int32_t  q4_seg_row0[kQ4MaxSeg]  = {};   // first stash row of the segment
    std::int32_t  q4_have_lo[kQ4MaxSeg]   = {};   // rows [have_lo, have_hi) are in the stash
    std::int32_t  q4_have_hi[kQ4MaxSeg]   = {};
    std::int32_t  q4_used[kQ4MaxSeg]      = {};   // query rows actually scored per segment
    std::uint64_t q4_seg_hash[kQ4MaxSeg]  = {};
    std::int32_t  q4_rows           = 0;      // stash rows needed by the plan
    std::int32_t  q4_plan_end       = 0;      // max segment end
    std::int32_t  q4_chunk_base     = 0;
    std::int32_t  q4_chunk_rows     = 0;
    std::int32_t  q4_hist_blocks    = 0;
    std::int32_t  q4_layers         = 0;
    std::int32_t  q4_row_elems      = 0;      // n_heads * head_dim (bf16)
    std::int32_t  q4_stash_rows     = 0;      // rows per layer allocated
    std::optional<PinnedHostBuffer> q4_host_stash;
    std::optional<PinnedHostBuffer> tail_host_stash;
    std::int32_t tail_host_rows = 0;
    void*         q4_stash          = nullptr;  // bf16 [layers][stash_rows][row_elems]
    float*        q4_score          = nullptr;  // [kQ4MaxSeg][q4_score_cap]
    std::int32_t  q4_score_cap      = 0;
    std::vector<float> q4_host;
    // 3060 round 10: Eq.10 candidate mask (1 = kept anyway, left out of the softmax)
    std::uint8_t* q4_excl     = nullptr;   // device [q4_excl_cap]
    std::int32_t  q4_excl_cap = 0;
    std::int32_t  q4_excl_n   = 0;         // excluded blocks in the current scoring chunk (0 = none)
    std::vector<std::uint8_t> q4_excl_host;
    // ---- KVMem kv9 (item 1): Q stash of the last REAL user message, kept across tool rounds ------
    // Rows live in the same allocation, after the kQ4MainRows plan rows of every layer. Identity =
    // content range + hash (positions included), selection = the tool-case budget (qmax - qmax/2).
    // 3060 round 10: plan rows >= kvmem_q4_qmax(), Q-stash rows = kvmem_q4_qrows() (were 256 / 128)
    std::int32_t  q4q_cb = -1, q4q_ce = -1;
    std::uint64_t q4q_hash = 0;
    std::int32_t  q4q_nseg = 0;
    std::int32_t  q4q_begin[2] = {}, q4q_end[2] = {}, q4q_row0[2] = {};
    std::int32_t  q4q_have_lo[2] = {}, q4q_have_hi[2] = {};
    bool          q4q_inject = false;   // new request: copy matching real-message rows after alloc
};

// 3060 round 10 (user-approved): msg-mode query size, was hard-capped at 256. Env
// NINFER_TERNARY_KVMEM_SCORE_QUERY_MAX, default 512, range [8, 2048]. Host stash by default (GPU when CPU_RETRIEVAL=0) =
// layers x (roundup64(qmax) + qmax - qmax/2) rows x n_heads*head_dim x 2 B (Qwen3.6-27B: 192 KiB/row
// => 256: 72 MiB, 512: 144 MiB, 1024: 288 MiB). 256 + TAIL_ALIGN=0 + EQ10_EXCLUDE=0 = old behaviour.
inline std::int32_t kvmem_q4_qmax() noexcept {
    static const std::int32_t q = std::min<std::int32_t>(
        2048, std::max<std::int32_t>(8, kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_MAX", 512)));
    return q;
}
inline std::int32_t kvmem_q4_qrows() noexcept { return kvmem_q4_qmax() - kvmem_q4_qmax() / 2; }
// 3060 round 10 (user-approved, paper Eq.10 / Swift 1.5 engine): blocks that are kept anyway (sink,
// protected new content -- whose newest tail also covers the replayed recent tail) are left out of
// the softmax so their mass goes to the middle. NINFER_TERNARY_KVMEM_EQ10_EXCLUDE=0 = old behaviour.
inline bool kvmem_q4_eq10_exclude() noexcept {
    static const bool on = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_EQ10_EXCLUDE", 1) != 0;
    return on;
}

inline KvMemScoreProbe& kvmem_score_probe() {
    static KvMemScoreProbe probe;
    return probe;
}

// KVMem P3a-h: cut the P1c carry chain at a request boundary.
inline void kvmem_score_forget_carry(const char* why) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.scored_last_chunk || p.last_total_tokens != 0) {
        std::fprintf(stderr, "[kvmem-p3h] query carry cut (%s; had query_tokens=%d)\n", why,
                     static_cast<int>(p.last_total_tokens));
    }
    p.scored_last_chunk = false;
    p.last_total_tokens = 0;
}

inline void kvmem_score_fail(const char* why) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    p.failed = true;
    p.armed  = false;
    std::fprintf(stderr, "kvmem_score: DISABLED (%s)\n", why);
}

// ---- ① chunk 开始：确保容量 + 清零 + 记下"历史块数"--------------------------------------------
//
// history_blocks 取**打分开始那一刻**的 blocks_written()：本 chunk 自己还没进索引（索引在 chunk
// 之后才 append_round），所以它天然是"历史"——这正是检索该对的东西。
inline void kvmem_score_begin(std::int32_t requested_capacity, std::int32_t history_blocks,
                              std::int32_t tail_fill, std::int32_t block_tokens,
                              std::int32_t layers_total, std::int32_t n_heads,
                              std::int32_t n_kv_heads, std::int32_t head_dim,
                              cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.failed) { return; }

    if (requested_capacity <= 0 || history_blocks <= 0 || block_tokens <= 0) { p.armed = false; return; }

    if (p.score == nullptr || p.capacity_blocks < requested_capacity) {
        // 只在**首个 chunk** 分配；prefill 不在 CUDA graph 捕获期（捕获期禁 cudaMalloc，坑表 §3.7）
        if (p.score != nullptr) { (void)cudaFree(p.score); p.score = nullptr; }
        if (p.block_tokens_dev != nullptr) { (void)cudaFree(p.block_tokens_dev); p.block_tokens_dev = nullptr; }
        const std::size_t bytes = static_cast<std::size_t>(requested_capacity) * sizeof(float);
        if (cudaMalloc(reinterpret_cast<void**>(&p.score), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&p.block_tokens_dev),
                       static_cast<std::size_t>(requested_capacity) * sizeof(std::int32_t)) !=
                cudaSuccess) {
            kvmem_score_fail("cudaMalloc");
            return;
        }
        p.capacity_blocks = requested_capacity;
        p.score_host.assign(static_cast<std::size_t>(requested_capacity), 0.0F);
        p.tokens_host.assign(static_cast<std::size_t>(requested_capacity), 0);
        std::fprintf(stderr, "kvmem_score: ARMED capacity_blocks=%d layers_total=%d heads=%d kv_heads=%d "
                             "head_dim=%d\n", requested_capacity, layers_total, n_heads, n_kv_heads, head_dim);
    }

    p.n_blocks     = history_blocks;
    p.block_tokens = block_tokens;
    p.layers_total = layers_total;
    p.n_heads      = n_heads;
    p.n_kv_heads   = n_kv_heads;
    p.head_dim     = head_dim;
    p.armed        = true;
    p.scored_this_chunk = false;
    p.carry_ok = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL", kvmem_score_enabled() ? 36 : 0) > 0 &&
                 p.scored_last_chunk && p.scored_n_blocks > 0 &&
                 p.scored_n_blocks <= history_blocks;
    p.carry_prev_blocks = p.scored_n_blocks;
    p.carry_prev_tokens = p.last_total_tokens;
    p.carry_tokens      = 0;
    p.scored_last_chunk = false;

    // block_tokens：尾块（blocks_written-1）用真实填充，其余用名义块大小
    for (std::int32_t i = 0; i < history_blocks; ++i) {
        p.tokens_host[static_cast<std::size_t>(i)] =
            (i == history_blocks - 1) ? (tail_fill > 0 ? tail_fill : block_tokens) : block_tokens;
    }
    if (cudaMemcpyAsync(p.block_tokens_dev, p.tokens_host.data(),
                        static_cast<std::size_t>(history_blocks) * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        kvmem_score_fail("block_tokens H2D");
        return;
    }
    p.pending_clear = true;   // P1c: cleared (fully or only the new blocks) at the first accumulate
}

// ================================================================================================
// KVMem P4 (kv8): msg-mode query plan, row stash, scoring and fusion.
// ================================================================================================
//
// WHY: tail mode scores the history with the last QUERY_TAIL tokens of the final chunk. A question
// that sits at the START of a long message ("answer from the article below: ... [200K tokens]")
// is never in that tail, and a long question loses its head. msg mode uses the head AND the tail of
// the last user message (template tokens stripped); when that message is a tool result it adds the
// last real user message too, so an agent turn still retrieves the original task.
//
// HOW: the plan is derived from the full prompt ids at the request's first chunk. Each chunk copies
// its rows that fall inside a segment (the content-frame qn, before RoPE -- same point as tail mode)
// into a per-layer stash; the chunk that reaches the plan's end scores every segment from the stash
// against the history (blocks before that chunk), so a head segment computed hundreds of chunks
// earlier is still scored against ALL history. Segments are fused into probe.score_host, which the
// boundary assembly consumes exactly as before (scored_last_chunk / scored_n_blocks).
//
// Token ids are taken from the prompt itself (no tokenizer in the runtime): <|im_start|> = ids[0]
// (every rendered chat starts with it, same trick as kvmem_p3_split_point), the assistant role /
// newline come from the generation prompt, <|im_end|> from the token before it. Only the
// <tool_response> id is configurable (NINFER_TERNARY_KVMEM_TOOL_RESPONSE_ID, default 248066 =
// Qwen3.6 vocab); a wrong id only disables the tool-result special case.

struct KvMemQ4Plan {
    std::int32_t nseg = 0;
    std::int32_t begin[KvMemScoreProbe::kQ4MaxSeg] = {};
    std::int32_t end[KvMemScoreProbe::kQ4MaxSeg]   = {};
    bool tool = false;
    std::int32_t msg_begin = -1, msg_end = -1;    // last user message content
    std::int32_t real_begin = -1, real_end = -1;  // last real user message (tool case)
    // kv9: tool-budget selection of the last real user message (msg in a non-tool request)
    std::int32_t q_nseg = 0;
    std::int32_t q_begin[2] = {}, q_end[2] = {};
    std::int32_t q_cb = -1, q_ce = -1;
};

inline std::uint64_t kvmem_q4_hash(const std::int32_t* ids, std::int32_t b, std::int32_t e) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::int32_t i = b; i < e; ++i) {
        h ^= static_cast<std::uint32_t>(ids[i]);
        h *= 1099511628211ULL;
    }
    return h ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(b)) << 32) ^
           static_cast<std::uint32_t>(e);
}

inline KvMemQ4Plan kvmem_q4_make_plan(const std::int32_t* ids, std::int64_t n) noexcept {
    KvMemQ4Plan plan;
    if (ids == nullptr || n < 8) { return plan; }
    const std::int32_t start = ids[0];
    std::int64_t g = -1;   // generation prompt: the last <|im_start|> within the final 32 tokens
    for (std::int64_t p = n - 1; p >= 1 && p + 32 >= n; --p) {
        if (ids[p] == start) { g = p; break; }
    }
    if (g < 3 || g + 1 >= n) { return plan; }
    const std::int32_t role_asst = ids[g + 1];
    const std::int32_t nl        = (g + 2 < n) ? ids[g + 2] : -1;
    const std::int32_t im_end    = (ids[g - 1] == nl) ? ids[g - 2] : ids[g - 1];
    const std::int32_t tool_id =
        kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_TOOL_RESPONSE_ID", 248066);

    const auto content = [&](std::int64_t s, std::int64_t e, std::int64_t& cb, std::int64_t& ce) {
        cb = s + 2;
        if (cb < e && ids[cb] == nl) { ++cb; }
        ce = e;
        while (ce > cb && (ids[ce - 1] == nl || ids[ce - 1] == im_end)) { --ce; }
    };

    std::int64_t msg_end = g;
    std::int32_t user_role = -1;
    bool need_real = false;
    std::int64_t first_cb = -1, first_ce = -1, real_cb = -1, real_ce = -1;
    for (std::int64_t p = g - 1; p >= 0; --p) {
        if (ids[p] != start) { continue; }
        const std::int32_t role = ids[p + 1];
        if (role != role_asst) {
            std::int64_t cb = 0, ce = 0;
            content(p, msg_end, cb, ce);
            const bool is_tool = cb < ce && ids[cb] == tool_id;
            if (first_cb < 0) {
                first_cb  = cb;
                first_ce  = ce;
                user_role = role;
                plan.tool = is_tool;
                if (!is_tool) { break; }
                need_real = true;
            } else if (need_real && role == user_role && !is_tool && ce > cb) {
                real_cb = cb;
                real_ce = ce;
                break;
            }
        }
        msg_end = p;
    }
    if (first_cb < 0 || first_ce <= first_cb) { return plan; }
    plan.msg_begin = static_cast<std::int32_t>(first_cb);
    plan.msg_end   = static_cast<std::int32_t>(first_ce);
    if (real_cb >= 0) {
        plan.real_begin = static_cast<std::int32_t>(real_cb);
        plan.real_end   = static_cast<std::int32_t>(real_ce);
    }

    const std::int32_t qmax = kvmem_q4_qmax();
    const std::int32_t head_w = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_HEAD", 128);
    const std::int32_t tail_w = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_MSG_TAIL", 128);
    const auto push = [&](std::int64_t b, std::int64_t e) {
        if (e > b && plan.nseg < KvMemScoreProbe::kQ4MaxSeg) {
            plan.begin[plan.nseg] = static_cast<std::int32_t>(b);
            plan.end[plan.nseg]   = static_cast<std::int32_t>(e);
            ++plan.nseg;
        }
    };
    // 3060 round 10 (user-approved, ported from the Swift 1.5 engine): a long message's tail usually
    // is "document ... \n\n question"; start the tail right after the last paragraph (else line)
    // break inside it, so the query is the question rather than the document's last lines (>= 8
    // tokens kept). Only when the vocab check passes (the generation prompt's newline is in the
    // table). NINFER_TERNARY_KVMEM_QUERY_TAIL_ALIGN=0 restores the old cut.
    const bool align = g + 2 < n && kvmem_q4b_is_nl(ids[g + 2]) &&
                       kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_QUERY_TAIL_ALIGN", 1) != 0;
    // Shared by the plan and the real-message Q stash, so a later tool round asks for exactly the
    // rows that were stashed.
    const auto split = [&](std::int64_t cb, std::int64_t ce, std::int32_t budget, const auto& out) {
        const std::int64_t len = ce - cb;
        if (len <= 0 || budget <= 0) { return; }
        if (len <= budget) { out(cb, ce); return; }
        const std::int32_t hw = head_w + tail_w > 0 ? head_w : 1;
        const std::int32_t tw = head_w + tail_w > 0 ? tail_w : 1;
        const std::int32_t h  = static_cast<std::int32_t>(static_cast<std::int64_t>(budget) * hw / (hw + tw));
        const std::int32_t t  = budget - h;
        if (h > 0) { out(cb, cb + h); }
        if (t > 0) {
            std::int64_t tb = ce - t;
            if (align) {
                std::int64_t cut = -1;
                for (std::int64_t q = ce - 9; q >= tb && cut < 0; --q) { if (kvmem_q4b_is_para(ids[q])) { cut = q + 1; } }
                for (std::int64_t q = ce - 9; q >= tb && cut < 0; --q) { if (kvmem_q4b_is_nl(ids[q])) { cut = q + 1; } }
                if (cut > tb) { tb = cut; }
            }
            out(tb, ce);
        }
    };
    const auto add_msg = [&](std::int64_t cb, std::int64_t ce, std::int32_t budget) { split(cb, ce, budget, push); };
    const bool tool_on = kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_TOOL", 1) != 0;
    {   // kv9: which rows of the real user message a later tool round will ask for
        const std::int64_t qcb = plan.tool ? real_cb : first_cb;
        const std::int64_t qce = plan.tool ? real_ce : first_ce;
        const std::int32_t qb  = kvmem_q4_qrows();
        if (qcb >= 0 && qce > qcb && qb > 0) {
            plan.q_cb = static_cast<std::int32_t>(qcb);
            plan.q_ce = static_cast<std::int32_t>(qce);
            const std::int64_t len = qce - qcb;
            const auto qpush = [&](std::int64_t b, std::int64_t e) {
                if (e > b && plan.q_nseg < 2) {
                    plan.q_begin[plan.q_nseg] = static_cast<std::int32_t>(b);
                    plan.q_end[plan.q_nseg]   = static_cast<std::int32_t>(e);
                    ++plan.q_nseg;
                }
            };
            (void)len;
            split(qcb, qce, qb, qpush);
        }
    }
    if (plan.tool && real_cb >= 0 && tool_on) {
        add_msg(first_cb, first_ce, qmax / 2);
        add_msg(real_cb, real_ce, qmax - qmax / 2);
    } else {
        add_msg(first_cb, first_ce, qmax);
    }
    return plan;
}

// Called for every non-replay prefill chunk, right after kvmem_score_begin (also when the history is
// still empty -- the first chunk of a fresh sequence may hold the head of the last user message).
inline void kvmem_q4_begin_chunk(const std::int32_t* ids, std::int64_t n, std::int32_t chunk_base,
                                 std::int32_t history_blocks, std::int32_t capacity_blocks,
                                 std::int32_t layers_total, std::int32_t n_heads,
                                 std::int32_t head_dim, cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled() || !kvmem_q4_track()) { return; }
    KvMemScoreProbe& p  = kvmem_score_probe();
    KvMemQ4Request& rq  = kvmem_q4_request();
    const bool continues = rq.req_start >= 0 && rq.prompt_tokens == n && rq.last_end == chunk_base;
    p.q4_in_chunk       = true;
    p.q4_chunk_base     = chunk_base;
    p.q4_chunk_rows     = 0;
    p.q4_complete_chunk = false;
    p.q4_cleared        = false;
    p.q4_hist_blocks    = history_blocks;
    if (!continues) {
        rq.prompt_tokens = n;
        rq.req_start     = chunk_base;
        ++rq.serial;
        p.q4_scored = false;
        if (kvmem_q4_protect_new() && !kvmem_q4_protect_v1()) { kvmem_q4b_make_protect(ids, n); }   // KVMem P4b (kv8b)
        if (kvmem_q4_msg_mode()) {
            const KvMemQ4Plan plan = kvmem_q4_make_plan(ids, n);
            bool same = plan.nseg == p.q4_nseg && plan.nseg > 0;
            std::uint64_t hash[KvMemScoreProbe::kQ4MaxSeg] = {};
            for (std::int32_t s = 0; s < plan.nseg; ++s) {
                hash[s] = kvmem_q4_hash(ids, plan.begin[s], plan.end[s]);
                same = same && plan.begin[s] == p.q4_seg_begin[s] && plan.end[s] == p.q4_seg_end[s] &&
                       hash[s] == p.q4_seg_hash[s];
            }
            p.q4_active   = plan.nseg > 0;
            p.q4_tool     = plan.tool;
            p.q4_nseg     = plan.nseg;
            p.q4_rows     = 0;
            p.q4_plan_end = 0;
            for (std::int32_t s = 0; s < plan.nseg; ++s) {
                p.q4_seg_begin[s] = plan.begin[s];
                p.q4_seg_end[s]   = plan.end[s];
                p.q4_seg_hash[s]  = hash[s];
                p.q4_seg_row0[s]  = p.q4_rows;
                p.q4_rows += plan.end[s] - plan.begin[s];
                p.q4_plan_end = std::max(p.q4_plan_end, plan.end[s]);
                if (!same) {   // rows of an identical earlier plan (same tokens, same positions) stay usable
                    p.q4_have_lo[s] = plan.end[s];
                    p.q4_have_hi[s] = plan.begin[s];
                }
            }
            std::fprintf(stderr,
                         "[kvmem-q4] plan prompt=%lld req_start=%d segs=%d tool=%d rows=%d "
                         "msg=[%d,%d) real=[%d,%d) reuse_stash=%d",
                         static_cast<long long>(n), chunk_base, plan.nseg, plan.tool ? 1 : 0,
                         p.q4_rows, plan.msg_begin, plan.msg_end, plan.real_begin, plan.real_end,
                         same ? 1 : 0);
            for (std::int32_t s = 0; s < plan.nseg; ++s) {
                std::fprintf(stderr, " [%d,%d)", plan.begin[s], plan.end[s]);
            }
            std::fprintf(stderr, "\n");
            // kv9 (item 1): the real user message's own stash (survives tool rounds)
            p.q4q_inject = false;
            if (plan.q_nseg > 0) {
                const std::uint64_t qh = kvmem_q4_hash(ids, plan.q_cb, plan.q_ce);
                bool qsame = qh == p.q4q_hash && plan.q_cb == p.q4q_cb && plan.q_ce == p.q4q_ce &&
                             plan.q_nseg == p.q4q_nseg;
                for (std::int32_t k = 0; qsame && k < plan.q_nseg; ++k) {
                    qsame = plan.q_begin[k] == p.q4q_begin[k] && plan.q_end[k] == p.q4q_end[k];
                }
                if (!qsame) {
                    p.q4q_cb = plan.q_cb; p.q4q_ce = plan.q_ce; p.q4q_hash = qh; p.q4q_nseg = plan.q_nseg;
                    std::int32_t r = 0;
                    for (std::int32_t k = 0; k < plan.q_nseg; ++k) {
                        p.q4q_begin[k] = plan.q_begin[k]; p.q4q_end[k] = plan.q_end[k]; p.q4q_row0[k] = r;
                        r += plan.q_end[k] - plan.q_begin[k];
                        p.q4q_have_lo[k] = plan.q_end[k]; p.q4q_have_hi[k] = plan.q_begin[k];
                    }
                }
                std::int32_t qhave = 0;
                for (std::int32_t k = 0; k < p.q4q_nseg; ++k) {
                    qhave += std::max(0, p.q4q_have_hi[k] - p.q4q_have_lo[k]);
                }
                p.q4q_inject = plan.tool && qhave > 0;
                std::fprintf(stderr, "[kvmem-q9] real-msg stash [%d,%d) segs=%d same=%d have=%d inject=%d\n",
                             plan.q_cb, plan.q_ce, plan.q_nseg, qsame ? 1 : 0, qhave, p.q4q_inject ? 1 : 0);
            }
            // Rows before req_start can only come from the stash (earlier request, same plan). If no
            // segment row is reachable at all (e.g. the message came from a prefix cache hit), this
            // request falls back to tail mode.
            std::int32_t reachable = 0;
            for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
                reachable += std::max(0, p.q4_seg_end[s] - std::max(p.q4_seg_begin[s], chunk_base));
                if (p.q4_have_lo[s] < p.q4_have_hi[s]) { reachable += p.q4_have_hi[s] - p.q4_have_lo[s]; }
            }
            if (p.q4q_inject) { reachable += 1; }   // kv9: real-message rows come from the Q stash
            if (p.q4_active && reachable == 0) {
                p.q4_active = false;
                std::fprintf(stderr, "[kvmem-q4] message rows not reachable (before req_start, not stashed) => tail mode\n");
            }
        } else {
            p.q4_active = false;
        }
    }
    if (!kvmem_q4_msg_mode() || !p.q4_active || p.failed) {
        // tail mode runs this chunk: never let its P1c carry build on a device buffer that msg mode
        // did not write.
        if (p.q4_owns_scores) { p.carry_ok = false; p.q4_owns_scores = false; }
        return;
    }

    // stash + per-segment score buffers (grow only; prefill is never inside a CUDA graph capture)
    const std::int32_t row_elems = n_heads * head_dim;
    if (p.q4_stash == nullptr ||
        p.q4_stash_rows < std::max(p.q4_rows, kvmem_q4_qmax()) + kvmem_q4_qrows() ||
        p.q4_layers != layers_total || p.q4_row_elems != row_elems) {
        if (p.q4_stash != nullptr) {
            if (p.q4_host_stash) { p.q4_host_stash.reset(); }
            else { (void)cudaFree(p.q4_stash); }
            p.q4_stash = nullptr;
        }
        // kv9: main rows fixed at >= kQ4MainRows (no re-allocation between plans) + Q-stash rows
        const std::int32_t rows_cap =
            ((std::max(p.q4_rows, kvmem_q4_qmax()) + 63) / 64) * 64 + kvmem_q4_qrows();
        const std::size_t bytes = static_cast<std::size_t>(layers_total) * rows_cap *
                                  static_cast<std::size_t>(row_elems) * 2U;
        if (kvmem_cpu_retrieval_enabled()) {
            try {
                p.q4_host_stash.emplace(bytes);
                p.q4_stash = p.q4_host_stash->data();
            } catch (...) {
                kvmem_score_fail("q4 host stash allocation");
                return;
            }
        } else if (cudaMalloc(&p.q4_stash, bytes) != cudaSuccess) {
            p.q4_stash = nullptr;
            kvmem_score_fail("q4 stash cudaMalloc");
            return;
        }
        p.q4_stash_rows = rows_cap;
        p.q4_layers     = layers_total;
        p.q4_row_elems  = row_elems;
        for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
            p.q4_have_lo[s] = p.q4_seg_end[s];
            p.q4_have_hi[s] = p.q4_seg_begin[s];
        }
        for (std::int32_t k = 0; k < p.q4q_nseg; ++k) {
            p.q4q_have_lo[k] = p.q4q_end[k];
            p.q4q_have_hi[k] = p.q4q_begin[k];
        }
        p.q4q_inject = false;
        std::fprintf(stderr, "[kvmem-q4] stash allocated layers=%d rows=%d row_elems=%d bytes=%zu storage=%s\n",
                     layers_total, rows_cap, row_elems, bytes,
                     kvmem_cpu_retrieval_enabled() ? "host" : "device");
    }
    if (p.q4_score == nullptr || p.q4_score_cap < capacity_blocks) {
        if (p.q4_score != nullptr) { (void)cudaFree(p.q4_score); p.q4_score = nullptr; }
        if (cudaMalloc(reinterpret_cast<void**>(&p.q4_score),
                       static_cast<std::size_t>(KvMemScoreProbe::kQ4MaxSeg) * capacity_blocks *
                           sizeof(float)) != cudaSuccess) {
            p.q4_score = nullptr;
            kvmem_score_fail("q4 score cudaMalloc");
            return;
        }
        p.q4_score_cap = capacity_blocks;
    }
    if (kvmem_q4_eq10_exclude() && (p.q4_excl == nullptr || p.q4_excl_cap < capacity_blocks)) {
        if (p.q4_excl != nullptr) { (void)cudaFree(p.q4_excl); p.q4_excl = nullptr; }
        if (cudaMalloc(reinterpret_cast<void**>(&p.q4_excl), static_cast<std::size_t>(capacity_blocks)) !=
            cudaSuccess) {
            p.q4_excl = nullptr;
            kvmem_score_fail("q4 exclude cudaMalloc");
            return;
        }
        p.q4_excl_cap = capacity_blocks;
    }
    // kv9 (item 1): tool round -- copy the real user message's rows from the Q stash into its plan
    // segments (same absolute positions, same tokens), so the question keeps steering retrieval.
    if (p.q4q_inject) {
        if (kvmem_cpu_retrieval_enabled() && cudaStreamSynchronize(stream) != cudaSuccess) {
            kvmem_score_fail("q9 host injection sync");
            return;
        }
        p.q4q_inject = false;
        const std::size_t row_bytes = static_cast<std::size_t>(p.q4_row_elems) * 2U;
        const std::size_t pitch     = static_cast<std::size_t>(p.q4_stash_rows) * row_bytes;
        const std::int32_t qbase    = p.q4_stash_rows - kvmem_q4_qrows();
        std::int32_t injected = 0;
        for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
            if (p.q4_have_lo[s] < p.q4_have_hi[s]) { continue; }
            for (std::int32_t k = 0; k < p.q4q_nseg; ++k) {
                if (p.q4q_begin[k] != p.q4_seg_begin[s] || p.q4q_end[k] != p.q4_seg_end[s]) { continue; }
                const std::int32_t lo = p.q4q_have_lo[k], hi = p.q4q_have_hi[k];
                if (lo >= hi) { break; }
                char* base = static_cast<char*>(p.q4_stash);
                const std::size_t src = static_cast<std::size_t>(qbase + p.q4q_row0[k] + lo - p.q4q_begin[k]) * row_bytes;
                const std::size_t dst = static_cast<std::size_t>(p.q4_seg_row0[s] + lo - p.q4_seg_begin[s]) * row_bytes;
                if (kvmem_cpu_retrieval_enabled()) {
                    for (std::int32_t l = 0; l < p.q4_layers; ++l) {
                        std::memcpy(base + static_cast<std::size_t>(l) * pitch + dst,
                                    base + static_cast<std::size_t>(l) * pitch + src,
                                    static_cast<std::size_t>(hi - lo) * row_bytes);
                    }
                } else if (cudaMemcpy2DAsync(base + dst, pitch, base + src, pitch,
                                      static_cast<std::size_t>(hi - lo) * row_bytes,
                                      static_cast<std::size_t>(p.q4_layers), cudaMemcpyDeviceToDevice,
                                      stream) != cudaSuccess) {
                    kvmem_score_fail("q9 inject copy");
                    return;
                }
                p.q4_have_lo[s] = lo;
                p.q4_have_hi[s] = hi;
                injected += hi - lo;
                break;
            }
        }
        std::fprintf(stderr, "[kvmem-q9] injected real-msg rows=%d\n", injected);
    }
    // msg mode owns this request's scoring: the tail-mode accumulate/finish stay idle, and scores
    // produced by an earlier chunk of THIS request stay "fresh" for the boundary assembly.
    p.armed             = false;
    p.scored_last_chunk = p.q4_scored;
}

// Rows of segment s usable for scoring = stash rows from earlier chunks + this chunk's rows.
inline bool kvmem_q4_avail(const KvMemScoreProbe& p, std::int32_t s, std::int32_t& lo,
                           std::int32_t& hi) noexcept {
    std::int32_t hl = p.q4_have_lo[s], hh = p.q4_have_hi[s];
    const std::int32_t cl = std::max(p.q4_seg_begin[s], p.q4_chunk_base);
    const std::int32_t ch = std::min(p.q4_seg_end[s], p.q4_chunk_base + p.q4_chunk_rows);
    if (cl < ch) {
        if (hl < hh && hh >= cl && hl <= ch) { hl = std::min(hl, cl); hh = std::max(hh, ch); }
        else { hl = cl; hh = ch; }
    }
    lo = hl;
    hi = hh;
    return hl < hh;
}

inline void kvmem_q4_accumulate(std::int32_t fidx, const void* q, std::int32_t rows,
                                cudaStream_t stream) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.q4_in_chunk || !p.q4_active || p.failed || rows <= 0 || q == nullptr) { return; }
    if (fidx < 0 || fidx >= p.q4_layers || p.q4_stash == nullptr) { return; }
    p.q4_chunk_rows = rows;
    const std::size_t row_bytes = static_cast<std::size_t>(p.q4_row_elems) * 2U;
    const std::int32_t base     = p.q4_chunk_base;
    char* stash_l = static_cast<char*>(p.q4_stash) +
                    static_cast<std::size_t>(fidx) * p.q4_stash_rows * row_bytes;
    for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
        const std::int32_t lo = std::max(p.q4_seg_begin[s], base);
        const std::int32_t hi = std::min(p.q4_seg_end[s], base + rows);
        if (lo >= hi) { continue; }
        if (cudaMemcpyAsync(stash_l + static_cast<std::size_t>(p.q4_seg_row0[s] + lo - p.q4_seg_begin[s]) * row_bytes,
                            static_cast<const char*>(q) + static_cast<std::size_t>(lo - base) * row_bytes,
                            static_cast<std::size_t>(hi - lo) * row_bytes,
                            kvmem_cpu_retrieval_enabled() ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            kvmem_score_fail("q4 stash copy");
            return;
        }
    }
    {   // kv9 (item 1): also keep the real user message's rows in the Q stash
        char* qstash_l = stash_l + static_cast<std::size_t>(p.q4_stash_rows - kvmem_q4_qrows()) * row_bytes;
        for (std::int32_t k = 0; k < p.q4q_nseg; ++k) {
            const std::int32_t lo = std::max(p.q4q_begin[k], base);
            const std::int32_t hi = std::min(p.q4q_end[k], base + rows);
            if (lo >= hi) { continue; }
            if (cudaMemcpyAsync(qstash_l + static_cast<std::size_t>(p.q4q_row0[k] + lo - p.q4q_begin[k]) * row_bytes,
                                static_cast<const char*>(q) + static_cast<std::size_t>(lo - base) * row_bytes,
                                static_cast<std::size_t>(hi - lo) * row_bytes,
                            kvmem_cpu_retrieval_enabled() ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                kvmem_score_fail("q9 stash copy");
                return;
            }
        }
    }
    // Score once per request: in the first chunk that has history and reaches the plan's end.
    if (p.q4_scored || p.q4_hist_blocks <= 0 || p.n_blocks <= 0 || p.q4_score == nullptr ||
        base + rows < p.q4_plan_end) {
        return;
    }
    ops::MeanKIndex* index = ops::mean_k_index_for(p.layers_total, p.n_kv_heads, p.head_dim,
                                                  p.capacity_blocks);
    if (index == nullptr || index->blocks_written() <= 0 || fidx >= index->layers()) { return; }
    const Tensor& sums = index->layer_sums(fidx);
    if (sums.data == nullptr) { return; }
    if (!p.q4_cleared) {
        if (cudaMemsetAsync(p.q4_score, 0,
                            static_cast<std::size_t>(KvMemScoreProbe::kQ4MaxSeg) * p.q4_score_cap *
                                sizeof(float),
                            stream) != cudaSuccess) {
            kvmem_score_fail("q4 score memset");
            return;
        }
        p.q4_cleared = true;
        p.q4_excl_n  = 0;
        if (kvmem_q4_eq10_exclude() && p.q4_excl != nullptr && p.q4_excl_cap >= p.n_blocks &&
            p.block_tokens > 0) {
            const std::int32_t nb = p.n_blocks;
            p.q4_excl_host.assign(static_cast<std::size_t>(nb), 0U);
            const std::int32_t sink_b = std::min(
                nb, kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096) / p.block_tokens);
            for (std::int32_t b = 0; b < sink_b; ++b) { p.q4_excl_host[static_cast<std::size_t>(b)] = 1U; }
            if (kvmem_q4_protect_new() && !kvmem_q4_protect_v1()) {
                const KvMemQ4Protect& pr = kvmem_q4_protect();
                for (int r = 0; r < pr.n; ++r) {
                    if (pr.e[r] <= pr.b[r] || pr.b[r] < 0) { continue; }
                    const std::int32_t b1 = std::min(nb - 1, (pr.e[r] - 1) / p.block_tokens);
                    for (std::int32_t b = pr.b[r] / p.block_tokens; b <= b1; ++b) {
                        p.q4_excl_host[static_cast<std::size_t>(b)] = 1U;
                    }
                }
            }
            std::int32_t cnt = 0;
            for (const std::uint8_t v : p.q4_excl_host) { cnt += v != 0U ? 1 : 0; }
            if (cnt > 0 && cnt < nb) {
                if (cudaMemcpyAsync(p.q4_excl, p.q4_excl_host.data(), static_cast<std::size_t>(nb),
                                    cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                    kvmem_score_fail("q4 exclude H2D");
                    return;
                }
                p.q4_excl_n = cnt;
            }
            std::fprintf(stderr, "[kvmem-q4] eq10 exclude n_blocks=%d sink=%d excluded=%d%s\n", nb, sink_b,
                         cnt, (cnt > 0 && cnt < nb) ? "" : " (off: nothing to exclude / nothing left)");
        }
    }
    if (kvmem_cpu_retrieval_enabled()) {
        for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
            std::int32_t lo = 0, hi = 0;
            p.q4_used[s] = kvmem_q4_avail(p, s, lo, hi) ? hi - lo : 0;
        }
        p.q4_complete_chunk = true;
        return; // Host scoring runs once, after all layers' query copies complete.
    }
    for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
        std::int32_t lo = 0, hi = 0;
        if (!kvmem_q4_avail(p, s, lo, hi)) { p.q4_used[s] = 0; continue; }
        KvMemRetrieveConfig cfg;
        cfg.n_layers          = 1;
        cfg.n_layers_total    = p.layers_total;
        cfg.n_query_tokens    = hi - lo;
        cfg.n_heads           = p.n_heads;
        cfg.n_kv_heads        = p.n_kv_heads;
        cfg.head_dim          = p.head_dim;
        cfg.n_blocks          = p.n_blocks;
        cfg.q_layer_stride    = hi - lo;
        cfg.kbar_layer_stride = p.capacity_blocks;
        cfg.q_token_begin     = 0;
        cfg.budget_blocks     = 0;
        cfg.sink_blocks       = 0;
        cfg.recent_blocks     = 0;
        cfg.mask_mode         = KvMemRetrieveMaskMode::Never;
        cfg.dtype             = KvMemRetrieveDtype::BF16;
        const void* qs = stash_l + static_cast<std::size_t>(p.q4_seg_row0[s] + lo - p.q4_seg_begin[s]) * row_bytes;
        const cudaError_t st = ops::kvmem_retrieve_scores(
            cfg, p.q4_score + static_cast<std::size_t>(s) * p.q4_score_cap, qs,
            static_cast<const float*>(sums.data), p.block_tokens_dev, stream,
            p.q4_excl_n > 0 ? p.q4_excl : nullptr);
        if (st != cudaSuccess) {
            std::fprintf(stderr, "[kvmem-q4] retrieve failed layer=%d seg=%d: %s\n", fidx, s,
                         cudaGetErrorString(st));
            kvmem_score_fail("q4 retrieve");
            return;
        }
        p.q4_used[s] = hi - lo;
    }
    p.q4_complete_chunk = true;
}

// Chunk end (inside kvmem_score_finish): update request/stash bookkeeping; if this chunk scored,
// fuse the segments into score_host and mark it fresh for the boundary assembly.
inline void kvmem_q4_finish(cudaStream_t stream) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.q4_in_chunk) { return; }
    p.q4_in_chunk = false;
    KvMemQ4Request& rq = kvmem_q4_request();
    if (p.q4_chunk_rows > 0) { rq.last_end = p.q4_chunk_base + p.q4_chunk_rows; }
    if (!p.q4_active) { return; }
    for (std::int32_t s = 0; s < p.q4_nseg; ++s) {
        std::int32_t lo = 0, hi = 0;
        if (kvmem_q4_avail(p, s, lo, hi)) { p.q4_have_lo[s] = lo; p.q4_have_hi[s] = hi; }
    }
    if (p.q4_chunk_rows > 0 && p.q4_stash != nullptr) {   // kv9: Q-stash rows written by this chunk
        for (std::int32_t k = 0; k < p.q4q_nseg; ++k) {
            const std::int32_t cl = std::max(p.q4q_begin[k], p.q4_chunk_base);
            const std::int32_t ch = std::min(p.q4q_end[k], p.q4_chunk_base + p.q4_chunk_rows);
            if (cl >= ch) { continue; }
            std::int32_t& hl = p.q4q_have_lo[k];
            std::int32_t& hh = p.q4q_have_hi[k];
            if (hl < hh && hh >= cl && hl <= ch) { hl = std::min(hl, cl); hh = std::max(hh, ch); }
            else { hl = cl; hh = ch; }
        }
    }
    if (!p.q4_complete_chunk || p.failed) { return; }
    p.q4_complete_chunk = false;
    const std::int32_t nb = p.n_blocks;
    const std::int32_t ns = p.q4_nseg;
    p.q4_host.assign(static_cast<std::size_t>(ns) * nb, 0.0F);
    if (kvmem_cpu_retrieval_enabled()) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) { kvmem_score_fail("q4 host sync"); return; }
        const auto started = std::chrono::steady_clock::now();
        try {
            MeanKIndex* index = mean_k_index_for(p.layers_total, p.n_kv_heads, p.head_dim,
                                               p.capacity_blocks);
            if (index == nullptr || !index->host_resident()) {
                kvmem_score_fail("q4 missing host index");
                return;
            }
            std::vector<KvMemCpuQuery> queries(static_cast<std::size_t>(ns));
            const auto* rows = static_cast<const std::uint16_t*>(p.q4_stash);
            for (std::int32_t seg = 0; seg < ns; ++seg) {
                std::int32_t lo = 0, hi = 0;
                if (!kvmem_q4_avail(p, seg, lo, hi)) { continue; }
                queries[seg] = {rows + static_cast<std::size_t>(p.q4_seg_row0[seg] + lo -
                               p.q4_seg_begin[seg]) * p.q4_row_elems,
                               static_cast<std::uint32_t>(hi - lo),
                               static_cast<std::size_t>(p.q4_stash_rows)};
            }
            std::vector<std::vector<float>> scores;
            const std::span<const std::uint8_t> exclude = p.q4_excl_n > 0
                ? std::span<const std::uint8_t>(p.q4_excl_host.data(), nb)
                : std::span<const std::uint8_t>{};
            kvmem_cpu_scores(index->host_data(), index->layer_elements(),
                             p.layers_total, p.n_kv_heads, p.n_heads, p.head_dim,
                             std::span<const std::int32_t>(p.tokens_host.data(), nb),
                             exclude, queries, scores);
            for (std::int32_t seg = 0; seg < ns; ++seg) {
                std::copy(scores[seg].begin(), scores[seg].end(),
                          p.q4_host.begin() + static_cast<std::size_t>(seg) * nb);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[kvmem-cpu] scoring failed: %s\n", e.what());
            kvmem_score_fail("q4 CPU scoring");
            return;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr, "[kvmem-cpu] score mode=msg blocks=%d ms=%.3f\n", nb, ms);
    } else {
        for (std::int32_t s = 0; s < ns; ++s) {
            if (cudaMemcpyAsync(p.q4_host.data() + static_cast<std::size_t>(s) * nb,
                                p.q4_score + static_cast<std::size_t>(s) * p.q4_score_cap,
                                static_cast<std::size_t>(nb) * sizeof(float), cudaMemcpyDeviceToHost,
                                stream) != cudaSuccess) {
                kvmem_score_fail("q4 score D2H");
                return;
            }
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) { kvmem_score_fail("q4 sync"); return; }
    }

    std::int32_t active = 0, used_total = 0;
    for (std::int32_t s = 0; s < ns; ++s) {
        if (p.q4_used[s] > 0) { ++active; used_total += p.q4_used[s]; }
    }
    if (active == 0) {
        std::fprintf(stderr, "[kvmem-q4] no query rows available (prefix-cached message?) => no scores\n");
        return;
    }
    // per-segment normalised mass (each segment sums to 1 over all history blocks)
    std::vector<float> mean(static_cast<std::size_t>(nb), 0.0F);
    for (std::int32_t s = 0; s < ns; ++s) {
        if (p.q4_used[s] <= 0) { continue; }
        const float w = 1.0F / (static_cast<float>(p.q4_used[s]) * static_cast<float>(active));
        const float* src = p.q4_host.data() + static_cast<std::size_t>(s) * nb;
        for (std::int32_t b = 0; b < nb; ++b) { mean[static_cast<std::size_t>(b)] += src[b] * w; }
    }
    const bool sum_mode = kvmem_q4_merge_sum() || active == 1;
    std::vector<std::int32_t> top(static_cast<std::size_t>(ns) * 3, -1);
    std::vector<std::int32_t> best(static_cast<std::size_t>(nb), nb);
    std::vector<std::int32_t> order(static_cast<std::size_t>(nb));
    for (std::int32_t s = 0; s < ns; ++s) {
        if (p.q4_used[s] <= 0) { continue; }
        const float* src = p.q4_host.data() + static_cast<std::size_t>(s) * nb;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [src](std::int32_t a, std::int32_t b) { return src[a] > src[b]; });
        for (std::int32_t r = 0; r < nb; ++r) {
            std::int32_t& bb = best[static_cast<std::size_t>(order[static_cast<std::size_t>(r)])];
            bb = std::min(bb, r);
        }
        for (std::int32_t r = 0; r < 3 && r < nb; ++r) {
            top[static_cast<std::size_t>(s) * 3 + r] = order[static_cast<std::size_t>(r)];
        }
    }
    for (std::int32_t b = 0; b < nb; ++b) {
        const float m = std::min(1.0F, mean[static_cast<std::size_t>(b)]);
        p.score_host[static_cast<std::size_t>(b)] =
            sum_mode ? m : static_cast<float>(nb - best[static_cast<std::size_t>(b)]) + 0.5F * m;
    }
    p.scored_n_blocks   = nb;
    p.scored_last_chunk = true;
    p.q4_scored         = true;
    p.q4_owns_scores    = true;
    p.last_total_tokens = used_total;

    std::fprintf(stderr, "[kvmem-q4] scored n_blocks=%d merge=%s query_tokens=%d tool=%d segs:", nb,
                 sum_mode ? "sum" : "split", used_total, p.q4_tool ? 1 : 0);
    for (std::int32_t s = 0; s < ns; ++s) {
        std::fprintf(stderr, " [%d,%d) used=%d top=%d,%d,%d;", p.q4_seg_begin[s], p.q4_seg_end[s],
                     p.q4_used[s], top[static_cast<std::size_t>(s) * 3],
                     top[static_cast<std::size_t>(s) * 3 + 1], top[static_cast<std::size_t>(s) * 3 + 2]);
    }
    std::fprintf(stderr, "\n");

    // Same evidence line as tail mode (KEPT = what a budget/sink selection over these scores keeps).
    const std::int32_t budget_tokens = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
    const std::int32_t sink_tokens   = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);
    KvMemSelectConfig scfg;
    scfg.total_blocks  = nb;
    scfg.budget_blocks = budget_tokens / p.block_tokens;
    scfg.sink_blocks   = sink_tokens / p.block_tokens;
    scfg.recent_blocks = 0;
    const KvMemSelectResult sel = ops::kvmem_select_blocks(scfg, p.score_host.data(), 0);
    std::fprintf(stderr, "kvmem_score: KEPT");
    for (std::size_t i = 0; i < sel.kept.size(); ++i) {
        std::fprintf(stderr, " %d", static_cast<int>(sel.kept[i]));
    }
    std::fprintf(stderr, "\nkvmem_score: SELECT label=q4 n_blocks=%d kept=%zu query_tokens=%d\n", nb,
                 sel.kept.size(), used_total);
}

// ---- ② 每层一次：把该层的 query 行累加到 score 上 ----------------------------------------------
//
// 调用点必须在 `ops::rope(qn, ...)` **之前**：rope 会原地覆盖 qn，之后就没有内容帧的 query 了。
inline void kvmem_score_accumulate(std::int32_t fidx, const void* q, std::int32_t n_query_tokens,
                                   std::int32_t chunk_tokens, cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.q4_in_chunk) {   // KVMem P4 (kv8)
        p.q4_chunk_rows = chunk_tokens > 0 ? chunk_tokens : n_query_tokens;
        if (p.q4_active && kvmem_q4_msg_mode()) {
            kvmem_q4_accumulate(fidx, q, p.q4_chunk_rows, stream);
            return;
        }
    }
    if (!p.armed || p.failed) { return; }
    if (n_query_tokens <= 0 || p.n_blocks <= 0) { return; }

    // ★ query span（2026-09-26 语义档）：NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=N ⇒ 只把本 chunk 的
    //   **最后 N 个 token** 当检索 query（官方 kvmem_set_query_span 的等价物）。默认 0 = 整 chunk
    //   （旧行为，逐位不变）。MAXQ 的门限看 **span 长度**、不看 chunk 长度。
    const std::int32_t query_tail = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL", kvmem_score_enabled() ? 36 : 0);
    std::int32_t span_begin  = 0;
    std::int32_t span_tokens = (chunk_tokens > 0 && chunk_tokens < n_query_tokens) ? chunk_tokens
                                                                                  : n_query_tokens;
    if (query_tail > 0 && chunk_tokens > 0) {
        const std::int32_t tail = query_tail < chunk_tokens ? query_tail : chunk_tokens;
        span_begin  = chunk_tokens - tail;
        span_tokens = tail;
    }
    const std::int32_t maxq = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_MAXQ", 256);
    if (span_tokens > maxq) {   // query 太长：不打分（它也不是检索 query）
        if (!p.logged_once) {
            p.logged_once = true;
            std::fprintf(stderr, "kvmem_score: skip long query span tokens=%d (> MAXQ=%d); chunk=%d begin=%d\n",
                         span_tokens, maxq, chunk_tokens, span_begin);
        }
        return;
    }
    if (p.pending_clear) {
        p.pending_clear  = false;
        const bool carry = p.carry_ok && query_tail > 0 && chunk_tokens > 0 &&
                           chunk_tokens < query_tail;
        const std::int32_t clear_from = carry ? p.carry_prev_blocks : 0;
        if (kvmem_cpu_retrieval_enabled()) {
            std::fill(p.score_host.begin() + clear_from,
                      p.score_host.begin() + p.n_blocks, 0.0F);
        } else if (clear_from < p.n_blocks &&
            cudaMemsetAsync(p.score + clear_from, 0,
                            static_cast<std::size_t>(p.n_blocks - clear_from) * sizeof(float),
                            stream) != cudaSuccess) {
            kvmem_score_fail("score memset");
            return;
        }
        p.carry_tokens = carry ? p.carry_prev_tokens : 0;
        if (carry) {
            std::fprintf(stderr, "kvmem_score: carry chunk=%d onto previous query_tokens=%d\n",
                         chunk_tokens, p.carry_tokens);
        }
    }
    p.chunk_tokens      = chunk_tokens;   // live 平面的行数（= q_layer_stride）
    p.query_span_begin  = span_begin;
    p.query_span_tokens = span_tokens;    // 供 finish 的 oracle 自检用（正确口径 = span 的 token 数）

    ops::MeanKIndex* index = ops::mean_k_index_for(p.layers_total, p.n_kv_heads, p.head_dim,
                                                  p.capacity_blocks);
    if (index == nullptr || index->blocks_written() <= 0) { return; }
    if (fidx < 0 || fidx >= index->layers()) { return; }

    if (kvmem_cpu_retrieval_enabled()) {
        try {
            if (!p.tail_host_stash || p.tail_host_rows < span_tokens) {
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    kvmem_score_fail("tail host resize sync"); return;
                }
                p.tail_host_stash.emplace(static_cast<std::size_t>(p.layers_total) *
                    span_tokens * p.n_heads * p.head_dim * sizeof(std::uint16_t));
                p.tail_host_rows = span_tokens;
            }
        } catch (...) { kvmem_score_fail("tail host stash allocation"); return; }
        const std::size_t row_bytes = static_cast<std::size_t>(p.n_heads) * p.head_dim * 2U;
        auto* dst = static_cast<char*>(p.tail_host_stash->data()) +
                    static_cast<std::size_t>(fidx) * p.tail_host_rows * row_bytes;
        if (cudaMemcpyAsync(dst, static_cast<const char*>(q) +
                            static_cast<std::size_t>(span_begin) * row_bytes,
                            static_cast<std::size_t>(span_tokens) * row_bytes,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            kvmem_score_fail("tail query D2H"); return;
        }
        p.scored_this_chunk = true;
        return;
    }

    const Tensor& sums = index->layer_sums(fidx);
    if (sums.data == nullptr) { return; }

    KvMemRetrieveConfig cfg;
    cfg.n_layers          = 1;                      // 逐层调用
    cfg.n_layers_total    = p.layers_total;         // head_w = 1/(总层数×query 头数)
    cfg.n_query_tokens    = span_tokens;
    cfg.n_heads           = p.n_heads;
    cfg.n_kv_heads        = p.n_kv_heads;
    cfg.head_dim          = p.head_dim;
    cfg.n_blocks          = p.n_blocks;
    cfg.q_layer_stride    = chunk_tokens;           // live plane 的 token 数（>= n_query_tokens）
    cfg.kbar_layer_stride = p.capacity_blocks;      // >= n_blocks
    cfg.q_token_begin     = span_begin;
    cfg.budget_blocks     = 0;                      // 掩码只用在这里；探针不设带（Official 规则见下）
    cfg.sink_blocks       = 0;
    cfg.recent_blocks     = 0;
    cfg.mask_mode         = KvMemRetrieveMaskMode::Never;   // 探针要"全历史"的原始分数，不套带掩码
    cfg.dtype             = KvMemRetrieveDtype::BF16;

    const cudaError_t status =
        ops::kvmem_retrieve_scores(cfg, p.score, q, static_cast<const float*>(sums.data),
                                   p.block_tokens_dev, stream);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "kvmem_score: launch failed layer=%d: %s\n", fidx, cudaGetErrorString(status));
        kvmem_score_fail("launch");
        return;
    }
    p.scored_this_chunk = true;
}

// ---- ③ chunk 收尾：D2H + 选块 + 一行证据日志 ---------------------------------------------------
//
// 放在与本文件同类的位置：prefill chunk 的尾巴、**任何捕获之外**（现有 kvmem_index 探针就在那儿
// 做 cudaMemcpyAsync + cudaStreamSynchronize，理由照抄它：一次同步换一个外部可核的数）。
//
// 免 oracle 自检：契约保证 `sum_b score[b] == n_query_tokens`（与带掩码与否无关）⇒ sum_score 就是
// 内置 oracle。对不上说明打分器没跑、跑了半截、或几何配错 —— 三种都得当场喊出来。
inline void kvmem_score_finish(const char* label, std::int32_t query_tokens,
                               cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    kvmem_q4_finish(stream);   // KVMem P4 (kv8)
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.armed || p.failed || p.n_blocks <= 0) { return; }
    p.armed = false;
    if (!p.scored_this_chunk) { return; }   // 本 chunk 没打过（如长 ingest）⇒ 不打印全零假日志
    // ★ 出处登记（2026-09-26 语义档的准入闸）：这份分数描述块 [0, n_blocks)，且来自**最近一个
    //   chunk 的 query**。语义档只在 scored_last_chunk 为真时才用它装窗（宁可退压力档，也不装
    //   一个来路不明的窗）。
    p.scored_n_blocks   = p.n_blocks;
    p.scored_last_chunk = true;
    const std::int32_t q_used =
        p.carry_tokens + (p.query_span_tokens > 0 ? p.query_span_tokens : p.chunk_tokens);
    p.last_total_tokens = q_used;

    // 选块的带（只影响这一行日志；本探针不改可见集 ⇒ 不改变引擎行为）。
    // 默认取"小预算"以便**逼出选择**：budget 32768 / sink 4096 / recent 0（官方那台仪器用的就是这组）。
    const std::int32_t budget_tokens = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
    const std::int32_t sink_tokens   = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);
    const std::int32_t recent_tokens = std::getenv("NINFER_TERNARY_KVMEM_SCORE_RECENT") != nullptr
                                           ? kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_RECENT", 0)
                                           : 0;

    const std::size_t bytes = static_cast<std::size_t>(p.n_blocks) * sizeof(float);
    if (kvmem_cpu_retrieval_enabled()) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) { kvmem_score_fail("tail host sync"); return; }
        try {
            MeanKIndex* index = mean_k_index_for(p.layers_total, p.n_kv_heads, p.head_dim,
                                               p.capacity_blocks);
            if (!index || !index->host_resident() || !p.tail_host_stash) {
                kvmem_score_fail("tail missing host buffers"); return;
            }
            const KvMemCpuQuery query{static_cast<const std::uint16_t*>(p.tail_host_stash->data()),
                static_cast<std::uint32_t>(p.query_span_tokens),
                static_cast<std::size_t>(p.tail_host_rows)};
            std::vector<std::vector<float>> scores;
            kvmem_cpu_scores(index->host_data(), index->layer_elements(), p.layers_total,
                             p.n_kv_heads, p.n_heads, p.head_dim,
                             std::span<const std::int32_t>(p.tokens_host.data(), p.n_blocks),
                             {}, std::span<const KvMemCpuQuery>(&query, 1), scores);
            for (std::int32_t b = 0; b < p.n_blocks; ++b) { p.score_host[b] += scores[0][b]; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[kvmem-cpu] tail scoring failed: %s\n", e.what());
            kvmem_score_fail("tail CPU scoring"); return;
        }
    } else {
        if (cudaMemcpyAsync(p.score_host.data(), p.score, bytes, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess) { kvmem_score_fail("score D2H"); return; }
        if (cudaStreamSynchronize(stream) != cudaSuccess) { kvmem_score_fail("sync"); return; }
    }

    double sum_score = 0.0;
    for (std::int32_t i = 0; i < p.n_blocks; ++i) { sum_score += static_cast<double>(p.score_host[i]); }

    KvMemSelectConfig scfg;
    scfg.total_blocks  = p.n_blocks;
    scfg.budget_blocks = budget_tokens / p.block_tokens;
    scfg.sink_blocks   = sink_tokens / p.block_tokens;
    scfg.recent_blocks = recent_tokens / p.block_tokens;
    const KvMemSelectResult sel = ops::kvmem_select_blocks(scfg, p.score_host.data(), 0);

    // 选择结构的形状（这是判据的核心：非连续 ⇒ 真在选择）
    std::int32_t runs = 0;
    for (std::size_t i = 0; i < sel.kept.size(); ++i) {
        if (i == 0 || sel.kept[i] != sel.kept[i - 1] + 1) { ++runs; }
    }
    const std::int32_t skip = p.n_blocks - static_cast<std::int32_t>(sel.kept.size());
    const std::int32_t kept_first = sel.kept.empty() ? -1 : sel.kept.front();
    const std::int32_t kept_last  = sel.kept.empty() ? -1 : sel.kept.back();

    // ★ 把**选中的块号本身**打出来：没有它就无法核对"含针的块有没有被选中"（聚合量 runs/skip 做不到）。
    //   与官方实现同形（它也是逐行 `KVMEM_TRACE selected <ids...>`）。只在短 query chunk 上出现。
    {
        std::fprintf(stderr, "kvmem_score: KEPT");
        for (std::size_t i = 0; i < sel.kept.size(); ++i) {
            std::fprintf(stderr, " %d", static_cast<int>(sel.kept[i]));
        }
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr,
                 "kvmem_score: SELECT label=%s n_blocks=%d budget_blocks=%d sink_blocks=%d "
                 "recent_blocks=%d kept=%zu runs=%d skip=%d window_tokens=%zu "
                 "sink_kept=%d recent_kept=%d scored_kept=%d candidates=%d kept_range=[%d,%d] "
                 "sum_score=%.3f query_tokens=%d oracle_ok=%d scale_floor=%.6f\n",
                 label, p.n_blocks, scfg.budget_blocks, scfg.sink_blocks, scfg.recent_blocks,
                 sel.kept.size(), runs, skip,
                 sel.kept.size() * static_cast<std::size_t>(p.block_tokens), sel.sink_kept,
                 sel.recent_kept, sel.scored_kept, sel.candidates, kept_first, kept_last, sum_score,
                 q_used, (q_used > 0 && sum_score > 0.5 * q_used &&
                                  sum_score < 2.0 * q_used)
                                     ? 1
                                     : 0,
                 static_cast<double>(ops::kvmem_retrieve_scale(p.head_dim)));
}

// 逐块分数导出（给离线判据用：核对"含针的块"有没有被选中）。只在 switch 打开且本 chunk 打过时写。
inline void kvmem_score_dump_if_requested(const char* label) noexcept {
    if (!kvmem_score_enabled()) { return; }
    const char* path = std::getenv("NINFER_TERNARY_KVMEM_SCORE_DUMP");
    if (path == nullptr || path[0] == '\0') { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.n_blocks <= 0) { return; }
    FILE* f = std::fopen(path, "ab");
    if (f == nullptr) { return; }
    std::fprintf(f, "# %s n_blocks=%d\n", label, p.n_blocks);
    for (std::int32_t i = 0; i < p.n_blocks; ++i) {
        std::fprintf(f, "%d %.6f\n", i, static_cast<double>(p.score_host[static_cast<std::size_t>(i)]));
    }
    std::fclose(f);
}

} // namespace ninfer::ops::detail
