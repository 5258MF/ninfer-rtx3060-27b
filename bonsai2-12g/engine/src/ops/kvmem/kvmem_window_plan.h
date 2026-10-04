#pragma once

// KVMem 装配臂（arm-assembly）· host-only 契约。
//
// 为什么单独一个头：本臂要接的三件东西已经存在，但彼此不认识 ——
//   ① 选择结果（ops/kvmem/kvmem_select.h：块 id 升序列表，纯 host）；
//   ② 装配计划（ops/kvmem/kvmem_window_assembly.h：块→页适配器 + 计划，纯 host）；
//   ③ 窗口载体（ops/kvmem/kvmem_window.h：arm/frontier/rope_position）。
// 本头只放"纯 host、无 CUDA、不依赖目标树"的那部分：开关、选择决定、重烘判定、KVMI-012 守卫。
// 引擎侧（program_impl.h）只做搬运：块→页句柄、assemble_window、rerope、arm。这正是
// kvmem_window_assembly.h:44-49 那条纪律（"跑不了 GPU 也能判"）在本臂上的延续。
//
// 开关纪律与 kvmem_shadow.h / kvmem_window.h 相同：env 只在本文件读一次（唯一的读取点），
// 默认关，opt-in 极性 `== "1"`。关掉时下列每个函数都不可达，所以 OFF 臂逐位不变
// （kvmem_window.h:46-51 的设计说明）。
//
// 规格与判据：E:\infer-build\exp\mem-arm\PATCH-arm-assembly.md

#include "core/paged_kv_cache.h"          // kPagedKVPageSize（判据的块大小；house rule：不写死 64）
#include "ops/kvmem/kvmem_select.h"
#include "ops/kvmem/kvmem_q4.h"   // KVMem P4 (kv8): protect-new
#include "ops/kvmem/kvmem_q4b.h"  // KVMem P4b (kv8b): protect-new v2
#include "ops/kvmem/kvmem_window_assembly.h" // BlockPageMap：计划侧的"块→槽"证据

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace ninfer::ops::detail {

// NINFER_TERNARY_KVMEM_WINDOW_ASSEMBLY=1 才允许"按选择装配窗口并 arm 载体"。默认关。
[[nodiscard]] inline bool kvmem_window_assembly_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_WINDOW_ASSEMBLY");
        if (value != nullptr && value[0] != '\0') { return std::string(value) == "1"; }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return kv != nullptr && std::string(kv) == "1";
    }();
    return enabled;
}

// NINFER_TERNARY_KVMEM_WINDOW_ASSEMBLY_STRICT=1：把"窗口必须仍然真的含有递归态所属的那段 KV
// （[0, S) 逐块在原槽）"这条要求，也施加到**非导入**的态上（本 lane 自己长出来的态）。
//
// 为什么默认关：对非导入的态，KVMem 的设计前提就是"递归态装着远处的记忆、窗口装着当下可见的块"，
// 强制前缀保留等于禁止这个功能本身。但这条前提在本引擎**不可证**（program.h:451-453 点名：
// "对混合模型这就是 KVMI-012 的全部"）⇒ 想要最保守的行为就把这个开关打开。
// 对**导入**的态（restore-by-pos / 长锚 / 共享前缀 / rewrite 检查点）无论本开关如何都强制执行：
// 那种情形下 W4 自己承诺过"KV 前缀 [0,p) 必须在"（PATCH-restore-by-pos.md §4 前置 1）。
[[nodiscard]] inline bool kvmem_assembly_strict() noexcept {
    static const bool strict = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_WINDOW_ASSEMBLY_STRICT");
        return value != nullptr && std::string(value) == "1";
    }();
    return strict;
}

// ---------------------------------------------------------------------------------------------
// ★ 守卫的**三分判据**（用户 2026-09-24 裁定：放宽合法形态、只拒真·错配；**红臂不许放宽掉**）
//   ① pass-noop         窗口**没 arm**（载体惰性）⇒ 可见集与今天完全相同 ⇒ 放行
//   ② pass-samesource   窗口 arm 了但**没有真压缩**（W >= F，合法空转，kvmem_window.h:38-39）；
//                       **或**真压缩了但**窗口仍逐块含 [0,S)**（S = 注入态前沿，且 S <= W）⇒ 放行
//   ③ reject-compacted  真压缩（W < F）**且**窗口丢了"递归态所属的块" ⇒ 抛（**红臂就在这一格**）
//   ④ reject-config     与 KVMI-012 无关的硬配置拒绝（非 fp8 KV / 共享地址 / 检查点保护 / 更靠后的
//                       长锚）。旁路**只降级 ③**，④ 不受影响。
// ---------------------------------------------------------------------------------------------
enum class KvmemGuardVerdict : std::uint8_t {
    PassNoop,
    PassSameSource,
    RejectCompacted,
    RejectConfig,
};

[[nodiscard]] inline const char* kvmem_guard_name(KvmemGuardVerdict verdict) noexcept {
    switch (verdict) {
        case KvmemGuardVerdict::PassNoop: return "pass-noop";
        case KvmemGuardVerdict::PassSameSource: return "pass-samesource";
        case KvmemGuardVerdict::RejectCompacted: return "reject-compacted";
        case KvmemGuardVerdict::RejectConfig: return "reject-config";
    }
    return "unknown";
}

// NINFER_TERNARY_KVMEM_ALLOW_MISMATCH=1：把 ③ reject-compacted 从"抛错"**降级为大声告警**，
// 并在日志里打 marked-mismatch（响应体侧的标记需要 serve 层改动，本轮未做；见规格 §4.6）。
// ⚠ 纪律（写死）：**用旁路跑出来的结果不得当作正确性证据**，只用于判断"是不是守卫挡住了通路"。
// 默认关 ⇒ 行为与严格版完全一致。
[[nodiscard]] inline bool kvmem_allow_mismatch() noexcept {
    static const bool allowed = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_ALLOW_MISMATCH");
        return value != nullptr && std::string(value) == "1";
    }();
    return allowed;
}

// 计划侧（request_plan_impl.h 的 W4 守卫）与装配臂**共用**的判据 —— 两处必须同一套理由，
// 否则"一处放行、另一处拒答"就是新的静默不一致。
//   window_armed/window_frontier : 载体状态（kvmem_window.h:78/83）
//   execution_frontier           : 序列前沿 F（arm 的比较基准）
//   state_frontier               : S = 注入态所属的时刻（计划侧 = reuse_base / restore_pos）
//   pages                        : 该 lane 的"块→槽"表（空表 = 从未装配过 ⇒ 页表没重排）
[[nodiscard]] inline KvmemGuardVerdict kvmem_window_state_guard(
    bool window_armed, std::int32_t window_frontier, std::uint32_t execution_frontier,
    std::uint32_t state_frontier, const BlockPageMap<std::int32_t>* pages, std::string* why,
    std::int32_t block_tokens = kPagedKVPageSize) {
    const auto note = [why](const std::string& text) {
        if (why != nullptr) { *why = text; }
    };
    if (!window_armed) {
        note("window carrier is inert");
        return KvmemGuardVerdict::PassNoop;
    }
    const std::uint32_t window = static_cast<std::uint32_t>(window_frontier);
    if (window >= execution_frontier) {
        note("window is armed but NOT compacted (W >= F): the visible set is today's");
        return KvmemGuardVerdict::PassSameSource;
    }
    if (window < state_frontier) {
        note("compacted window W=" + std::to_string(window) +
             " is shorter than the state's moment S=" + std::to_string(state_frontier));
        return KvmemGuardVerdict::RejectCompacted;
    }
    if (pages == nullptr || pages->size() == 0) {
        // 没有"块→槽"表 ⇒ 从未装配过 ⇒ 页表没被重排，只是窗口前沿前移 ⇒ [0,W) 仍是真前缀、
        // [0,S) ⊆ [0,W) 逐块在原槽（address 的构造性质：成员按序 append，logical_kv_store.h:1463-1500）
        // ⇒ 同源，放行。这一支正是 NINFER_TERNARY_KVMEM_WINDOW_DROP 那一档的形态。
        note("compacted with no page reorder recorded (no block->slot map) and W >= S: [0,S) intact");
        return KvmemGuardVerdict::PassSameSource;
    }
    if (block_tokens <= 0) { block_tokens = 1; }
    const std::uint32_t below =
        state_frontier == 0
            ? 0U
            : 1U + (state_frontier - 1U) / static_cast<std::uint32_t>(block_tokens);
    for (std::uint32_t block = 0; block < below; ++block) {
        const BlockPageMap<std::int32_t>::Lookup found = pages->lookup(block);
        if (!found.valid || found.slot != static_cast<std::int32_t>(block)) {
            note("compacted window drops KV the recurrent state belongs to (KVMI-012): block " +
                 std::to_string(block) + " is no longer at slot " + std::to_string(block) +
                 " while S=" + std::to_string(state_frontier));
            return KvmemGuardVerdict::RejectCompacted;
        }
    }
    note("compacted window still holds every block below S at its own slot");
    return KvmemGuardVerdict::PassSameSource;
}

// 与 kvmem_score.h:56-61 同形：正数才生效，否则回落。装配侧刻意沿用**打分探针同名的三个 env**
// （NINFER_TERNARY_KVMEM_SCORE_BUDGET / _SINK / _RECENT），这样"日志里那行 KEPT"与"实际装配的
// 窗口"用的是同一组带，不会张冠李戴。
[[nodiscard]] inline std::int32_t kvmem_assembly_env_i32(const char* name,
                                                         std::int32_t fallback) noexcept {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') { return fallback; }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? static_cast<std::int32_t>(parsed) : fallback;
}

// KVMem P1e: query replay length in tokens (0 = off). Read once.
[[nodiscard]] inline std::uint32_t kvmem_replay_tokens() noexcept {
    static const std::uint32_t tokens = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_REPLAY_TOKENS");
        if (value != nullptr && value[0] != '\0') {
            const long parsed = std::strtol(value, nullptr, 10);
            return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 0U;
        }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return (kv != nullptr && kv[0] != '\0' && kv[0] != '0') ? 48U : 0U;
    }();
    return tokens;
}

// KVMem P1e: first token of the replayed tail for a prompt of `prompt_tokens` tokens (0 = none).
// The tail never reaches past the last two blocks, which the semantic gear keeps when replay is on.
[[nodiscard]] inline std::uint32_t kvmem_replay_begin(std::uint32_t prompt_tokens,
                                                      std::uint32_t block_tokens) noexcept {
    std::uint32_t tokens = kvmem_replay_tokens();
    if (tokens == 0 || block_tokens == 0) { return 0U; }
    const std::uint32_t tail = prompt_tokens % block_tokens == 0 ? block_tokens
                                                                 : prompt_tokens % block_tokens;
    tokens = std::min(tokens, tail + block_tokens);
    return tokens < prompt_tokens ? prompt_tokens - tokens : 0U;
}

// ---- KVMem kv9 (item 2): replay range == retrieval range == THIS round's input -----------------
// Paper: the tokens that pick the blocks are the tokens that are re-prefilled over the new view.
// REPLAY_MODE=msg: replay starts at the <|im_start|> of the first message after the last assistant
// message (the trailing run of user-role messages: the user message, or the tool results), through
// the generation prompt. A longer input keeps only its last REPLAY_MAX tokens (contiguous tail; the
// window keeps them as recent blocks). Falls back to the legacy REPLAY_TOKENS tail when the input
// cannot be located or starts before this request's first prefill position.
[[nodiscard]] inline bool kvmem_replay_msg_mode() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_REPLAY_MODE");
        if (value != nullptr && value[0] != '\0') { return std::string(value) == "msg"; }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return kv != nullptr && kv[0] != '\0' && kv[0] != '0';
    }();
    return on;
}
[[nodiscard]] inline std::uint32_t kvmem_replay_max() noexcept {
    static const std::uint32_t cap = [] {
        // default = the tail half of the protect-new budget: the replayed tail then lies inside what
        // protect-new keeps anyway and costs no retrieval slots (12288 -> 6144).
        const std::int32_t protect = kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW_MAX", 12288);
        std::int32_t c = kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_REPLAY_MAX", protect / 2);
        const std::int32_t budget = kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
        c = std::min(c, budget / 3);
        return static_cast<std::uint32_t>(std::max<std::int32_t>(c, 64));
    }();
    return cap;
}
[[nodiscard]] inline std::uint32_t kvmem_replay_input_begin(const std::int32_t* ids,
                                                            std::uint32_t n) noexcept {
    if (ids == nullptr || n < 64U) { return 0U; }
    const std::int32_t start = ids[0];
    std::uint32_t g = 0;
    for (std::uint32_t p = n - 1U; p > 0 && p + 32U >= n; --p) {
        if (ids[p] == start) { g = p; break; }
    }
    std::uint32_t out = 0;
    if (g > 0 && g + 1U < n) {
        const std::int32_t role_asst = ids[g + 1U];
        // position 0 counts too (a chat without a system message starts with the user message)
        std::int64_t m = -1;
        for (std::int64_t p = static_cast<std::int64_t>(g) - 1; p >= 0; --p) {
            if (ids[p] == start) { m = p; break; }
        }
        if (m >= 0 && m + 1 < static_cast<std::int64_t>(g) && ids[m + 1] != role_asst) {
            const std::int32_t role = ids[m + 1];
            std::int64_t first = m;
            for (std::int64_t p = m - 1; p >= 0; --p) {
                if (ids[p] != start) { continue; }
                if (ids[p + 1] != role) { break; }
                first = p;
            }
            out = static_cast<std::uint32_t>(first);
            if (out == 0) { out = 1; }   // 0 means "not found": replay from token 1 instead
        }
    }
    return out;
}
[[nodiscard]] inline std::uint32_t kvmem_replay_begin_for(const std::int32_t* ids,
                                                          std::uint32_t prompt_tokens,
                                                          std::uint32_t first_position,
                                                          std::uint32_t block_tokens) noexcept {
    const std::uint32_t legacy = kvmem_replay_begin(prompt_tokens, block_tokens);
    if (legacy == 0 || !kvmem_replay_msg_mode()) { return legacy; }
    std::uint32_t b = kvmem_replay_input_begin(ids, prompt_tokens);
    if (b == 0) { return legacy; }
    if (prompt_tokens - b > kvmem_replay_max()) { b = prompt_tokens - kvmem_replay_max(); }
    if (b < first_position || b >= prompt_tokens) { return legacy; }
    // exactly the input (+ generation prompt): a short tool result replays only itself -- and never
    // reaches back before an adopted continuation's first prefill position like the 48-token tail can.
    return b;
}
// The window must keep [begin, prompt) as its contiguous tail: set at the GDN save, read by the
// semantic window decision of the same prompt, cleared after the boundary.
struct KvMemReplayPlan {
    std::uint32_t prompt = 0;
    std::uint32_t begin  = 0;
};
inline KvMemReplayPlan& kvmem_replay_plan() {
    static KvMemReplayPlan plan;
    return plan;
}

// ★ 语义档开关（2026-09-26 第五棒）：NINFER_TERNARY_KVMEM_SEMANTIC=1 ⇒ prefill→decode 边界按
// **分数**装窗（kvmem_window_decide，官方 prefill→decode 边界的那次语义选择）。默认 0 = 压力档
// （今天的行为，逐位不变）。开关只在这里读（单一读取点纪律）。
[[nodiscard]] inline bool kvmem_semantic_gear() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_SEMANTIC");
        if (value != nullptr && value[0] != '\0') { return std::string(value) == "1"; }
        const char* kv = std::getenv("NINFER_TERNARY_KVMEM");
        return kv != nullptr && std::string(kv) == "1";
    }();
    return on;
}

// ★ 负控排除表（语义档红臂的仪器）：NINFER_TERNARY_KVMEM_SELECT_EXCLUDE="312-317,42" ⇒ 这些块在
// 装配的选择里被降为 -inf（分数永不选中它们）。默认空 = 不影响任何行为。它是"故意把针块排除在
// 窗口外 ⇒ 必须答不出"这条负控的仪器：喂给选择器**不同的输入**，不拆任何报警器。
[[nodiscard]] inline const std::vector<std::int32_t>& kvmem_select_excluded_blocks() {
    static const std::vector<std::int32_t> blocks = [] {
        std::vector<std::int32_t> out;
        const char* value = std::getenv("NINFER_TERNARY_KVMEM_SELECT_EXCLUDE");
        if (value == nullptr || value[0] == '\0') { return out; }
        const std::string text(value);
        std::size_t pos = 0;
        while (pos < text.size()) {
            const std::size_t comma = text.find(',', pos);
            const std::string item =
                text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            const std::size_t dash = item.find('-');
            const long a = std::strtol(item.c_str(), nullptr, 10);
            const long b =
                dash == std::string::npos ? a : std::strtol(item.c_str() + dash + 1, nullptr, 10);
            for (long i = a; i <= b && i - a < 100000; ++i) {
                out.push_back(static_cast<std::int32_t>(i));
            }
            if (comma == std::string::npos) { break; }
            pos = comma + 1;
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }();
    return blocks;
}

// 选择结果 + 出处（provenance）。出处三项是**必须**的：选择是进程级的、相对当前 query 会过期，
// 没有出处就无法判断"手上这份选择描述的是哪个时刻的历史"（静默用错选择 = 静默错答案）。
struct KvmemWindowDecision {
    KvMemSelectResult selection{};
    std::int32_t  total_blocks       = 0;  // 可选的块数 = ceil(F / 块)（**含半写尾块**）
    std::int32_t  scored_blocks      = 0;  // 其中前多少块有分数（其余 = 无信号，由 recent 保进窗）
    std::int32_t  block_tokens       = 0;  // kPagedKVPageSize，由调用方传入（house rule：不写死）
    std::int32_t  last_block_columns = 0;  // 候选历史里最后一个块的真实填充（<=0 = 满块）
    std::uint32_t history_frontier   = 0;  // 这份选择描述的 token 前沿
    std::uint64_t stamp              = 0;  // 哪一次打分产生的（调用方给，用来判断"要不要重新装配"）
    bool          valid              = false;
};

// 从 host 分数数组得到"窗口要装哪些块"。分数数组来自选择侧（kvmem_score.h 的探针把它落到
// host 上了），本函数**只读**它。返回 valid=false 表示"这次没有可用的选择"⇒ 调用方必须保持
// 窗口不动（宁可不装配，也不装配一个空的/来路不明的窗）。
//
// ★ 2026-09-26 语义档（第五棒）的三处口径（run-needle512 教出来的，别退回去）：
//   ① `total_blocks` = **ceil(F / 块)**、**含半写尾块**；`scored_blocks` = 分数覆盖了其中前多少块
//      （打分只对着"历史"，最后那个 chunk 自己的块没被分到）。尾块的真实填充由
//      `last_block_columns` 携带（assemble_block_window 按 kept.back() == total-1 认领它）。
//   ② [scored_blocks, total_blocks) 是**无信号**（-inf）：不会顶掉有分数的块，但**必须能进窗** ——
//      它们就是"问"自己的上下文。⇒ recent 强制 ≥ max(1, total - scored)：**不许把提问本身
//      丢出窗口**（丢过一次的代价：答案在聊填充句、答非所问）。
//   ③ 负控排除表（kvmem_select_excluded_blocks）把这些块降为 -inf，且**逐块核对**没进 kept ——
//      排除了还在窗里就大声喊出来（红臂不许假绿）。
[[nodiscard]] inline KvmemWindowDecision
kvmem_window_decide(const float* scores, std::int32_t scored_blocks, std::int32_t total_blocks,
                    std::int32_t block_tokens, std::int32_t last_block_columns,
                    std::uint32_t history_frontier, std::uint64_t stamp) noexcept {
    KvmemWindowDecision out;
    out.total_blocks       = total_blocks;
    out.scored_blocks      = scored_blocks;
    out.block_tokens       = block_tokens;
    out.last_block_columns = last_block_columns;
    out.history_frontier   = history_frontier;
    out.stamp              = stamp;
    if (scores == nullptr || scored_blocks <= 0 || total_blocks <= 0 || block_tokens <= 0) {
        return out;
    }
    if (scored_blocks > total_blocks) { scored_blocks = total_blocks; out.scored_blocks = scored_blocks; }

    // 本地缓冲：有分数段 + 无信号段 + 负控排除（原数组只读，绝不回写探针状态）。
    static const float kNoSignal = -std::numeric_limits<float>::infinity();
    std::vector<float> buf(static_cast<std::size_t>(total_blocks), kNoSignal);
    std::copy_n(scores, static_cast<std::size_t>(scored_blocks), buf.begin());
    const std::vector<std::int32_t>& excluded = kvmem_select_excluded_blocks();
    for (const std::int32_t block : excluded) {
        if (block >= 0 && block < total_blocks) { buf[static_cast<std::size_t>(block)] = kNoSignal; }
    }

    const std::int32_t budget_tokens =
        kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
    const std::int32_t sink_tokens = kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);
    const std::int32_t recent_env =
        kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_RECENT", 0);

    KvMemSelectConfig config;
    config.total_blocks  = total_blocks;
    config.budget_blocks = budget_tokens / block_tokens;
    config.sink_blocks   = sink_tokens / block_tokens;
    // ②：无信号段 + 至少最新一块（"问"所在的尾块）必须在窗里。
    config.recent_blocks = std::max(recent_env, std::max(1, total_blocks - scored_blocks));
    if (kvmem_replay_tokens() != 0) { config.recent_blocks = std::max(config.recent_blocks, 2); }
    if (kvmem_replay_tokens() != 0 && kvmem_replay_msg_mode()) {   // KVMem kv9 (item 2)
        const KvMemReplayPlan& rp = kvmem_replay_plan();
        const std::int64_t bt = block_tokens;
        if (rp.prompt != 0 && rp.begin != 0 && rp.begin < rp.prompt &&
            static_cast<std::int64_t>(rp.prompt) > (total_blocks - 1) * bt &&
            static_cast<std::int64_t>(rp.prompt) <= total_blocks * bt) {
            const std::int32_t need = total_blocks - static_cast<std::int32_t>(rp.begin / block_tokens);
            config.recent_blocks = std::max(config.recent_blocks, need);
            std::fprintf(stderr, "[kvmem-r9] replay tail kept as recent: begin=%u prompt=%u blocks=%d recent=%d\n",
                         rp.begin, rp.prompt, need, config.recent_blocks);
        }
    }
    // KVMem P4 (kv8): keep THIS request's new content (its own prefill: previous answer re-rendered +
    // new user / tool message) in the window instead of letting it compete with old history. Up to
    // SCORE_PROTECT_NEW_MAX tokens; a longer new part keeps its head (forced by score) and its tail
    // (recent), the middle competes as before.
    if (kvmem_q4_protect_new() && kvmem_q4_protect_v1()) {   // v1 (kv8); v2 below -- KVMem P4b (kv8b)
        const KvMemQ4Request& rq = kvmem_q4_request();
        const std::int32_t cap_blocks = std::max<std::int32_t>(
            2, kvmem_q4_env_i32("NINFER_TERNARY_KVMEM_SCORE_PROTECT_NEW_MAX", 12288) / block_tokens);
        if (rq.req_start >= 0) {
            const std::int32_t first_new  = rq.req_start / block_tokens;
            const std::int32_t new_blocks = total_blocks - first_new;
            std::int32_t forced = 0;
            if (new_blocks > 0 && first_new >= 0) {
                if (new_blocks <= cap_blocks) {
                    config.recent_blocks = std::max(config.recent_blocks, new_blocks);
                } else {
                    const std::int32_t head = cap_blocks / 2;
                    config.recent_blocks = std::max(config.recent_blocks, cap_blocks - head);
                    for (std::int32_t b = first_new; b < first_new + head && b < total_blocks; ++b) {
                        if (std::binary_search(excluded.begin(), excluded.end(), b)) { continue; }
                        buf[static_cast<std::size_t>(b)] = std::numeric_limits<float>::max();
                        ++forced;
                    }
                }
            }
            std::fprintf(stderr,
                         "[kvmem-q4] protect-new req_start=%d first_block=%d new_blocks=%d "
                         "cap_blocks=%d recent=%d head_forced=%d\n",
                         rq.req_start, first_new, new_blocks, cap_blocks, config.recent_blocks, forced);
        }
    }

    // KVMem P4b (kv8b): protect-new v2 -- force the message-aware ranges (kvmem_q4b.h) into the window.
    // Only middle blocks are candidates (sink / recent are kept anyway); at least max(64, 1/4) of the
    // scored budget always stays free for retrieval.
    if (kvmem_q4_protect_new() && !kvmem_q4_protect_v1()) {
        const KvMemQ4Protect& pr = kvmem_q4_protect();
        const std::int32_t sink_b   = std::min(config.sink_blocks, total_blocks);
        const std::int32_t recent_b = std::min(config.recent_blocks, std::max(0, total_blocks - sink_b));
        const std::int32_t cand_hi  = total_blocks - recent_b;
        const std::int32_t remaining = config.budget_blocks - sink_b - recent_b;
        const std::int32_t limit = std::max<std::int32_t>(0, remaining - std::max<std::int32_t>(64, remaining / 4));
        const float kForce = std::numeric_limits<float>::max();
        std::int32_t forced = 0, fixed = 0, dropped = 0;
        for (int r = 0; r < pr.n; ++r) {
            if (pr.e[r] <= pr.b[r] || pr.b[r] < 0) { continue; }
            const std::int32_t b0 = pr.b[r] / block_tokens, b1 = (pr.e[r] - 1) / block_tokens;
            for (std::int32_t blk = b0; blk <= b1 && blk < total_blocks; ++blk) {
                if (blk < sink_b || blk >= cand_hi) { ++fixed; continue; }
                if (std::binary_search(excluded.begin(), excluded.end(), blk)) { continue; }
                float& s = buf[static_cast<std::size_t>(blk)];
                if (s == kForce) { continue; }
                if (forced >= limit) { ++dropped; continue; }
                s = kForce;
                ++forced;
            }
        }
        std::fprintf(stderr,
                     "[kvmem-q4] protect2-window ranges=%d forced=%d in_sink_or_recent=%d dropped=%d limit=%d "
                     "remaining=%d total_blocks=%d\n",
                     pr.n, forced, fixed, dropped, limit, remaining, total_blocks);
    }
    out.selection = kvmem_select_blocks(config, buf.data(), 0);
    out.valid     = !out.selection.kept.empty();

    // ③：负控排除表的硬核对（排除了还进窗 = 负控是假的，当场喊）。
    if (!excluded.empty() && out.valid) {
        std::int32_t leaked = 0;
        for (const std::int32_t block : out.selection.kept) {
            if (std::binary_search(excluded.begin(), excluded.end(), block)) { ++leaked; }
        }
        std::fprintf(stderr,
                     "kvmem_window: negctl-exclude listed=%zu leaked_into_kept=%d (must be 0)\n",
                     excluded.size(), leaked);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// ★ prefill 期压力档（prefill-window/route-B）：官方 `pick_prefill_pressure_blocks` 的等价物。
//
// 官方语义（`kvmem_store.cpp:306-348`）：保住前 `sink` 块，其余预算槽位**从最新往回填满**；
// **注意力/检索/profile 分数一律忽略**（确定性）。它由 `kvmem_maybe_prefill_offload` →
// `kvmem_reselect_prefill_pressure` 在 **prefill 期间**调用，与 prefill→decode 边界那次
// **语义**选择（打分档）是**两件事，不许混**：
//   官方 `implementation_notes.md:318-325` 明写——把语义选择提前到 prefill 期，会让"新 hidden
//   state 被**旧 query** 塑形"，正是 `kvmem_prepare_prefill_window` 要防的失效模式。
//
// 为什么这里不需要新写选择器：`kvmem_select.cpp:104-105` 的分支——`remaining = budget - sink -
// recent <= 0` ⇒ "always-kept regions ARE the answer" ⇒ **中间块根本不看 scores**。于是
// `budget = sink + recent` 这一组参数就精确表达了官方的压力策略。
//
// `total_blocks <= 0`（历史为空，例如第 1 个 chunk）⇒ `valid=false`，调用方必须保持窗口不动
// （宁可不装配，也不装配一个空的/来路不明的窗）。
[[nodiscard]] inline KvmemWindowDecision
kvmem_window_decide_pressure(std::int32_t total_blocks, std::int32_t block_tokens,
                             std::int32_t last_block_columns, std::uint32_t history_frontier,
                             std::uint64_t stamp) noexcept {
    KvmemWindowDecision out;
    out.total_blocks       = total_blocks;
    out.block_tokens       = block_tokens;
    out.last_block_columns = last_block_columns;
    out.history_frontier   = history_frontier;
    out.stamp              = stamp;
    if (total_blocks <= 0 || block_tokens <= 0) { return out; }

    const std::int32_t budget_tokens =
        kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
    const std::int32_t sink_tokens = kvmem_assembly_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);

    const std::int32_t budget_blocks = budget_tokens / block_tokens;
    if (budget_blocks <= 0) { return out; }  // 预算 < 1 块：不动作（0 在 select 里是"不限"）
    std::int32_t sink_blocks = sink_tokens / block_tokens;
    sink_blocks              = std::min(sink_blocks, budget_blocks);

    KvMemSelectConfig config;
    config.total_blocks  = total_blocks;
    config.sink_blocks   = sink_blocks;
    // budget == sink + recent ⇒ `remaining <= 0` ⇒ 只留 sink 与最新尾部，完全不看分数。
    config.recent_blocks = budget_blocks - sink_blocks;
    config.budget_blocks = budget_blocks;

    // scores 在这条路径上不会被读（`remaining <= 0` 分支），但 `kvmem_select_blocks` 在
    // `budget_blocks > 0` 且 `scores == nullptr` 时会抛 ⇒ 传一个非空的进程寿命哨兵。
    static const float kNoSignal = -std::numeric_limits<float>::infinity();
    out.selection = kvmem_select_blocks(config, &kNoSignal, 0);
    out.valid     = !out.selection.kept.empty();
    return out;
}

// 重烘判定（★ 路 3 的判据）。
//
// 成立的充要条件：被选中块的窗口槽 == 块号（ordered_blocks[i] == i）。理由：可见集是**从 0 开始
// 的连续前缀**、页表第 r 行就是位置 r（ops/kernel/paged_kv_address.cuh:42-45：block_table[position>>6]），
// 而块 b 的 K 是在绝对位置 [b*block, b*block+block) 烘的。只有它落在槽 b 时，"槽序号 == 烘制位置"
// 才成立 ⇒ 一个字节都不用改。任何别的排列（含"中段+尾段"这种非连续选择）都必须从 raw-K 重烘。
struct KvmemRebakeNeed {
    std::uint32_t moved    = 0;  // 槽 != 块号 ⇒ 必须重烘的块数
    std::uint32_t identity = 0;  // 槽 == 块号 ⇒ 原位、免烘
    [[nodiscard]] bool needs_raw_k() const noexcept { return moved != 0; }
};

[[nodiscard]] inline KvmemRebakeNeed
kvmem_window_rebake_need(const std::vector<std::uint32_t>& ordered_blocks) noexcept {
    KvmemRebakeNeed need;
    for (std::size_t index = 0; index < ordered_blocks.size(); ++index) {
        if (ordered_blocks[index] == static_cast<std::uint32_t>(index)) {
            ++need.identity;
        } else {
            ++need.moved;
        }
    }
    return need;
}

// ★ KVMI-012 守卫（KV 侧）的输入。
//
// 诚实边界（写死在类型里，不许绕）：KV 侧**没有** StateImage 的那个 content_epoch，所以"态 ↔ KV
// 同源"只能靠结构证据钉。注意 KV 侧**有它自己的** per-page epoch
// （logical_kv_store.h:414-416 / :1770-1777），它钉的是"这个槽的页还是当初那个块吗"（由
// MappedBlockPages 的 current_epoch 回调使用），两套 epoch **不可互比**。
struct KvmemSameSourceInputs {
    std::uint32_t window_frontier = 0;   // 装配后的 W（窗口的 token 长度）
    bool          window_compacts = true; // 这次装配是否真压缩（W < F）；false ⇒ 三分判据的 ② 情形
    bool          allow_mismatch  = false; // 旁路：把 ③ 降级为告警（见 kvmem_allow_mismatch）
    std::uint32_t state_frontier  = 0;   // S：递归态所属的 token 时刻（= 本 lane 的 base）
    bool          state_imported  = false;  // 态是"取回来"的（不是本 lane 自己长出来的）
    bool          strict_prefix   = false;  // 对**非导入**的态也强制前缀保留
    std::int32_t  block_tokens    = 0;
    const std::vector<std::uint32_t>* ordered_blocks = nullptr;  // 窗口槽 → 块号
    bool          address_shared      = false;  // 该 KV 地址与别的序列共享
    bool          rebakeable_kv       = true;   // 目标 K 平面的编码有对应的重烘写出端
    std::uint32_t checkpoint_frontier = 0;      // 地址的 checkpoint 保护前沿
    std::uint32_t oldest_anchor_frontier = 0;   // 该 lane 上最靠后的长锚前沿（0 = 没有）
};

// 通过 ⇒ 允许装配并 arm；不通过 ⇒ **必须抛错拒答**（调用方负责抛，本函数只填 why）。
// 五条结构证据，每条都对应一个具体的静默错答：
//   1. 地址不共享 —— 否则重排页表会改到别人的可见集；
//   2. fp8 K —— 重烘写的是引擎自己的 Hadamard+fp8 码流（kvmem_rerope.h:33-40），别的 dtype 写错格式；
//   3. W >= S 且 [0,S) 逐块在原槽 —— 否则"窗口切掉了递归态所属的 KV"，这正是 KVMI-012；
//   4. checkpoint 保护前沿 <= W —— 否则下一次 rebuild_checkpoint_protection() 会 std::terminate()
//      （logical_kv_store.h:1879-1881），不是异常，HTTP 层看不见；
//   5. 没有比 W 更靠后的长锚 —— 锚是对外承诺"p 处的 KV 可取回"，压缩把它作废。
// 装配臂用。verdict（可选）回填三分判据，供调用方打一致的日志。
// **旁路只降级 reject-compacted**：config 类硬拒绝（非 fp8 / 共享地址 / 检查点 / 锚）不受影响。
[[nodiscard]] inline bool kvmem_window_same_source(const KvmemSameSourceInputs& in, std::string* why,
                                                   KvmemGuardVerdict* verdict = nullptr) {
    const auto put = [verdict](KvmemGuardVerdict value) {
        if (verdict != nullptr) { *verdict = value; }
    };
    put(KvmemGuardVerdict::PassSameSource);
    const auto reject = [&](const std::string& text, KvmemGuardVerdict value) {
        if (why != nullptr) { *why = text; }
        put(value);
        if (value == KvmemGuardVerdict::RejectCompacted && in.allow_mismatch) {
            // 旁路：降级为告警（**结果不得当作正确性证据**），标记留在 why 里。
            if (why != nullptr) { *why = text + " [marked-mismatch]"; }
            return true;
        }
        return false;
    };
    const auto fail = [&](const std::string& text) {
        return reject(text, KvmemGuardVerdict::RejectConfig);
    };
    // sm_86 port, 2026-09-27: the re-bake now has an int8 store side
    // (ops/kvmem/rerope/kvmem_rerope_launch.cu, ReropeStoreInt8G64), so the encodings this
    // arm can serve are fp8 AND int8 rather than fp8 alone. That matters on Ampere, where
    // fp8 KV cannot be read back at all -- its attention path needs sm_89 FP8 tensor cores --
    // so int8 is the only encoding that makes window assembly reachable there.
    //
    // Still worth knowing why the arm can look inert even when it is wired: the selection
    // budget defaults to 229,376 tokens (ops/kvmem/kvmem_port_bridge.h:105) while admission
    // caps a prompt at max_context, so `n <= budget` holds and selection keeps everything.
    // Lower it with NINFER_TERNARY_KVMEM_SELECT_BUDGET to make eviction -- and therefore the
    // re-bake -- actually happen.
    if (in.ordered_blocks == nullptr || in.block_tokens <= 0) {
        return fail("KVMem window guard: the window plan is malformed");
    }
    if (in.address_shared) {
        return fail("KVMem window guard: the KV address is shared with another sequence "
                    "(assembling a window would republish somebody else's visible set)");
    }
    if (!in.rebakeable_kv) {
        return fail("KVMem window guard: re-RoPE has a store side for the fp8/H and int8-g64 "
                    "K codecs only (run with --kv-dtype fp8 or int8)");
    }
    // 三分判据的 ②：这次装配**没有真压缩**（窗口就是整段前缀）⇒ [0,S) 必在其中 ⇒ 放行。
    if (in.window_compacts && (in.state_imported || in.strict_prefix)) {
        if (in.window_frontier < in.state_frontier) {
            return reject("KVMem window drops KV the recurrent state belongs to (KVMI-012): window W=" +
                        std::to_string(in.window_frontier) +
                        " is shorter than the state's moment S=" +
                        std::to_string(in.state_frontier), KvmemGuardVerdict::RejectCompacted);
        }
        const std::uint32_t block_tokens = static_cast<std::uint32_t>(in.block_tokens);
        const std::uint32_t below =
            in.state_frontier == 0
                ? 0U
                : 1U + (in.state_frontier - 1U) / block_tokens;  // 覆盖 [0,S) 的块数
        for (std::uint32_t block = 0; block < below; ++block) {
            if (in.ordered_blocks->size() <= static_cast<std::size_t>(block) ||
                (*in.ordered_blocks)[static_cast<std::size_t>(block)] != block) {
                return reject("KVMem window drops KV the recurrent state belongs to (KVMI-012): "
                              "block " + std::to_string(block) + " is not at window slot " +
                              std::to_string(block) + " while S=" +
                              std::to_string(in.state_frontier),
                              KvmemGuardVerdict::RejectCompacted);
            }
        }
    }
    if (in.checkpoint_frontier > in.window_frontier) {
        return fail("KVMem window would invalidate the KV checkpoint requirement at " +
                    std::to_string(in.checkpoint_frontier) + " (window W=" +
                    std::to_string(in.window_frontier) +
                    "); the next checkpoint-protection rebuild would terminate the process");
    }
    if (in.oldest_anchor_frontier > in.window_frontier) {
        return fail("KVMem window would invalidate a long anchor at " +
                    std::to_string(in.oldest_anchor_frontier) + " (window W=" +
                    std::to_string(in.window_frontier) + ")");
    }
    return true;
}

} // namespace ninfer::ops::detail
