// KVMem v5 (kv5.py): physical admission switch and the one-shot relax flag shared by the
// engine loop and the qwen3_6 program. Header-only; C++17 inline variables.
#pragma once
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace ninfer_kvmem_v5 {
inline std::atomic<bool> g_relaxed{false};
inline bool enabled() noexcept {
    static const bool on = [] {
        const char* k = std::getenv("NINFER_TERNARY_KVMEM");
        const char* w = std::getenv("NINFER_TERNARY_KVMEM_WINDOW_ASSEMBLY");
        const char* o = std::getenv("NINFER_TERNARY_KVMEM_PHYS_ADMISSION");
        const bool v = k && k[0] == '1' && !(w && w[0] == '0') && !(o && o[0] == '0');
        std::fprintf(stderr, "[kvmem-v5] physical admission %s\n", v ? "ON" : "OFF");
        return v;
    }();
    return on;
}
// Called when an idle engine cannot admit its FIFO head: fall back to address-space admission.
inline bool relax_once() noexcept {
    if (!enabled() || g_relaxed.load()) { return false; }
    g_relaxed.store(true);
    std::fprintf(stderr, "[kvmem-v5] idle engine blocked: relaxing to address-space admission once\n");
    return true;
}
inline void reset() noexcept {
    if (g_relaxed.exchange(false)) { std::fprintf(stderr, "[kvmem-v5] relaxed admission used; back to physical\n"); }
}
} // namespace ninfer_kvmem_v5
