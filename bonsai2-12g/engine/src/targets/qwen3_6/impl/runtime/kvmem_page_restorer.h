// prefill-window/route-B · A 方案：`stage_in_missing` 的 PageRestorer 适配器。
//
// 目的（照抄官方装配四步顺序 `selection -> stage-in -> stage-out -> assemble`）：
//   装配的**决策之后、装配之前**，把"被选中但不在 device"的块从 host 层搬回 device，使
//   `assemble_window` 不再撞 "selected block has no live page"（kvmem_window_assembly.cpp:94）。
//
// 官方依据（全部本地可查）：
//   · `kvmem_tiered_io_design.md:112` **"Evict before stage-in"**
//   · `:116` **"the working set is <= the window (<= budget <= pool), so the stage-in always fits"**
//     ⇒ 不设计"装不下"的分支；装不下只可能是配置错，让哨兵叫。
//   · `:117` **"The two sets are disjoint (stage_out = resident - window, stage_in = window)"**
//   · `:129` 顺序：`selection -> stage-in -> stage-out -> assemble`
//
// 本适配器只做**机制**（查驻留 + 搬回）；顺序与策略由调用点负责。
// 接口极性：装配链按**位置（槽）**问，我们的表按**块号**答（`kvmem_window_assembly.h:74-85`
// 把这个混淆点名为静默错答案）⇒ 本适配器**只吃槽**，槽→逻辑页的翻译由调用方一次性建立。
#pragma once

#include "core/paged_kv_cache.h"
#include "ops/kvmem/kvmem_window_assembly.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

class KVMemPageRestorer final : public ops::PageRestorer {
public:
    struct Sources {
        LogicalKVPageStore*                     pages    = nullptr;
        HostKVExtentStore*                      host     = nullptr;
        const std::vector<LogicalKVPageHandle>* slot_page = nullptr;  // 槽 -> 逻辑页
        cudaStream_t                            stream   = nullptr;
    };

    explicit KVMemPageRestorer(Sources sources) noexcept : s_(sources) {}

    // 注意力现在能不能读这一页？
    [[nodiscard]] bool device_resident(ops::BlockSlot slot) const override {
        const LogicalKVPageHandle page = page_for(slot);
        return page.valid() && s_.pages != nullptr && s_.pages->device_resident(page);
    }

    // 搬回来。要么成功（之后 `device_resident(slot)` 为真），要么返回 false 且**不留半成品**
    // —— 调用方据此放弃装配，而不是装一个缺页的窗口（`kvmem_window_assembly.h:304-306`）。
    [[nodiscard]] bool stage_in(ops::BlockSlot slot) override {
        if (s_.pages == nullptr || s_.host == nullptr || s_.stream == nullptr) { return false; }
        const LogicalKVPageHandle page = page_for(slot);
        if (!page.valid()) { return false; }
        if (s_.pages->device_resident(page)) { return true; }   // 已在位：无动作
        // 宿主副本必须在册。不在册 ⇒ 这一页不是"被降级"，而是从未有过宿主副本
        // （例如被别的路径释放）⇒ **宁可不装**，也不装一个来路不明的页。
        if (!s_.pages->host_resident(page)) { return false; }
        const HostKVPageReplica& replica = s_.pages->host_replica(page);
        const HostKVExtentCapability capability = replica.extent;
        if (!s_.host->valid(capability)) { return false; }

        // ★ 方向与压力路径的 stage_out 相反，逐条对应（program_impl.h 的 copy_to_host 那一处）：
        //   stage_out: prepare(pages, slots) -> device_sources -> copy_to_host
        //   stage_in : host_replica(页).extent -> reserve_device_replica -> copy_from_host
        //              -> publish_device_replica
        DeviceKVPageReservation reservation;
        DeviceKVPageHandle destination;
        try {
            destination = s_.pages->reserve_device_replica(page, reservation);
        } catch (...) {
            return false;   // 不可预留（epoch/coverage 不符、已被 pin 等）⇒ 拒绝，不装
        }
        if (!destination.valid()) { return false; }
        s_.pages->physical_pool().copy_from_host(s_.host->view(capability), destination, s_.stream);
        s_.pages->publish_device_replica(page);
        if (!s_.pages->device_resident(page)) { return false; }
        ++staged_;
        return true;
    }

    [[nodiscard]] std::uint32_t staged_count() const noexcept { return staged_; }

private:
    [[nodiscard]] LogicalKVPageHandle page_for(ops::BlockSlot slot) const {
        if (s_.slot_page == nullptr || slot < 0 ||
            static_cast<std::size_t>(slot) >= s_.slot_page->size()) {
            return {};
        }
        return (*s_.slot_page)[static_cast<std::size_t>(slot)];
    }
    Sources       s_;
    std::uint32_t staged_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
