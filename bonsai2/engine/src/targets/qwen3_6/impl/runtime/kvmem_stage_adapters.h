// prefill-window/route-B · A 方案：官方两相装配（`stage_assemble_and_resolve`）的三个注入式适配器。
//
// 官方为什么要两相（`kvmem_window_assembly.h:329-350`，转引）：
//   phase 1 (materialization 之前): PLACEMENT  — 这一块将占哪个槽（洞 = -1）
//   phase 2 (materialization 之后): RESOLUTION — 它现在是哪个逻辑页（恒等，绝不用哨兵）
//   *"`stage_in_missing` assumes the page source can already name a slot for every selected block.
//     At the WIRING POINT it cannot … a block whose page is not on the device HAS no such handle yet …
//     So there is nothing to 'fill' before assembly."*
//   ⇒ **洞在"槽里有没有活页"这一侧，不在"有没有逻辑页句柄"那一侧。**
//
// 装配顺序（官方 `kvmem_tiered_io_design.md:129`）：`selection -> stage-in -> stage-out -> assemble`
// 三条铁律（同文）：`:112` **先 evict 后 stage-in**；`:116` **工作集 ≤ 窗口 ≤ 预算 ≤ 池 ⇒ stage-in
// 永远装得下**（不设计装不下的分支）；`:117` **两集合不相交**。
#pragma once

#include "core/paged_kv_cache.h"
#include "ops/kvmem/kvmem_window_assembly.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// ---- phase 2：槽 → 逻辑页（恒等；绝不用哨兵）-------------------------------------------------
// 我们的窗口是**紧凑打包**的：窗口顺序里第 i 个块占槽 i。所以 resolver 就是恒等映射；
// 真正需要它的是"拿到该槽的**句柄**"，这件事由 `KVAddressSpaceStore::logical_page(address, i)` 做。
class KVMemSlotResolver final : public ops::PageResolver {
public:
    explicit KVMemSlotResolver(const KVAddressSpaceStore& addresses, KVAddressSpaceHandle address)
        : addresses_(addresses), address_(address) {}

    [[nodiscard]] std::uint64_t logical_page(ops::BlockSlot slot) const override {
        if (slot < 0) { return 0; }
        return static_cast<std::uint64_t>(slot);
    }

    [[nodiscard]] LogicalKVPageHandle handle(ops::BlockSlot slot) const {
        if (slot < 0) { return {}; }
        return addresses_.logical_page(address_, static_cast<std::uint32_t>(slot));
    }

    // ★ 2026-09-26 语义档修复：**窗位 ≠ 块号**。检索要的是"块自己的页"（V 在页里、K 原地重烘），
    //   必须走块→页注册表；membership 是"窗口槽→页"且装配时整行重写，按窗位取页会装成别的块的页
    //   （K 对 V 错的静默错答案，§31）。
    [[nodiscard]] LogicalKVPageHandle block_handle(std::uint32_t block) const {
        return addresses_.block_page(address_, block);
    }

    [[nodiscard]] std::span<const LogicalKVPageHandle> window_members() const {
        return addresses_.last_window_members();
    }

    [[nodiscard]] KVAddressSpaceHandle address() const noexcept { return address_; }
    [[nodiscard]] KVAddressSpaceStore& addresses() const noexcept {
        return const_cast<KVAddressSpaceStore&>(addresses_);
    }

private:
    const KVAddressSpaceStore& addresses_;
    KVAddressSpaceHandle       address_;
};

// ---- phase 1：设计槽（placement）------------------------------------------------------------
// 报洞的唯一条件：该槽的逻辑页**不在 device**（被降级/换出）—— 正是要 stage-in 的那一类。
// ⚠ "块→槽表里有没有条目"**不参与判断**：表只回答"哪个格"，页驻留是另一回事。
// （`kvmem_window_assembly.h:198-203` 的 `invalidate` 说"tier 把块搬走 ⇒ 该行停止回答"，
//  而两相设计恰恰要求这类块**仍然有槽可指**，否则洞永远填不上。）
class KVMemWindowPlacement final : public ops::PagePlacementSource {
public:
    KVMemWindowPlacement(const LogicalKVPageStore& pages, const KVMemSlotResolver& resolver,
                         const std::vector<std::int32_t>* ordered_blocks)
        : pages_(pages), resolver_(resolver), blocks_(ordered_blocks) {}

    // READ ONLY：本函数**不许**触发 stage-in（`kvmem_window_assembly.h:354-355`）。
    // ★ 2026-09-26：设计槽 = 紧凑窗位（position）；**页是"该位那块自己的页"**（块→页注册表）。
    //   报 -1 的唯一情形 = 这块**根本无页可恢复**（那才轮到 "no slot was designed" 的抛）；
    //   驻留与否是 stager 的事——旧判据把"非驻留"报成 -1 ⇒ stage-in 成了死代码（§31）。
    [[nodiscard]] ops::BlockSlot pending_slot(std::uint32_t position) const override {
        if (blocks_ == nullptr || position >= blocks_->size()) { return -1; }
        const LogicalKVPageHandle page =
            resolver_.block_handle(static_cast<std::uint32_t>((*blocks_)[position]));
        if (!page.valid()) { return -1; }
        return static_cast<ops::BlockSlot>(position);
    }

private:
    const LogicalKVPageStore& pages_;
    const KVMemSlotResolver&  resolver_;
    const std::vector<std::int32_t>* blocks_ = nullptr;
};

// ---- 按槽的三步恢复（stage-in）---------------------------------------------------------------
// 官方原话：*"The engine's implementation is the three-step restore (`reserve_device_replica` ->
// host-to-device copy -> `publish_device_replica`) against that slot's address entry."*
class KVMemPageStager final : public ops::PageStager {
public:
    KVMemPageStager(LogicalKVPageStore& pages, HostKVExtentStore& host,
                    const KVMemSlotResolver& resolver,
                    const std::vector<std::int32_t>* ordered_blocks, cudaStream_t stream)
        : pages_(pages), host_(host), resolver_(resolver), blocks_(ordered_blocks), stream_(stream) {}

    // 3060 round 10 (gather, ported from the Swift 1.5 engine): stage_in only reserves + publishes;
    // the H2D copies are queued and issued by flush() as one batched gather (per-page fallback).
    // Safe because (a) reserve/publish are metadata only, (b) evict_one never picks a page the new
    // window wants (every staged page is wanted), and (c) all copies are stream-ordered: a victim's
    // D2H is enqueued before the deferred H2D that may reuse its device page. The destructor
    // flushes too, so an aborted assembly never leaves a published page without its data.
    ~KVMemPageStager() {
        try { flush(); } catch (...) { std::fprintf(stderr, "[kvmem-si] gather flush failed in destructor\n"); }
    }
    KVMemPageStager(const KVMemPageStager&)            = delete;
    KVMemPageStager& operator=(const KVMemPageStager&) = delete;

    void flush() {
        if (pend_dst_.empty()) { return; }
        bool gathered = false;
        if (pend_dst_.size() >= 2) {
            std::vector<const std::byte*> records;
            records.reserve(pend_src_.size());
            const HostKVPageLayout* layout = &pend_src_.front().layout();
            bool same = true;
            for (const HostKVAllocationConstView& v : pend_src_) {
                if (!(v.layout() == *layout)) { same = false; break; }
                records.push_back(v.data());
            }
            if (same) {
                gathered = pages_.physical_pool().gather_from_host(
                    records, std::span<const DeviceKVPageHandle>(pend_dst_), *layout, stream_);
            }
        }
        if (!gathered) {
            for (std::size_t i = 0; i < pend_dst_.size(); ++i) {
                pages_.physical_pool().copy_from_host(
                    pend_src_[i], std::span<const DeviceKVPageHandle>(&pend_dst_[i], 1), stream_);
            }
        }
        static const bool trace = std::getenv("KV9_XFER_TRACE") != nullptr;
        if (trace) {
            std::fprintf(stderr, "[kvmem-si] flush pages=%zu gather=%d\n", pend_dst_.size(), gathered ? 1 : 0);
        }
        pend_src_.clear();
        pend_dst_.clear();
    }

    [[nodiscard]] bool device_resident(ops::BlockSlot slot) const override {
        const LogicalKVPageHandle page = page_for(slot);
        return page.valid() && pages_.device_resident(page);
    }

    // 要么成功（之后 `device_resident(slot)` 为真），要么返回 false 且**不留半成品** —— 调用方据此
    // 放弃装配，而不是装一个缺页的窗口（`kvmem_window_assembly.h:304-306`）。
    // ★ 2026-09-26：①按**块**取页（§31：窗位≠块号，V 必须来自块自己的页）；②**先 evict 后
    //   stage-in**——池小窗大时先把"旧窗里要离开的成员"搬去宿主腾位（release_stale_member 摘行 +
    //   还 active 引用，can_drop_device_replica 才肯放），腾一页换一页。
    [[nodiscard]] bool stage_in(ops::BlockSlot slot) override {
        const LogicalKVPageHandle page = page_for(slot);
        if (!page.valid()) {
            std::fprintf(stderr, "[kvmem-si] a: no page (stale/never registered) slot=%d\n",
                         static_cast<int>(slot));
            return false;
        }
        if (pages_.device_resident(page)) { return true; }   // 已在位：无动作
        if (!pages_.host_resident(page)) {
            std::fprintf(stderr, "[kvmem-si] b: no host replica slot=%d\n", static_cast<int>(slot));
            return false;   // 无宿主副本 ⇒ 拒答，不装来路不明的页
        }
        const HostKVPageReplica& replica = pages_.host_replica(page);
        if (!host_.valid(replica.extent)) {
            std::fprintf(stderr, "[kvmem-si] c: host extent invalid slot=%d\n",
                         static_cast<int>(slot));
            return false;
        }

        // 三步恢复（方向与压力路径的 stage_out 相反，逐条对应）：
        // ★ 预订单必须是**地址的预订单**（挂在池上、有配额）：默认构造的单子 resize 直接 bad allocation
        //   （SEM10/SEM11 实测）；引擎恢复事务同款来源（program_impl.h:4980-4982）。
        DeviceKVPageReservation& reservation =
            resolver_.addresses().page_reservation(resolver_.address());
        // ★ 授予配额：materialize_one 从预订单配额取页（每次 draw 减 1）；地址预订单的剩余配额
        //   已被既有 materialize 用尽（SEM13 实测 pool_free=23 仍败、且六项条件全过）⇒ 照恢复事务
        //   的做法 resize 授予（program_impl.h:4961）。draw 后配额自然回到原值，无需归还。
        pages_.physical_pool().resize_reservation(reservation, reservation.pages() + 1U);
        // 分辨探针：reserve 的六条拒绝项里，可从公开谓词读到的全打出来（host 副本是否"当前"是
        // 头号嫌疑：换出后页的 content_epoch/committed_columns 再动过 ⇒ 副本过期 ⇒ 拒绝恢复）。
        std::fprintf(stderr,
                     "[kvmem-si] reserve-pre slot=%d host_current=%d committed=%u pin=%d dev=%d\n",
                     static_cast<int>(slot), pages_.host_replica_current(page) ? 1 : 0,
                     pages_.committed_columns(page), pages_.can_pin_source(page) ? 1 : 0,
                     pages_.device_resident(page) ? 1 : 0);
        DeviceKVPageHandle destination = reserve_or_null(page, reservation);
        for (int attempt = 0; !destination.valid() && attempt < 64; ++attempt) {
            if (!evict_one()) { break; }
            destination = reserve_or_null(page, reservation);
        }
        if (!destination.valid()) {
            std::fprintf(stderr, "[kvmem-si] d: reserve+evict failed slot=%d pool_free=%u\n",
                         static_cast<int>(slot), pages_.physical_pool().available_pages());
            return false;
        }
        // 签名：copy_from_host(host 视图, span<const DeviceKVPageHandle>, stream) ⇒ 单页也要包成 span。
        if (defer()) {
            const HostKVAllocationConstView src = host_.view(replica.extent);
            if (!src.valid() || src.page_count() != 1) {
                throw std::invalid_argument("Paged KV H2D geometry or extent is inconsistent");
            }
            pend_src_.push_back(src);
            pend_dst_.push_back(destination);
        } else {
            const DeviceKVPageHandle destinations[1] = {destination};
            pages_.physical_pool().copy_from_host(host_.view(replica.extent),
                                                  std::span<const DeviceKVPageHandle>(destinations, 1),
                                                  stream_);
        }
        pages_.publish_device_replica(page);
        if (!pages_.device_resident(page)) { return false; }
        ++staged_;
        return true;
    }

    [[nodiscard]] std::uint32_t staged_count() const noexcept { return staged_; }

private:
    [[nodiscard]] LogicalKVPageHandle page_for(ops::BlockSlot slot) const {
        if (blocks_ == nullptr || slot < 0 ||
            static_cast<std::size_t>(slot) >= blocks_->size()) {
            return {};
        }
        return resolver_.block_handle(
            static_cast<std::uint32_t>((*blocks_)[static_cast<std::size_t>(slot)]));
    }

    [[nodiscard]] DeviceKVPageHandle reserve_or_null(LogicalKVPageHandle page,
                                                     DeviceKVPageReservation& reservation) {
        try {
            return pages_.reserve_device_replica(page, reservation);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[kvmem-diag] reserve exception: %s\n", e.what());
            return {};
        } catch (...) {
            std::fprintf(stderr, "[kvmem-diag] reserve nonstandard exception\n");
            return {};
        }
    }

    // 腾位：从**旧窗成员**里挑一个"新窗不要的整页"，走压力路径同款三步搬到宿主后放掉设备副本。
    [[nodiscard]] bool evict_one() {
        std::uint32_t seen = 0, stale = 0, not_resident = 0, half = 0, wanted_cnt = 0,
                     kept_row = 0, unpinned = 0, prepared = 0, dropped = 0;
        for (const LogicalKVPageHandle victim : resolver_.window_members()) {
            ++seen;
            if (!pages_.valid(victim)) { ++stale; continue; }
            if (!pages_.device_resident(victim)) { ++not_resident; continue; }
            if (pages_.committed_columns(victim) <
                static_cast<std::uint32_t>(kPagedKVPageSize)) {
                ++half;
                continue;   // 半写尾页不换出（coverage 契约）
            }
            bool wanted = false;
            if (blocks_ != nullptr) {
                for (const std::int32_t block : *blocks_) {
                    if (resolver_.block_handle(static_cast<std::uint32_t>(block)) == victim) {
                        wanted = true;
                        break;
                    }
                }
            }
            if (wanted) { ++wanted_cnt; continue; }   // 新窗还要它，不许动
            const bool released =
                resolver_.addresses().release_stale_member(resolver_.address(), victim);
            if (!released) {
                ++kept_row;
                if (kept_row <= 8U) {
                    std::fprintf(stderr,
                        "[kvmem-diag] nonmember refs=%u active=%u writer=%u pins=%u host_current=%d\n",
                        pages_.address_references(victim), pages_.active_address_references(victim),
                        static_cast<unsigned>(pages_.writer_references(victim)),
                        pages_.source_pins(victim), pages_.host_replica_current(victim) ? 1 : 0);
                }
                const char* fix = std::getenv("NINFER_KVMEM_ORPHAN_EVICT_FIX");
                // No active owner or writer may be bypassed. Final drop still uses all store guards.
                if (fix == nullptr || fix[0] != '1' ||
                    pages_.active_address_references(victim) != 0 ||
                    pages_.writer_references(victim) != 0 || pages_.source_pins(victim) != 0) {
                    continue;
                }
                std::fprintf(stderr, "[kvmem-diag] eligible nonmember eviction candidate\n");
            }
            if (released && pages_.writer_references(victim) != 0) {
                pages_.set_writer(victim, false);
            }
            if (!pages_.can_pin_source(victim)) { ++unpinned; continue; }
            // Revision 2: prepare() rejects host-resident pages. Reuse a current,
            // valid host replica rather than attempting a duplicate D2H reservation.
            const char* reuse_fix = std::getenv("NINFER_KVMEM_ORPHAN_EVICT_FIX");
            if (reuse_fix != nullptr && reuse_fix[0] == '1' && pages_.host_resident(victim)) {
                if (pages_.host_replica_current(victim) &&
                    host_.valid(pages_.host_replica(victim).extent) &&
                    pages_.can_drop_device_replica(victim) && pages_.drop_device_replica(victim)) {
                    std::fprintf(stderr, "[kvmem-diag-v2] dropped device; reused current host replica\n");
                    ++dropped;
                    return true;
                }
                continue;
            }
            std::vector<LogicalKVPageHandle> one{victim};
            std::optional<HostKVExtentReservation> reserved = host_.prepare(pages_, one);
            if (!reserved) { continue; }
            ++prepared;
            std::vector<DeviceKVPageHandle> sources(1);
            host_.device_sources(*reserved, sources);
            pages_.physical_pool().copy_to_host(sources, host_.writable_view(*reserved), stream_);
            (void)host_.publish(std::move(*reserved));
            if (pages_.drop_device_replica(victim)) { ++dropped; return true; }
        }
        std::fprintf(stderr,
                     "[kvmem-si] evict scan seen=%u stale=%u not_resident=%u half=%u wanted=%u "
                     "kept_row=%u unpinned=%u prepared=%u dropped=%u\n",
                     seen, stale, not_resident, half, wanted_cnt, kept_row, unpinned, prepared,
                     dropped);
        return false;
    }

    [[nodiscard]] static bool defer() noexcept {
        static const bool on = std::getenv("KV9_NOGATHER") == nullptr;
        return on;
    }

    LogicalKVPageStore&       pages_;
    HostKVExtentStore&        host_;
    const KVMemSlotResolver&  resolver_;
    const std::vector<std::int32_t>* blocks_ = nullptr;
    cudaStream_t              stream_;
    std::uint32_t             staged_ = 0;
    std::vector<HostKVAllocationConstView> pend_src_;   // 3060 gather: queued H2D sources
    std::vector<DeviceKVPageHandle>        pend_dst_;   // ... and their device pages
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
