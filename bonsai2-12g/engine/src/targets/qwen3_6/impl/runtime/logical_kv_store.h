#pragma once

#include <cstdlib>  // KVMem kv8c

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class LogicalKVPageStore;
class KVAddressSpaceStore;
class HostKVExtentStore;
class KVPrefixForkReservation;
class KVActiveSnapshotReservation;

class HostKVExtentCapability {
public:
    HostKVExtentCapability() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(HostKVExtentCapability,
                                         HostKVExtentCapability) noexcept = default;

private:
    HostKVExtentCapability(const HostKVExtentStore* owner, std::uint32_t index,
                           std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const HostKVExtentStore* owner_ = nullptr;
    std::uint32_t index_            = 0;
    std::uint32_t generation_       = 0;

    friend class HostKVExtentStore;
};

struct HostKVPageReplica {
    HostKVExtentCapability extent;
    std::uint32_t page_offset       = 0;
    std::uint32_t membership_node   = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
};

class LogicalKVPageHandle {
public:
    LogicalKVPageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(LogicalKVPageHandle,
                                         LogicalKVPageHandle) noexcept = default;

private:
    LogicalKVPageHandle(const LogicalKVPageStore* owner, std::uint32_t index,
                        std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const LogicalKVPageStore* owner_ = nullptr;
    std::uint32_t index_             = 0;
    std::uint32_t generation_        = 0;

    friend class LogicalKVPageStore;
    friend class HostKVExtentStore;
};

class KVAddressSpaceHandle {
public:
    KVAddressSpaceHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(KVAddressSpaceHandle,
                                         KVAddressSpaceHandle) noexcept = default;

private:
    KVAddressSpaceHandle(const KVAddressSpaceStore* owner, std::uint32_t index,
                         std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const KVAddressSpaceStore* owner_ = nullptr;
    std::uint32_t index_              = 0;
    std::uint32_t generation_         = 0;

    friend class KVAddressSpaceStore;
};

class KVActivationReservation {
public:
    KVActivationReservation() noexcept = default;

    KVActivationReservation(KVActivationReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), address_(other.address_),
          requested_entitlement_(other.requested_entitlement_),
          activation_frontier_(other.activation_frontier_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)) {}

    KVActivationReservation& operator=(KVActivationReservation&&)      = delete;
    KVActivationReservation(const KVActivationReservation&)            = delete;
    KVActivationReservation& operator=(const KVActivationReservation&) = delete;

private:
    KVActivationReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle address,
                            std::uint32_t requested_entitlement,
                            std::optional<std::uint32_t> activation_frontier,
                            DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row) noexcept
        : owner_(&owner), address_(address), requested_entitlement_(requested_entitlement),
          activation_frontier_(activation_frontier), page_reservation_(std::move(page_reservation)),
          row_(std::move(row)) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle address_;
    std::uint32_t requested_entitlement_ = 0;
    std::optional<std::uint32_t> activation_frontier_;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;

    friend class KVAddressSpaceStore;
};

class KVPrefixForkReservation {
public:
    KVPrefixForkReservation() noexcept = default;
    ~KVPrefixForkReservation();

    KVPrefixForkReservation(KVPrefixForkReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), source_(other.source_),
          destination_(other.destination_), frontier_(other.frontier_),
          full_pages_(other.full_pages_), tail_columns_(other.tail_columns_),
          requested_entitlement_(other.requested_entitlement_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)),
          tail_destination_(other.tail_destination_),
          staged_tail_release_(other.staged_tail_release_),
          tail_source_settled_(other.tail_source_settled_) {
        other.tail_destination_.reset();
    }

    KVPrefixForkReservation& operator=(KVPrefixForkReservation&&)      = delete;
    KVPrefixForkReservation(const KVPrefixForkReservation&)            = delete;
    KVPrefixForkReservation& operator=(const KVPrefixForkReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return tail_columns_ != 0; }

private:
    KVPrefixForkReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle source,
                            KVAddressSpaceHandle destination, std::uint32_t frontier,
                            std::uint32_t full_pages, std::uint32_t tail_columns,
                            std::uint32_t requested_entitlement,
                            DeviceKVPageReservation&& page_reservation, KVExecutionRowLease&& row,
                            std::optional<LogicalKVPageHandle> tail_destination,
                            bool staged_tail_release) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          full_pages_(full_pages), tail_columns_(tail_columns),
          requested_entitlement_(requested_entitlement),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)),
          tail_destination_(tail_destination), staged_tail_release_(staged_tail_release) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_              = 0;
    std::uint32_t full_pages_            = 0;
    std::uint32_t tail_columns_          = 0;
    std::uint32_t requested_entitlement_ = 0;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;
    std::optional<LogicalKVPageHandle> tail_destination_;
    bool staged_tail_release_ = false;
    bool tail_source_settled_ = false;

    friend class KVAddressSpaceStore;
};

struct KVActiveSnapshotShape {
    std::uint32_t full_pages           = 0;
    std::uint32_t immutable_full_pages = 0;
    std::uint32_t unique_full_pages    = 0;
    std::uint32_t tail_columns         = 0;

    [[nodiscard]] std::uint32_t copied_pages() const noexcept {
        return tail_columns == 0 ? 0U : 1U;
    }

    [[nodiscard]] friend bool operator==(KVActiveSnapshotShape,
                                         KVActiveSnapshotShape) noexcept = default;
};

// Transaction-exclusive conversion of one active address into an immutable checkpoint while
// installing a new active writer address. An active address may begin with aliased immutable full
// pages and end with unique mutable pages. Full pages are referenced in place; only a non-aligned
// mutable tail receives a new physical page.
class KVActiveSnapshotReservation {
public:
    KVActiveSnapshotReservation() noexcept = default;
    ~KVActiveSnapshotReservation();

    KVActiveSnapshotReservation(KVActiveSnapshotReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), source_(other.source_),
          destination_(other.destination_), frontier_(other.frontier_), shape_(other.shape_),
          entitlement_(other.entitlement_), tail_reservation_(std::move(other.tail_reservation_)),
          tail_destination_(other.tail_destination_) {
        other.tail_destination_.reset();
    }

    KVActiveSnapshotReservation& operator=(KVActiveSnapshotReservation&&)      = delete;
    KVActiveSnapshotReservation(const KVActiveSnapshotReservation&)            = delete;
    KVActiveSnapshotReservation& operator=(const KVActiveSnapshotReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return shape_.copied_pages() != 0; }

private:
    KVActiveSnapshotReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle source,
                                KVAddressSpaceHandle destination, std::uint32_t frontier,
                                KVActiveSnapshotShape shape, std::uint32_t entitlement,
                                DeviceKVPageReservation&& tail_reservation,
                                std::optional<LogicalKVPageHandle> tail_destination) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          shape_(shape), entitlement_(entitlement), tail_reservation_(std::move(tail_reservation)),
          tail_destination_(tail_destination) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_ = 0;
    KVActiveSnapshotShape shape_;
    std::uint32_t entitlement_ = 0;
    DeviceKVPageReservation tail_reservation_;
    std::optional<LogicalKVPageHandle> tail_destination_;

    friend class KVAddressSpaceStore;
};

class LogicalKVPageStore {
public:
    LogicalKVPageStore(DeviceKVPagePool& physical, std::uint32_t logical_capacity)
        : physical_(&physical), pages_(logical_capacity), free_(pages_.size()),
          free_count_(static_cast<std::uint32_t>(pages_.size())) {
        if (pages_.empty() || logical_capacity < physical.capacity_pages()) {
            throw std::invalid_argument("Logical KV page capacity is inconsistent");
        }
        for (std::uint32_t index = 0; index < pages_.size(); ++index) {
            free_[index] = static_cast<std::uint32_t>(pages_.size()) - 1U - index;
        }
        materialization_scratch_.reserve(physical.capacity_pages());
    }

    LogicalKVPageStore(const LogicalKVPageStore&)            = delete;
    LogicalKVPageStore& operator=(const LogicalKVPageStore&) = delete;
    LogicalKVPageStore(LogicalKVPageStore&&)                 = delete;
    LogicalKVPageStore& operator=(LogicalKVPageStore&&)      = delete;

    [[nodiscard]] DeviceKVPagePool& physical_pool() noexcept { return *physical_; }

    [[nodiscard]] const DeviceKVPagePool& physical_pool() const noexcept { return *physical_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(pages_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] LogicalKVPageHandle materialize(DeviceKVPageReservation& reservation) {
        if (free_count_ == 0) {
            throw std::logic_error("logical KV descriptors exhausted before physical capacity");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(lease));
        page.content_epoch     = next_epoch(page.content_epoch);
        page.committed_columns = 0;
        page.references        = 1;
        page.active_references = 0;
        page.writer_references = 1;
        page.protected_columns = 0;
        page.occupied          = true;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    void materialize(DeviceKVPageReservation& reservation,
                     std::span<LogicalKVPageHandle> destinations,
                     std::optional<LogicalKVPageHandle> preferred_predecessor = std::nullopt) {
        if (destinations.empty()) { return; }
        if (destinations.size() > free_count_ ||
            destinations.size() > materialization_scratch_.capacity()) {
            throw std::logic_error("logical KV batch materialization exceeds descriptor capacity");
        }
        for (const LogicalKVPageHandle destination : destinations) {
            if (destination.valid()) {
                throw std::logic_error("logical KV batch destination is already populated");
            }
        }
        for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
            const std::uint32_t index = free_[free_count_ - 1U - offset];
            if (pages_[index].occupied) {
                throw std::logic_error("logical KV free descriptor is occupied");
            }
        }

        std::optional<DeviceKVPageHandle> physical_predecessor;
        if (preferred_predecessor) { physical_predecessor = physical(*preferred_predecessor); }
        materialization_scratch_.clear();
        physical_->materialize(reservation, static_cast<std::uint32_t>(destinations.size()),
                               materialization_scratch_, physical_predecessor);
        for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
            const std::uint32_t index = free_[free_count_ - 1U - offset];
            Page& page                = pages_[index];
            page.device_replica.emplace(std::move(materialization_scratch_[offset]));
            page.content_epoch     = next_epoch(page.content_epoch);
            page.committed_columns = 0;
            page.references        = 1;
            page.active_references = 0;
            page.writer_references = 1;
            page.protected_columns = 0;
            page.occupied          = true;
            destinations[offset]   = LogicalKVPageHandle(this, index, page.generation);
        }
        free_count_ -= static_cast<std::uint32_t>(destinations.size());
        materialization_scratch_.clear();
    }

    [[nodiscard]] LogicalKVPageHandle
    materialize_transfer_destination(DeviceKVPageReservation& reservation,
                                     std::uint32_t committed_columns) {
        if (committed_columns == 0 ||
            committed_columns > static_cast<std::uint32_t>(kPagedKVPageSize) || free_count_ == 0) {
            throw std::invalid_argument("logical KV transfer destination is invalid");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(lease));
        page.content_epoch      = next_epoch(page.content_epoch);
        page.committed_columns  = committed_columns;
        page.references         = 0;
        page.active_references  = 0;
        page.writer_references  = 0;
        page.protected_columns  = 0;
        page.destination_pinned = true;
        page.occupied           = true;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    void publish_transfer_destination(LogicalKVPageHandle handle, bool writer) noexcept {
        if (!valid(handle)) { std::terminate(); }
        Page& page = pages_[handle.index_];
        if (!page.device_replica || page.pending_device_replica || page.host_replica ||
            !page.destination_pinned || page.references != 0 || page.writer_references != 0 ||
            page.source_pins != 0) {
            std::terminate();
        }
        page.references         = 1;
        page.writer_references  = writer ? 1U : 0U;
        page.destination_pinned = false;
    }

    void abort_transfer_destination(LogicalKVPageHandle handle,
                                    DeviceKVPageReservation& reservation) noexcept {
        if (!valid(handle)) { return; }
        Page& page = pages_[handle.index_];
        if (!page.device_replica || page.pending_device_replica || page.host_replica ||
            !page.destination_pinned || page.references != 0 || page.writer_references != 0 ||
            page.source_pins != 0) {
            return;
        }
        physical_->dematerialize_one(reservation, std::move(*page.device_replica));
        page.device_replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool valid(LogicalKVPageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < pages_.size() &&
               pages_[handle.index_].occupied &&
               pages_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] std::uint32_t descriptor_index(LogicalKVPageHandle handle) const {
        (void)require(handle);
        return handle.index_;
    }

    [[nodiscard]] DeviceKVPageHandle physical(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        if (!page.device_replica) {
            throw std::logic_error("logical KV page has no Device replica");
        }
        return page.device_replica->handle();
    }

    [[nodiscard]] bool device_resident(LogicalKVPageHandle handle) const {
        return require(handle).device_replica.has_value();
    }

    [[nodiscard]] bool host_resident(LogicalKVPageHandle handle) const {
        return require(handle).host_replica.has_value();
    }

    [[nodiscard]] const HostKVPageReplica& host_replica(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        if (!page.host_replica) { throw std::logic_error("logical KV page has no Host replica"); }
        return *page.host_replica;
    }

    [[nodiscard]] std::uint64_t content_epoch(LogicalKVPageHandle handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t committed_columns(LogicalKVPageHandle handle) const {
        return require(handle).committed_columns;
    }

    [[nodiscard]] std::uint32_t address_references(LogicalKVPageHandle handle) const {
        return require(handle).references;
    }

    [[nodiscard]] std::uint32_t active_address_references(LogicalKVPageHandle handle) const {
        return require(handle).active_references;
    }

    [[nodiscard]] std::uint8_t writer_references(LogicalKVPageHandle handle) const {
        return require(handle).writer_references;
    }

    [[nodiscard]] std::uint32_t protected_columns(LogicalKVPageHandle handle) const {
        return require(handle).protected_columns;
    }

    [[nodiscard]] std::uint32_t source_pins(LogicalKVPageHandle handle) const {
        return require(handle).source_pins;
    }

    [[nodiscard]] bool can_pin_source(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica.has_value() && page.writer_references == 0 &&
               !page.destination_pinned &&
               page.source_pins != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] bool can_pin_active_source(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica.has_value() && page.writer_references == 1 &&
               page.references == 1 && !page.destination_pinned &&
               page.source_pins != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] DeviceKVPageHandle reserve_device_replica(LogicalKVPageHandle handle,
                                                            DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.device_replica || page.pending_device_replica || !page.host_replica ||
            page.destination_pinned || page.source_pins != 0 ||
            page.host_replica->content_epoch != page.content_epoch ||
            page.host_replica->committed_columns != page.committed_columns) {
            std::fprintf(stderr,
                         "[kv-restore] NOT RESERVABLE dev=%d pending=%d host=%d dest_pin=%d "
                         "src_pins=%u epoch_ok=%d cols_ok=%d\n",
                         page.device_replica.has_value() ? 1 : 0,
                         page.pending_device_replica.has_value() ? 1 : 0,
                         page.host_replica.has_value() ? 1 : 0, page.destination_pinned ? 1 : 0,
                         page.source_pins,
                         page.host_replica && page.host_replica->content_epoch == page.content_epoch
                             ? 1
                             : 0,
                         page.host_replica && page.host_replica->committed_columns ==
                                                   page.committed_columns
                             ? 1
                             : 0);
            throw std::logic_error("logical KV Device restore is not reservable");
        }
        page.pending_device_replica.emplace(physical_->materialize_one(reservation));
        page.destination_pinned = true;
        return page.pending_device_replica->handle();
    }

    void publish_device_replica(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.device_replica || !page.pending_device_replica || !page.destination_pinned ||
            !page.host_replica || page.host_replica->content_epoch != page.content_epoch ||
            page.host_replica->committed_columns != page.committed_columns) {
            throw std::logic_error("logical KV Device restore is not publishable");
        }
        page.device_replica = std::move(page.pending_device_replica);
        page.pending_device_replica.reset();
        page.destination_pinned = false;
    }

    void abort_device_replica(LogicalKVPageHandle handle,
                              DeviceKVPageReservation& reservation) noexcept {
        if (!valid(handle)) { return; }
        Page& page = pages_[handle.index_];
        if (!page.pending_device_replica || !page.destination_pinned) { return; }
        physical_->dematerialize_one(reservation, std::move(*page.pending_device_replica));
        page.pending_device_replica.reset();
        page.destination_pinned = false;
    }

    // KVMem v5: number of device-resident pages held only by idle cache owners (referenced, no
    // active reference / writer / pin / pending replica), minus `exclude`. Read only.
    [[nodiscard]] std::uint32_t
    kvmem_idle_resident_count(std::span<const LogicalKVPageHandle> exclude) const {
        std::vector<char> skip(pages_.size(), 0);
        for (const LogicalKVPageHandle h : exclude) {
            if (valid(h) && h.index_ < skip.size()) { skip[h.index_] = 1; }
        }
        std::uint32_t n = 0;
        for (std::size_t i = 0; i < pages_.size(); ++i) {
            if (skip[i]) { continue; }
            const Page& page = pages_[i];
            if (!page.occupied || !page.device_replica || page.pending_device_replica) { continue; }
            if (page.references == 0 || page.active_references != 0 || page.source_pins != 0 ||
                page.destination_pinned || page.writer_references != 0) {
                continue;
            }
            ++n;
        }
        return n;
    }

    // KVMem P3d-1: census of device-resident pages; with sweep, drop every replica that is safe to
    // drop (current host copy, no writer/active/pin). Returns a printable summary.
    [[nodiscard]] std::string kvmem_census(bool sweep) noexcept {
        std::uint32_t resident = 0, host_cur = 0, writer = 0, active = 0, pinned = 0, pending = 0,
                      nohost = 0, orphan = 0, droppable = 0, dropped = 0;
        for (std::size_t i = 0; i < pages_.size(); ++i) {
            Page& page = pages_[i];
            if (!page.occupied || !page.device_replica) { continue; }
            ++resident;
            const bool cur = page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
                             page.host_replica->committed_columns == page.committed_columns;
            if (cur) { ++host_cur; } else { ++nohost; }
            if (page.writer_references) { ++writer; }
            if (page.active_references) { ++active; }
            if (page.source_pins || page.destination_pinned) { ++pinned; }
            if (page.pending_device_replica) { ++pending; }
            if (page.active_references == 0 && page.writer_references == 0) { ++orphan; }
            const bool ok = cur && !page.pending_device_replica && page.writer_references == 0 &&
                            page.active_references == 0 && page.source_pins == 0 &&
                            !page.destination_pinned;
            if (ok) {
                ++droppable;
                if (sweep) { page.device_replica.reset(); ++dropped; }
            }
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "resident=%u host_cur=%u nohost=%u writer=%u active=%u pinned=%u pending=%u "
                      "idle=%u droppable=%u dropped=%u",
                      resident, host_cur, nohost, writer, active, pinned, pending, orphan, droppable,
                      dropped);
        return std::string(buf);
    }

    [[nodiscard]] bool drop_device_replica(LogicalKVPageHandle handle) noexcept {
        if (!can_drop_device_replica(handle)) { return false; }
        Page& page = pages_[handle.index_];
        page.device_replica.reset();
        return true;
    }

    [[nodiscard]] bool host_replica_current(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
               page.host_replica->committed_columns == page.committed_columns;
    }

    // STORAGE DECOUPLING (2026-09-26, class A / narrow form): a page whose coverage is COMPLETE can
    // never receive another coverage commit (commit_coverage() rejects columns < committed_columns),
    // so its writer flag no longer carries information -- it exists only to license those commits.
    // Such a page is safe to demote provided every OTHER invariant still holds. This is deliberately
    // narrower than relaxing writer_references globally: the in-flight tail page (coverage < page_size)
    // keeps its writer flag and therefore stays resident, which is what the coverage contract requires.
    [[nodiscard]] bool can_drop_committed_device_replica(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica && host_replica_current(handle) &&
               !page.pending_device_replica && page.active_references == 0 &&
               page.source_pins == 0 && !page.destination_pinned &&
               page.committed_columns >= static_cast<std::uint32_t>(kPagedKVPageSize);
    }

    // Same preconditions, but the page may still be flagged as a writer -- dropping a device replica
    // does NOT clear committed coverage, so the page can still be committed later if it is staged back
    // in. Returns false (rather than throwing) when the page is not eligible.
    [[nodiscard]] bool drop_committed_device_replica(LogicalKVPageHandle handle) noexcept {
        if (!can_drop_committed_device_replica(handle)) { return false; }
        Page& page = pages_[handle.index_];
        page.device_replica.reset();
        return true;
    }
    [[nodiscard]] bool can_drop_device_replica(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica && host_replica_current(handle) &&
               !page.pending_device_replica && page.writer_references == 0 &&
               page.active_references == 0 && page.source_pins == 0 && !page.destination_pinned;
    }

    void retain_active_reference(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.active_references == std::numeric_limits<std::uint32_t>::max() ||
            page.active_references >= page.references) {
            throw std::logic_error("logical KV active reference is not retainable");
        }
        ++page.active_references;
    }

    void release_active_reference(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.active_references == 0) {
            throw std::logic_error("logical KV page has no active reference");
        }
        --page.active_references;
    }

    void retain_reference(LogicalKVPageHandle handle, bool writer) {
        Page& page = require(handle);
        if (page.references == std::numeric_limits<std::uint32_t>::max() ||
            (writer && page.writer_references != 0)) {
            throw std::logic_error("logical KV page reference is not retainable");
        }
        ++page.references;
        if (writer) { page.writer_references = 1; }
    }

    [[nodiscard]] bool can_retain_reference(LogicalKVPageHandle handle,
                                            bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references != std::numeric_limits<std::uint32_t>::max() &&
               (!writer || (page.writer_references == 0 && page.references == 0)) &&
               !page.destination_pinned;
    }

    void set_writer(LogicalKVPageHandle handle, bool writer) {
        Page& page = require(handle);
        if (writer && page.writer_references != 0) {
            throw std::logic_error("logical KV page already has a writer");
        }
        page.writer_references = writer ? 1U : 0U;
    }

    [[nodiscard]] bool can_set_writer(LogicalKVPageHandle handle, bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return writer ? page.writer_references == 0 && page.references == 1 &&
                            page.source_pins == 0 && page.device_replica.has_value()
                      : page.writer_references == 1;
    }

    void protect_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV protection exceeds committed coverage");
        }
        page.protected_columns = std::max(page.protected_columns, columns);
    }

    void set_protected_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV protection exceeds committed coverage");
        }
        page.protected_columns = columns;
    }

    void clear_protection(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.references != 1 || page.source_pins != 0) {
            throw std::logic_error("shared logical KV protection cannot be cleared in place");
        }
        page.protected_columns = 0;
    }

    void pin_source(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("logical KV source pin count overflow");
        }
        ++page.source_pins;
    }

    void unpin_source(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.source_pins == 0) { throw std::logic_error("logical KV source is not pinned"); }
        --page.source_pins;
    }

    void commit_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (!page.device_replica || page.writer_references != 1 ||
            columns < page.committed_columns ||
            columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
            throw std::invalid_argument("logical KV committed coverage is not monotonic");
        }
        page.committed_columns = columns;
    }

    void destructive_truncate(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns || columns < page.protected_columns ||
            page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
            page.destination_pinned || page.host_replica) {
            throw std::invalid_argument("logical KV page is not destructively truncatable");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
        }
    }

    [[nodiscard]] bool can_destructive_truncate(LogicalKVPageHandle handle,
                                                std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && columns >= page.protected_columns &&
               page.references == 1 && page.writer_references == 1 && page.source_pins == 0 &&
               !page.destination_pinned && !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool can_destructive_truncate_inactive(LogicalKVPageHandle handle,
                                                         std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && page.references == 1 &&
               page.writer_references == 0 && page.source_pins == 0 && !page.destination_pinned &&
               !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool
    can_destructive_truncate_inactive_after_host_release(LogicalKVPageHandle handle,
                                                         std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && page.references == 1 &&
               page.writer_references == 0 && page.source_pins == 0 && !page.destination_pinned &&
               page.device_replica.has_value();
    }

    void destructive_truncate_inactive(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (!can_destructive_truncate_inactive(handle, columns)) {
            throw std::logic_error("inactive logical KV page is not destructively truncatable");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
        }
        page.protected_columns = std::min(page.protected_columns, columns);
    }

    [[nodiscard]] bool can_dematerialize(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references == 1 && page.writer_references == 1 && page.source_pins == 0 &&
               !page.destination_pinned && !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool can_release_reference(LogicalKVPageHandle handle,
                                             bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        if (page.references == 0 || (writer && page.writer_references != 1) ||
            page.active_references >= page.references || page.source_pins != 0 ||
            page.destination_pinned) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_reference_after_active_reference(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references != 0 && page.active_references != 0 &&
               page.active_references <= page.references && page.writer_references <= 1 &&
               page.source_pins == 0 && !page.destination_pinned;
    }

    [[nodiscard]] bool release_reference(LogicalKVPageHandle handle, bool writer) noexcept {
        if (!can_release_reference(handle, writer)) { return false; }
        Page& page = pages_[handle.index_];
        if (writer) { page.writer_references = 0; }
        if (--page.references != 0) { return true; }
        page.device_replica.reset();
        // A Host extent owns its page capability until the extent is released as one atomic
        // allocation. Keep the zero-reference logical descriptor alive long enough for
        // HostKVExtentStore to detach that replica; without a Host replica the descriptor can be
        // reclaimed immediately.
        if (!page.host_replica) { release_descriptor(handle, page); }
        return true;
    }

    [[nodiscard]] bool can_attach_host_replica(LogicalKVPageHandle handle, std::uint64_t epoch,
                                               std::uint32_t coverage) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return !page.host_replica && epoch == page.content_epoch &&
               coverage == page.committed_columns;
    }

    void attach_host_replica(LogicalKVPageHandle handle, HostKVPageReplica replica) {
        Page& page = require(handle);
        if (!replica.extent.valid() ||
            replica.membership_node == std::numeric_limits<std::uint32_t>::max() ||
            !can_attach_host_replica(handle, replica.content_epoch, replica.committed_columns)) {
            throw std::logic_error("logical KV Host replica is not publishable");
        }
        page.host_replica = replica;
    }

    void rebind_host_replica(LogicalKVPageHandle handle, HostKVPageReplica expected,
                             HostKVPageReplica replacement) noexcept {
        if (!valid(handle)) { std::terminate(); }
        Page& page = pages_[handle.index_];
        if (!page.host_replica || page.host_replica->extent != expected.extent ||
            page.host_replica->page_offset != expected.page_offset ||
            page.host_replica->membership_node != expected.membership_node ||
            page.host_replica->content_epoch != expected.content_epoch ||
            page.host_replica->committed_columns != expected.committed_columns ||
            expected.content_epoch != page.content_epoch || !replacement.extent.valid() ||
            replacement.membership_node != expected.membership_node ||
            replacement.content_epoch != expected.content_epoch ||
            replacement.committed_columns != expected.committed_columns) {
            std::terminate();
        }
        page.host_replica = replacement;
    }

    [[nodiscard]] bool detach_host_replica(LogicalKVPageHandle handle,
                                           HostKVExtentCapability extent) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        if (!page.host_replica || page.host_replica->extent != extent ||
            (!page.device_replica && page.references != 0)) {
            return false;
        }
        page.host_replica.reset();
        if (page.references == 0 && page.source_pins == 0 && !page.device_replica) {
            release_descriptor(handle, page);
        }
        return true;
    }

    void dematerialize(LogicalKVPageHandle handle, DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
            page.destination_pinned || page.host_replica || !page.device_replica) {
            throw std::logic_error("logical KV page is shared or has no Device replica");
        }
        physical_->dematerialize_one(reservation, std::move(*page.device_replica));
        page.device_replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool release(LogicalKVPageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        return release_reference(handle, page.writer_references != 0);
    }

private:
    struct Page {
        std::uint32_t generation        = 1;
        std::uint64_t content_epoch     = 0;
        std::uint32_t committed_columns = 0;
        std::uint32_t references        = 0;
        std::uint32_t active_references = 0;
        std::uint32_t protected_columns = 0;
        std::uint32_t source_pins       = 0;
        std::uint8_t writer_references  = 0;
        bool destination_pinned         = false;
        bool occupied                   = false;
        std::optional<DeviceKVPageLease> device_replica;
        std::optional<DeviceKVPageLease> pending_device_replica;
        std::optional<HostKVPageReplica> host_replica;
    };

    [[nodiscard]] static std::uint64_t next_epoch(std::uint64_t epoch) noexcept {
        ++epoch;
        return epoch == 0 ? 1 : epoch;
    }

    [[nodiscard]] Page& require(LogicalKVPageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    [[nodiscard]] const Page& require(LogicalKVPageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    void release_descriptor(LogicalKVPageHandle handle, Page& page) noexcept {
        if (page.active_references != 0) { std::terminate(); }
        page.committed_columns  = 0;
        page.references         = 0;
        page.active_references  = 0;
        page.protected_columns  = 0;
        page.source_pins        = 0;
        page.writer_references  = 0;
        page.destination_pinned = false;
        page.occupied           = false;
        page.pending_device_replica.reset();
        page.host_replica.reset();
        if (++page.generation == 0) { ++page.generation; }
        free_[free_count_++] = handle.index_;
    }

    DeviceKVPagePool* physical_ = nullptr;
    std::vector<Page> pages_;
    std::vector<std::uint32_t> free_;
    std::vector<DeviceKVPageLease> materialization_scratch_;
    std::uint32_t free_count_ = 0;
};

// ---------------------------------------------------------------------------------------------
// KVMem WINDOW ASSEMBLY (the P2 shape of the KV workspace)
//
// WHY THIS EXISTS
//   This engine has no attention mask: the visible set is a CONTIGUOUS PREFIX [0, p] and the live
//   positions must be SEQUENTIAL (include/ninfer/ops/softmax_attention.h:118-123). Holes cannot be
//   expressed at all. So KVMem's "select an arbitrary subset of KV pages" has to become "assemble
//   those pages into a compact window" BEFORE attention runs. That transformation is exactly
//   "reorder the ordered-membership row": publish_membership() already knows how to turn that row into
//   page-table entries -- it was merely private and could only publish the row as it stood.
//
// WHAT THE CALLER STILL HAS TO DO (this is why the struct carries `tokens`)
//   Row r of a KV execution table is assumed to be at position r, and the K stored in those rows is
//   RoPE-baked at its stored position. So after assembling a window: the next token's cache position
//   must be `tokens - 1` inside the window (not the sequence's raw frontier), and the attention
//   envelope must cover `tokens`. An envelope SMALLER than the window throws; a larger one only
//   over-commits the graph, which is why the wiring has to pass the window length through.
// ---------------------------------------------------------------------------------------------
struct KVWindowAssembly {
    std::uint32_t pages  = 0;  // logical pages in the window (== ordered_pages.size())
    std::uint32_t tokens = 0;  // compact window length; the last page may be a partial tail
};

class KVAddressSpaceStore {
public:
    KVAddressSpaceStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                        std::uint32_t address_capacity, std::uint32_t page_capacity)
        : pages_(&pages), tables_(&tables), page_capacity_(page_capacity),
          addresses_(address_capacity), free_(address_capacity),
          memberships_(checked_membership_cells(address_capacity, page_capacity)),
          block_pages_(checked_membership_cells(address_capacity, page_capacity)),
          block_k_baked_at_(checked_membership_cells(address_capacity, page_capacity),
                            ~std::uint32_t{0}),
          free_count_(address_capacity) {
        if (address_capacity == 0 || page_capacity == 0 ||
            page_capacity != tables.logical_page_capacity()) {
            throw std::invalid_argument("KV address-space geometry is invalid");
        }
        for (std::uint32_t index = 0; index < address_capacity; ++index) {
            free_[index] = address_capacity - 1U - index;
        }
        publish_scratch_.reserve(page_capacity_);
    }

    KVAddressSpaceStore(const KVAddressSpaceStore&)            = delete;
    KVAddressSpaceStore& operator=(const KVAddressSpaceStore&) = delete;
    KVAddressSpaceStore(KVAddressSpaceStore&&)                 = delete;
    KVAddressSpaceStore& operator=(KVAddressSpaceStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(addresses_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_active(std::uint32_t entitlement,
                                                                    std::int32_t execution_row) {
        if (entitlement == 0 || entitlement > page_capacity_) { return std::nullopt; }
        std::optional<KVAddressSpaceHandle> handle = create_inactive();
        if (!handle) { return std::nullopt; }
        try {
            activate(*handle, entitlement, execution_row);
            return handle;
        } catch (...) {
            (void)release(*handle);
            throw;
        }
    }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_inactive() noexcept {
        if (free_count_ == 0) { return std::nullopt; }
        const std::uint32_t index = free_[--free_count_];
        Address& address          = addresses_[index];
        if (address.occupied) {
            free_[free_count_++] = index;
            return std::nullopt;
        }
        address.occupied = true;
        return KVAddressSpaceHandle(this, index, address.generation);
    }

    [[nodiscard]] bool valid(KVAddressSpaceHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < addresses_.size() &&
               addresses_[handle.index_].occupied &&
               addresses_[handle.index_].generation == handle.generation_;
    }

    void activate(KVAddressSpaceHandle handle, std::uint32_t entitlement,
                  std::int32_t execution_row) {
        Address& address = require(handle);
        if (entitlement < address.page_count) {
            throw std::logic_error("KV address space is not activatable");
        }
        auto reservation = prepare_activation(handle, entitlement, execution_row);
        commit_activation(std::move(reservation));
    }

    [[nodiscard]] KVActivationReservation
    prepare_activation(KVAddressSpaceHandle handle, std::uint32_t entitlement,
                       std::int32_t execution_row,
                       std::optional<std::uint32_t> activation_frontier = std::nullopt) {
        Address& address = require(handle);
        if (address.active || address.row || address.reservation.valid() || entitlement == 0 ||
            entitlement > page_capacity_ ||
            (activation_frontier && *activation_frontier > address.committed_frontier)) {
            throw std::logic_error("KV address space is not reservable for activation");
        }
        const std::uint32_t required_pages =
            activation_frontier ? pages_for_tokens(*activation_frontier) : address.page_count;
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        std::uint32_t missing_replicas      = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            if (!pages_->device_resident(membership(address, page))) { ++missing_replicas; }
        }
        const std::uint32_t growth =
            entitlement > required_pages ? entitlement - required_pages : 0U;
        const std::uint64_t required = static_cast<std::uint64_t>(missing_replicas) + growth;
        if (required > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("KV activation reservation overflow");
        }
        pages_->physical_pool().resize_reservation(reservation,
                                                   static_cast<std::uint32_t>(required));
        KVExecutionRowLease row = tables_->acquire(execution_row);
        return KVActivationReservation(*this, handle, entitlement, activation_frontier,
                                       std::move(reservation), std::move(row));
    }

    [[nodiscard]] std::vector<LogicalKVPageHandle>
    missing_device_pages(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        std::vector<LogicalKVPageHandle> missing;
        missing.reserve(address.page_count);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->device_resident(logical)) { missing.push_back(logical); }
        }
        return missing;
    }

    [[nodiscard]] DeviceKVPageReservation& page_reservation(KVActivationReservation& activation) {
        if (activation.owner_ != this || !valid(activation.address_)) {
            throw std::logic_error("KV activation reservation is stale");
        }
        return activation.page_reservation_;
    }

    void commit_activation(KVActivationReservation&& activation, cudaStream_t stream = nullptr) {
        if (activation.owner_ != this) {
            throw std::logic_error("KV activation reservation belongs to another store");
        }
        if (!activation.row_) {
            throw std::logic_error("KV activation reservation has no execution row");
        }
        if (!valid(activation.address_)) {
            throw std::logic_error("KV activation reservation address is stale");
        }
        Address& address = require(activation.address_);
        if (address.active || address.row || address.reservation.valid() ||
            activation.requested_entitlement_ == 0 ||
            activation.requested_entitlement_ > page_capacity_ ||
            (activation.activation_frontier_ &&
             address.committed_frontier != *activation.activation_frontier_)) {
            throw std::logic_error("KV activation destination changed after reservation");
        }
        const std::uint32_t expected = activation.requested_entitlement_ > address.page_count
                                           ? activation.requested_entitlement_ - address.page_count
                                           : 0U;
        if (!activation.page_reservation_.belongs_to(pages_->physical_pool()) ||
            activation.page_reservation_.pages() != expected) {
            throw std::logic_error("KV activation capacity reservation changed");
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->device_resident(logical) || (pages_->address_references(logical) == 1 &&
                                                      !pages_->can_set_writer(logical, true))) {
                throw std::logic_error("KV activation has an unavailable Device page");
            }
        }
        address.reservation = std::move(activation.page_reservation_);
        address.row.emplace(std::move(*activation.row_));
        activation.row_.reset();
        address.active              = true;
        address.checkpoint_frontier = 0;
        activation.owner_           = nullptr;
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->address_references(logical) == 1) {
                pages_->clear_protection(logical);
                pages_->set_writer(logical, true);
            }
            pages_->retain_active_reference(logical);
        }
        publish_membership(address, stream);
    }

    void deactivate(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0 &&
                !pages_->can_set_writer(logical, false)) {
                throw std::logic_error("KV deactivation has an invalid writer reference");
            }
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0) { pages_->set_writer(logical, false); }
            pages_->release_active_reference(logical);
        }
        address.row.reset();
        address.reservation.release();
        address.active = false;
    }

    [[nodiscard]] KVPrefixForkReservation
    prepare_prefix_fork(KVAddressSpaceHandle source_handle, KVAddressSpaceHandle destination_handle,
                        std::uint32_t frontier, std::uint32_t entitlement,
                        std::int32_t execution_row, bool staged_tail_release = false) {
        Address& source      = require(source_handle);
        Address& destination = require(destination_handle);
        if (source.active || source.row || source.reservation.valid()) {
            throw std::logic_error("KV retained-prefix source is still active");
        }
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0) {
            throw std::logic_error("KV retained-prefix destination is not empty");
        }
        if (frontier == 0 || frontier > source.committed_frontier) {
            throw std::logic_error("KV retained-prefix frontier is unavailable");
        }
        const std::uint32_t required_pages = pages_for_tokens(frontier);
        if (entitlement < required_pages || entitlement > page_capacity_) {
            throw std::invalid_argument("KV retained-prefix entitlement is invalid");
        }
        const std::uint32_t page_size    = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t full_pages   = frontier / page_size;
        const std::uint32_t tail_columns = frontier % page_size;
        if (staged_tail_release && tail_columns == 0) {
            throw std::invalid_argument("staged KV prefix fork has no partial tail");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const std::uint32_t required      = page < full_pages ? page_size : tail_columns;
            if (!pages_->device_resident(logical) ||
                pages_->committed_columns(logical) < required || !pages_->can_pin_source(logical) ||
                (page < full_pages && !pages_->can_retain_reference(logical, false))) {
                throw std::logic_error("KV retained-prefix source is not stable");
            }
        }

        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        pages_->physical_pool().resize_reservation(
            reservation, staged_tail_release ? 1U : entitlement - full_pages);
        KVExecutionRowLease row = tables_->acquire(execution_row);
        std::optional<LogicalKVPageHandle> tail_destination;
        if (tail_columns != 0) {
            tail_destination = pages_->materialize_transfer_destination(reservation, tail_columns);
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->pin_source(membership(source, page));
        }
        return KVPrefixForkReservation(*this, source_handle, destination_handle, frontier,
                                       full_pages, tail_columns, entitlement,
                                       std::move(reservation), std::move(row), tail_destination,
                                       staged_tail_release);
    }

    [[nodiscard]] DeviceKVPageHandle
    prefix_fork_tail_source(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (fork.tail_columns_ == 0) {
            throw std::logic_error("page-aligned KV prefix fork has no tail source");
        }
        const Address& source = require(fork.source_);
        return pages_->physical(membership(source, fork.full_pages_));
    }

    [[nodiscard]] DeviceKVPageHandle
    prefix_fork_tail_destination(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (!fork.tail_destination_) {
            throw std::logic_error("page-aligned KV prefix fork has no tail destination");
        }
        return pages_->physical(*fork.tail_destination_);
    }

    [[nodiscard]] LogicalKVPageHandle
    prefix_fork_tail_logical_source(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (fork.tail_columns_ == 0) {
            throw std::logic_error("page-aligned KV prefix fork has no logical tail source");
        }
        const Address& source = require(fork.source_);
        return membership(source, fork.full_pages_);
    }

    void settle_prefix_fork_tail_source(KVPrefixForkReservation& fork) {
        require_prefix_fork(fork);
        if (!fork.staged_tail_release_ || fork.tail_source_settled_ || fork.tail_columns_ == 0) {
            throw std::logic_error("KV prefix-fork tail is not stage-releasable");
        }
        Address& source                   = require(fork.source_);
        const LogicalKVPageHandle logical = membership(source, fork.full_pages_);
        if (!pages_->valid(logical) || pages_->source_pins(logical) == 0 ||
            !pages_->device_resident(logical)) {
            throw std::logic_error("KV prefix-fork tail source changed before settlement");
        }
        pages_->unpin_source(logical);
        fork.tail_source_settled_ = true;
    }

    void complete_prefix_fork_after_tail_release(KVPrefixForkReservation& fork) {
        require_prefix_fork(fork);
        if (!fork.staged_tail_release_ || !fork.tail_source_settled_ || fork.tail_columns_ == 0) {
            throw std::logic_error("KV prefix-fork staged release is incomplete");
        }
        const Address& source             = require(fork.source_);
        const LogicalKVPageHandle logical = membership(source, fork.full_pages_);
        if (!pages_->valid(logical) || pages_->device_resident(logical) ||
            !pages_->host_resident(logical)) {
            throw std::logic_error("KV prefix-fork retained tail was not moved to Host");
        }
        const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
        const std::uint32_t growth         = fork.requested_entitlement_ - required_pages;
        pages_->physical_pool().resize_reservation(fork.page_reservation_, growth);
    }

    void commit_prefix_fork(KVPrefixForkReservation&& fork, cudaStream_t stream = nullptr) {
        require_prefix_fork(fork);
        Address& source                    = require(fork.source_);
        Address& destination               = require(fork.destination_);
        const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
        const std::uint32_t growth         = fork.requested_entitlement_ - required_pages;
        if (!fork.row_ || destination.active || destination.row ||
            destination.reservation.valid() || destination.page_count != 0 ||
            fork.page_reservation_.pages() != growth) {
            throw std::logic_error("KV prefix-fork destination changed before publication");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const bool settled_tail = page == fork.full_pages_ && fork.tail_source_settled_;
            if ((!settled_tail &&
                 (pages_->source_pins(logical) == 0 || !pages_->device_resident(logical))) ||
                (settled_tail && (!fork.staged_tail_release_ || pages_->device_resident(logical) ||
                                  !pages_->host_resident(logical))) ||
                (page < fork.full_pages_ && !pages_->can_retain_reference(logical, false))) {
                throw std::logic_error("KV prefix-fork source changed before publication");
            }
        }
        if (fork.tail_columns_ != 0 &&
            (!fork.tail_destination_ || !pages_->valid(*fork.tail_destination_))) {
            throw std::logic_error("KV prefix-fork tail destination is stale");
        }

        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < fork.full_pages_; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(source, page)));
        }
        if (fork.tail_destination_) {
            publish_scratch_.push_back(pages_->physical(*fork.tail_destination_));
        }
        tables_->publish(fork.row_->handle(), 0, publish_scratch_, stream);

        for (std::uint32_t page = 0; page < fork.full_pages_; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            pages_->retain_reference(logical, false);
            pages_->protect_coverage(logical, static_cast<std::uint32_t>(kPagedKVPageSize));
            membership(destination, page) = logical;
        }
        if (fork.tail_destination_) {
            pages_->publish_transfer_destination(*fork.tail_destination_, true);
            membership(destination, fork.full_pages_) = *fork.tail_destination_;
            fork.tail_destination_.reset();
        }
        destination.page_count         = required_pages;
        destination.committed_frontier = fork.frontier_;
        destination.reservation        = std::move(fork.page_reservation_);
        destination.row.emplace(std::move(*fork.row_));
        fork.row_.reset();
        destination.active = true;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->retain_active_reference(membership(destination, page));
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            if (page == fork.full_pages_ && fork.tail_source_settled_) { continue; }
            pages_->unpin_source(membership(source, page));
        }
        fork.owner_ = nullptr;
    }

    void abort_prefix_fork(KVPrefixForkReservation& fork) noexcept {
        if (fork.owner_ != this) { return; }
        if (valid(fork.source_)) {
            Address& source                    = addresses_[fork.source_.index_];
            const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
            for (std::uint32_t page = 0; page < required_pages; ++page) {
                if (page == fork.full_pages_ && fork.tail_source_settled_) { continue; }
                const LogicalKVPageHandle logical = membership(source, page);
                if (pages_->valid(logical) && pages_->source_pins(logical) != 0) {
                    pages_->unpin_source(logical);
                }
            }
        }
        if (fork.tail_destination_) {
            pages_->abort_transfer_destination(*fork.tail_destination_, fork.page_reservation_);
            fork.tail_destination_.reset();
        }
        fork.owner_ = nullptr;
    }

    [[nodiscard]] KVActiveSnapshotShape active_snapshot_shape(KVAddressSpaceHandle source_handle,
                                                              std::uint32_t frontier) const {
        const Address& source = require_active(source_handle);
        if (frontier == 0 || frontier != source.committed_frontier) {
            throw std::logic_error("active KV snapshot frontier is invalid");
        }
        const std::uint32_t required_pages    = pages_for_tokens(frontier);
        const std::uint32_t entitlement_pages = entitlement(source);
        if (source.page_count < required_pages || entitlement_pages < required_pages) {
            throw std::logic_error("active KV snapshot frontier is not fully materialized");
        }

        const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
        KVActiveSnapshotShape shape{
            .full_pages   = frontier / page_size,
            .tail_columns = frontier % page_size,
        };
        bool mutable_full_seen = false;
        for (std::uint32_t page = 0; page < shape.full_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const std::uint32_t references    = pages_->address_references(logical);
            const std::uint8_t writers        = pages_->writer_references(logical);
            if (!pages_->device_resident(logical) || references == 0 ||
                pages_->committed_columns(logical) < page_size ||
                !pages_->can_retain_reference(logical, false)) {
                throw std::logic_error("active KV snapshot full page is not stable");
            }
            if (writers == 0) {
                if (mutable_full_seen || !pages_->can_pin_source(logical)) {
                    throw std::logic_error("active KV snapshot page ownership is not ordered");
                }
                ++shape.immutable_full_pages;
            } else if (writers == 1 && references == 1 && pages_->can_pin_active_source(logical)) {
                mutable_full_seen = true;
            } else {
                throw std::logic_error("active KV snapshot full page ownership is invalid");
            }
            if (references == 1) { ++shape.unique_full_pages; }
        }

        if (shape.tail_columns != 0) {
            const LogicalKVPageHandle tail = membership(source, shape.full_pages);
            if (!pages_->device_resident(tail) || pages_->writer_references(tail) != 1 ||
                pages_->address_references(tail) != 1 ||
                pages_->committed_columns(tail) < shape.tail_columns ||
                !pages_->can_pin_active_source(tail)) {
                throw std::logic_error("active KV snapshot tail is not a unique writer");
            }
        }
        return shape;
    }

    [[nodiscard]] KVActiveSnapshotReservation
    prepare_active_snapshot(KVAddressSpaceHandle source_handle,
                            KVAddressSpaceHandle destination_handle, std::uint32_t frontier) {
        Address& source      = require_active(source_handle);
        Address& destination = require(destination_handle);
        if (!source.row || !source.reservation.valid() || destination.active || destination.row ||
            destination.reservation.valid() || destination.page_count != 0 ||
            destination.committed_frontier != 0 || frontier == 0 ||
            frontier != source.committed_frontier) {
            throw std::logic_error("active KV snapshot endpoints are invalid");
        }
        const KVActiveSnapshotShape shape     = active_snapshot_shape(source_handle, frontier);
        const std::uint32_t required_pages    = pages_for_tokens(frontier);
        const std::uint32_t entitlement_pages = entitlement(source);
        if (source.page_count != required_pages) {
            throw std::logic_error("active KV snapshot source has untrimmed suffix pages");
        }

        DeviceKVPageReservation tail_reservation = pages_->physical_pool().make_empty_reservation();
        if (shape.copied_pages() != 0) {
            pages_->physical_pool().resize_reservation(tail_reservation, shape.copied_pages());
        }
        std::optional<LogicalKVPageHandle> tail_destination;
        try {
            if (shape.tail_columns != 0) {
                tail_destination =
                    pages_->materialize_transfer_destination(tail_reservation, shape.tail_columns);
            }
            for (std::uint32_t page = 0; page < required_pages; ++page) {
                pages_->pin_source(membership(source, page));
            }
        } catch (...) {
            if (tail_destination) {
                pages_->abort_transfer_destination(*tail_destination, tail_reservation);
            }
            throw;
        }
        return KVActiveSnapshotReservation(*this, source_handle, destination_handle, frontier,
                                           shape, entitlement_pages, std::move(tail_reservation),
                                           tail_destination);
    }

    [[nodiscard]] DeviceKVPageHandle
    active_snapshot_tail_source(const KVActiveSnapshotReservation& snapshot) const {
        require_active_snapshot(snapshot);
        if (snapshot.shape_.tail_columns == 0) {
            throw std::logic_error("page-aligned active KV snapshot has no tail source");
        }
        const Address& source = require(snapshot.source_);
        return pages_->physical(membership(source, snapshot.shape_.full_pages));
    }

    [[nodiscard]] DeviceKVPageHandle
    active_snapshot_tail_destination(const KVActiveSnapshotReservation& snapshot) const {
        require_active_snapshot(snapshot);
        if (!snapshot.tail_destination_) {
            throw std::logic_error("page-aligned active KV snapshot has no tail destination");
        }
        return pages_->physical(*snapshot.tail_destination_);
    }

    void commit_active_snapshot(KVActiveSnapshotReservation&& snapshot,
                                cudaStream_t stream = nullptr) {
        require_active_snapshot(snapshot);
        Address& source      = require_active(snapshot.source_);
        Address& destination = require(snapshot.destination_);
        const KVActiveSnapshotShape shape =
            active_snapshot_shape(snapshot.source_, snapshot.frontier_);
        const std::uint32_t required_pages  = pages_for_tokens(snapshot.frontier_);
        const std::uint32_t expected_growth = snapshot.entitlement_ - required_pages;
        if (shape != snapshot.shape_ || !source.row || !source.reservation.valid() ||
            source.page_count != required_pages ||
            source.committed_frontier != snapshot.frontier_ ||
            source.reservation.pages() != expected_growth || destination.active ||
            destination.row || destination.reservation.valid() || destination.page_count != 0 ||
            destination.committed_frontier != 0 || snapshot.tail_reservation_.pages() != 0) {
            throw std::logic_error("active KV snapshot changed before publication");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            if (pages_->source_pins(logical) == 0 || !pages_->device_resident(logical)) {
                throw std::logic_error("active KV snapshot source changed before publication");
            }
        }
        if (snapshot.shape_.tail_columns != 0 &&
            (!snapshot.tail_destination_ || !pages_->valid(*snapshot.tail_destination_))) {
            throw std::logic_error("active KV snapshot tail destination is stale");
        }

        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < snapshot.shape_.full_pages; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(source, page)));
        }
        if (snapshot.tail_destination_) {
            publish_scratch_.push_back(pages_->physical(*snapshot.tail_destination_));
        }
        tables_->publish(source.row->handle(), 0, publish_scratch_, stream);

        for (std::uint32_t page = 0; page < snapshot.shape_.full_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            if (pages_->writer_references(logical) != 0) { pages_->set_writer(logical, false); }
            pages_->retain_reference(logical, false);
            pages_->protect_coverage(logical, static_cast<std::uint32_t>(kPagedKVPageSize));
            membership(destination, page) = logical;
        }
        if (snapshot.tail_destination_) {
            const LogicalKVPageHandle tail = membership(source, snapshot.shape_.full_pages);
            pages_->set_writer(tail, false);
            pages_->protect_coverage(tail, snapshot.shape_.tail_columns);
            pages_->publish_transfer_destination(*snapshot.tail_destination_, true);
            membership(destination, snapshot.shape_.full_pages) = *snapshot.tail_destination_;
            snapshot.tail_destination_.reset();
        }
        destination.page_count         = required_pages;
        destination.committed_frontier = snapshot.frontier_;
        destination.reservation        = std::move(source.reservation);
        destination.row                = std::move(source.row);
        destination.active             = true;

        if (snapshot.shape_.tail_columns != 0) {
            pages_->release_active_reference(membership(source, snapshot.shape_.full_pages));
            pages_->retain_active_reference(membership(destination, snapshot.shape_.full_pages));
        }

        source.row.reset();
        source.active              = false;
        source.checkpoint_frontier = snapshot.frontier_;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->unpin_source(membership(source, page));
        }
        snapshot.owner_ = nullptr;
    }

    void abort_active_snapshot(KVActiveSnapshotReservation& snapshot) noexcept {
        if (snapshot.owner_ != this) { return; }
        if (valid(snapshot.source_)) {
            Address& source                    = addresses_[snapshot.source_.index_];
            const std::uint32_t required_pages = pages_for_tokens(snapshot.frontier_);
            for (std::uint32_t page = 0; page < required_pages && page < source.page_count;
                 ++page) {
                const LogicalKVPageHandle logical = membership(source, page);
                if (pages_->valid(logical) && pages_->source_pins(logical) != 0) {
                    pages_->unpin_source(logical);
                }
            }
        }
        if (snapshot.tail_destination_) {
            pages_->abort_transfer_destination(*snapshot.tail_destination_,
                                               snapshot.tail_reservation_);
            snapshot.tail_destination_.reset();
        }
        snapshot.owner_ = nullptr;
    }

    void resize_entitlement(KVAddressSpaceHandle handle, std::uint32_t entitlement) {
        Address& address = require_active(handle);
        if (entitlement < address.page_count || entitlement > page_capacity_) {
            throw std::invalid_argument("KV entitlement is smaller than mapped pages");
        }
        pages_->physical_pool().resize_reservation(address.reservation,
                                                   entitlement - address.page_count);
    }

    void release_growth_entitlement(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        pages_->physical_pool().resize_reservation(address.reservation, 0);
    }

    // STORAGE DECOUPLING (2026-09-26): map the pages for a POSITION RANGE without touching anything
    // outside it. ensure_mapped_to_tokens() cannot express this: its target is a contiguous prefix
    // length, so after a window assembly it tries to re-materialise the whole compacted-away history
    // (measured: requested=48 while 32 were free). The write path is already position-keyed
    // (ops/kv_cache/append/kernel.cuh: physical_page = block_table[position >> 6]), so all that is
    // needed is to give the upcoming positions a page and publish it AT ITS OWN OFFSET -- the table
    // supports that (KVExecutionTablePool::publish(row, logical_begin, pages, stream)).
    //
    // Block index == slot index == position / page_size by construction, for both the pre-compaction
    // prefix and the appended positions after it, so appending members and publishing them at their
    // own block offset is exactly what the position-keyed write expects.
    void ensure_mapped_positions(KVAddressSpaceHandle handle, std::uint32_t begin_tokens,
                                 std::uint32_t end_tokens, cudaStream_t stream = nullptr) {
        if (end_tokens <= begin_tokens) { return; }
        Address& address             = require_active(handle);
        const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t begin_block = begin_tokens / page_size;
        const std::uint32_t end_block   = 1U + (end_tokens - 1U) / page_size;
        if (end_block > page_capacity_) {
            throw std::invalid_argument("KV position mapping exceeds the address space");
        }
        // KVMem P4 (kv8) fix-2: an uncompacted (prefix-mode) address whose pages came back without
        // the KVMem bookkeeping (evicted idle continuation re-used later): its prefix pages ARE the
        // blocks [0, page_count), so register them and raise the high-water mark instead of
        // re-materialising block 0 over a live page ("materialization exceeds reserved capacity").
        if (!address.compacted && address.mapped_blocks < address.page_count) {
            const std::size_t ai = static_cast<std::size_t>(&address - addresses_.data());
            std::uint32_t registered = 0;
            for (std::uint32_t block = address.mapped_blocks; block < address.page_count; ++block) {
                LogicalKVPageHandle& bp = block_pages_[ai * page_capacity_ + block];
                if (bp.valid() && pages_->valid(bp) && bp == membership(address, block)) { continue; }
                bp = membership(address, block);
                block_k_baked_at_[ai * page_capacity_ + block] = block * page_size;
                ++registered;
            }
            std::fprintf(stderr,
                         "[kvmem-kv8fix] prefix pages without block map: mapped_blocks %u -> %u "
                         "(registered=%u)\n",
                         address.mapped_blocks, address.page_count, registered);
            address.mapped_blocks = address.page_count;
        }
        // KVMem kv8c (2026-09-30): this chunk re-writes blocks [begin_block, mapped_blocks) that are
        // already mapped; the append kernel writes them into membership(address, block) (slot ==
        // block on this path). If such a block lost its registry entry (kv7 freed the look-ahead
        // page; decode-tail pages are never registered) a later assembly throws "selected block N
        // has no page to stage in". Register the page the write really lands in.
        {
            static const bool c8_on = [] {
                const char* v = std::getenv("NINFER_TERNARY_KVMEM_KV8C_REGISTER");
                return v == nullptr || v[0] != '0';
            }();
            const std::size_t c8_ai = static_cast<std::size_t>(&address - addresses_.data());
            const std::uint32_t c8_end =
                std::min(std::min(end_block, address.mapped_blocks), address.page_count);
            if (c8_on && begin_block < c8_end &&
                block_pages_.size() >= (c8_ai + 1U) * page_capacity_ &&
                block_k_baked_at_.size() >= (c8_ai + 1U) * page_capacity_) {
                std::uint32_t c8_reg = 0, c8_mis = 0, c8_first = 0, c8_last = 0;
                for (std::uint32_t block = begin_block; block < c8_end; ++block) {
                    const LogicalKVPageHandle slot_page = membership(address, block);
                    if (!pages_->valid(slot_page)) { continue; }
                    LogicalKVPageHandle& bp = block_pages_[c8_ai * page_capacity_ + block];
                    if (bp.valid() && pages_->valid(bp)) {
                        if (!(bp == slot_page)) { ++c8_mis; }
                        continue;
                    }
                    if (c8_reg == 0) { c8_first = block; }
                    c8_last = block;
                    bp = slot_page;
                    block_k_baked_at_[c8_ai * page_capacity_ + block] = block * page_size;
                    ++c8_reg;
                }
                if (c8_reg != 0 || c8_mis != 0) {
                    std::fprintf(stderr,
                                 "[kvmem-kv8c] re-registered %u written block(s) [%u..%u] "
                                 "mismatch=%u range=[%u,%u) mapped_blocks=%u page_count=%u "
                                 "compacted=%d\n",
                                 c8_reg, c8_first, c8_last, c8_mis, begin_block, c8_end,
                                 address.mapped_blocks, address.page_count,
                                 address.compacted ? 1 : 0);
                }
            }
        }
        if (end_block <= address.mapped_blocks) { return; }
        std::fprintf(stderr,
                     "[kvmem-map] begin=%u end=%u begin_block=%u end_block=%u mapped_blocks=%u "
                     "page_count=%u compacted=%d\n",
                     begin_tokens, end_tokens, begin_block, end_block, address.mapped_blocks,
                     address.page_count, address.compacted ? 1 : 0);
        // Only positions at or beyond the current high-water mark get NEW pages; anything below it is
        // already mapped (either as the compact window or as previously appended positions).
        const std::uint32_t begin = std::max(begin_block, address.mapped_blocks);
        const std::uint32_t count = end_block - begin;
        if (count == 0) { return; }
        // Mirrors ensure_mapped_to_tokens(): `added` is a span VIEW onto this address's membership row
        // starting at `begin`, and DeviceKVPagePool::materialize() fills every element of that span with
        // a fresh lease. Handing it anything else (e.g. a pre-sized vector of default handles, which an
        // earlier revision did) leaves the slots unfilled and every later pages_->physical() then reads
        // a stale handle -- measured as "logical KV page handle is stale" during warmup.
        const std::size_t address_index = static_cast<std::size_t>(&address - addresses_.data());
        // The target slots may still hold handles of pages that were just staged OUT (their rows are
        // above the window, so the assembly's displace loop never touched them). DeviceKVPagePool::
        // materialize() refuses a destination that is already populated, and publishing a stale
        // handle is worse, so clear the range first. Safe because `mapped_blocks` says these positions
        // have no live mapping: everything at or above it was demoted.
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            membership(address, begin + offset) = {};
        }
        std::span<LogicalKVPageHandle> added(
            memberships_.data() + address_index * page_capacity_ + begin, count);
        // The predecessor is only a locality HINT (DeviceKVPagePool::materialize prefers a page
        // adjacent to it), so it must never be a page that is no longer resident: slot begin-1 can
        // belong to a page that was just staged out, and resolving its physical index then throws
        // "logical KV page handle is stale" (measured at from=49 right after a successful stage-out).
        std::optional<LogicalKVPageHandle> predecessor;
        if (begin != 0) {
            const LogicalKVPageHandle previous_page = membership(address, begin - 1U);
            if (pages_->valid(previous_page) && pages_->device_resident(previous_page)) {
                predecessor = previous_page;
            }
        }
        std::fprintf(stderr, "[kvmem-map] materialize from=%u count=%u resv=%u\n", begin, count,
                     address.reservation.pages());
        try {
            pages_->materialize(address.reservation, added, predecessor);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[kvmem-map] MATERIALIZE THROW: %s\n", e.what());
            throw;
        }
        std::fprintf(stderr, "[kvmem-map] materialize ok\n");
        // ★ 块→页注册表登记（2026-09-26 语义档修复）：这些块的 K 烘在自己的绝对位置（块号*页大小），
        //   且这张表**活过装配**（换出只清 membership，不清它）。检索要把"块自己的页"装进窗：
        //   V 在页里、K 原地重烘；拿 membership[窗口槽] 会装成别的块的页（§31 的静默错答案根因）。
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            const std::uint32_t block = begin + offset;
            block_pages_[address_index * page_capacity_ + block]     = added[offset];
            block_k_baked_at_[address_index * page_capacity_ + block] =
                block * static_cast<std::uint32_t>(kPagedKVPageSize);
        }
        try {
            publish_scratch_.clear();
            for (const LogicalKVPageHandle page : added) {
                if (!pages_->valid(page)) {
                    std::fprintf(stderr, "[kvmem-map] EMPTY HANDLE after materialize\n");
                    throw std::logic_error("KV position mapping left an empty handle");
                }
                try {
                    publish_scratch_.push_back(pages_->physical(page));
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[kvmem-map] PHYSICAL THROW: %s\n", e.what());
                    throw;
                }
            }
            std::fprintf(stderr, "[kvmem-map] physical ok n=%zu\n", publish_scratch_.size());
            // Publish AT THE POSITION OFFSET: the append kernel resolves its target page with
            // block_table[position >> 6], so row `begin` must hold the page for position
            // begin * page_size.
            tables_->publish(address.row->handle(), begin, publish_scratch_, stream);
            std::fprintf(stderr, "[kvmem-map] publish ok\n");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[kvmem-map] THROW: %s\n", e.what());
            for (LogicalKVPageHandle& page : added) {
                if (!pages_->valid(page)) { continue; }
                pages_->dematerialize(page, address.reservation);
                page = {};
            }
            throw;
        }
        for (const LogicalKVPageHandle page : added) { pages_->retain_active_reference(page); }
        // Keep BOTH quantities consistent: mapped_blocks is the position-range high-water mark, while
        // page_count (still consulted in prefix mode) must grow too, or the frontier checks reject a
        // frontier that IS covered.
        address.mapped_blocks = std::max(address.mapped_blocks, end_block);
        address.page_count    = std::max(address.page_count, end_block);
    }

    // Coverage is a lower bound. A speculative mapping may already extend beyond this stage's
    // needs; only an explicit truncate releases it, and commit_frontier publishes valid tokens.
    void ensure_mapped_to_tokens(KVAddressSpaceHandle handle, std::uint32_t tokens,
                                 cudaStream_t stream = nullptr) {
        Address& address           = require_active(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        if (target > entitlement(address)) {
            throw std::invalid_argument(
                "KV coverage exceeds active entitlement: tokens=" + std::to_string(tokens) +
                " required_pages=" + std::to_string(target) +
                " mapped_pages=" + std::to_string(address.page_count) +
                " reserved_pages=" + std::to_string(address.reservation.pages()) +
                " entitlement=" + std::to_string(entitlement(address)));
        }
        if (target <= address.page_count) { return; }
        // STORAGE DECOUPLING (2026-09-26): after compaction ddress.page_count is the WINDOW size, so
        // extending coverage to 	arget must NOT mean "append pages (page_count .. target)" -- that
        // re-materialises the compacted-away history (measured: requested=48 while 32 were available).
        // The window already serves positions [0, committed_frontier); only coverage beyond the
        // frontier needs new pages. When the frontier is covered this is a no-op, which is exactly what
        // a bounded pool requires.
        if (address.compacted && target <= pages_for_tokens(address.committed_frontier)) { return; }        const std::uint32_t begin       = address.page_count;
        const std::uint32_t count       = target - begin;
        const std::size_t address_index = static_cast<std::size_t>(&address - addresses_.data());
        // The target slots may still hold handles of pages that were just staged OUT (their rows are
        // above the window, so the assembly's displace loop never touched them). DeviceKVPagePool::
        // materialize() refuses a destination that is already populated, and publishing a stale
        // handle is worse, so clear the range first. Safe because `mapped_blocks` says these positions
        // have no live mapping: everything at or above it was demoted.
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            membership(address, begin + offset) = {};
        }
        std::span<LogicalKVPageHandle> added(
            memberships_.data() + address_index * page_capacity_ + begin, count);
        // The predecessor is only a locality HINT (DeviceKVPagePool::materialize prefers a page
        // adjacent to it), so it must never be a page that is no longer resident: slot begin-1 can
        // belong to a page that was just staged out, and resolving its physical index then throws
        // "logical KV page handle is stale" (measured at from=49 right after a successful stage-out).
        std::optional<LogicalKVPageHandle> predecessor;
        if (begin != 0) {
            const LogicalKVPageHandle previous_page = membership(address, begin - 1U);
            if (pages_->valid(previous_page) && pages_->device_resident(previous_page)) {
                predecessor = previous_page;
            }
        }
        pages_->materialize(address.reservation, added, predecessor);
        try {
            publish_scratch_.clear();
            for (const LogicalKVPageHandle page : added) {
                try {
                    publish_scratch_.push_back(pages_->physical(page));
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[kvmem-map] PHYSICAL THROW: %s\n", e.what());
                    throw;
                }
            }
            tables_->publish(address.row->handle(), begin, publish_scratch_, stream);
            std::fprintf(stderr, "[kvmem-map] publish ok\n");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[kvmem-map] PUBLISH THROW: %s\n", e.what());
            for (LogicalKVPageHandle& page : added) {
                pages_->dematerialize(page, address.reservation);
                page = {};
            }
            throw;
        }
        for (const LogicalKVPageHandle page : added) { pages_->retain_active_reference(page); }
        address.page_count   = target;
        address.mapped_blocks = std::max(address.mapped_blocks, target);
    }

    // KVMem P1b: after a prefill-time compaction new chunk pages sit right after the window, so
    // slot s holds ORIGINAL block s + block_shift. Register them so a later assembly resolves each
    // block to its own page (block_page()).
    // KVMem kv8c-2 (2026-09-30): register blocks that have a page in their own window slot but no
    // registry entry (decode pages are never registered), ONLY if the window is identity-laid-out:
    // every registered block in [0, n) must sit at its own slot, otherwise nothing is changed.
    std::uint32_t kvmem_kv8c_repair_identity_holes(KVAddressSpaceHandle handle, std::uint32_t n_blocks,
                                                   const char* tag) {
        static const bool on = [] {
            const char* v = std::getenv("NINFER_TERNARY_KVMEM_KV8C_REGISTER");
            return v == nullptr || v[0] != '0';
        }();
        if (!on) { return 0; }
        Address& address  = require_active(handle);
        const std::size_t ai = static_cast<std::size_t>(&address - addresses_.data());
        if (block_pages_.size() < (ai + 1U) * page_capacity_ ||
            block_k_baked_at_.size() < (ai + 1U) * page_capacity_) {
            return 0;
        }
        const std::uint32_t n = std::min(std::min(n_blocks, address.page_count), page_capacity_);
        const std::uint32_t none = ~std::uint32_t{0};
        std::uint32_t holes = 0, matched = 0, bad = none, first = none, last = 0;
        for (std::uint32_t block = 0; block < n; ++block) {
            const LogicalKVPageHandle slot_page = membership(address, block);
            const LogicalKVPageHandle bp = block_pages_[ai * page_capacity_ + block];
            if (bp.valid() && pages_->valid(bp)) {
                if (bp == slot_page) { ++matched; continue; }
                bad = block;
                break;
            }
            if (pages_->valid(slot_page)) {
                ++holes;
                if (first == none) { first = block; }
                last = block;
            }
        }
        if (holes == 0) { return 0; }
        if (bad != none) {
            std::fprintf(stderr,
                         "[kvmem-kv8c] %s: %u unregistered block(s) from %u, window not identity at block %u "
                         "-> not repaired\n", tag, holes, first, bad);
            return 0;
        }
        for (std::uint32_t block = first; block <= last; ++block) {
            const LogicalKVPageHandle slot_page = membership(address, block);
            LogicalKVPageHandle& bp = block_pages_[ai * page_capacity_ + block];
            if (bp.valid() && pages_->valid(bp)) { continue; }
            if (!pages_->valid(slot_page)) { continue; }
            bp = slot_page;
            block_k_baked_at_[ai * page_capacity_ + block] =
                block * static_cast<std::uint32_t>(kPagedKVPageSize);
        }
        std::fprintf(stderr,
                     "[kvmem-kv8c] %s: registered %u unregistered block(s) [%u..%u] (identity window, "
                     "matched=%u n=%u page_count=%u)\n",
                     tag, holes, first, last, matched, n, address.page_count);
        return holes;
    }

    void kvmem_register_shifted_blocks(KVAddressSpaceHandle handle, std::uint32_t slot_begin,
                                       std::uint32_t slot_end, std::uint32_t block_shift) {
        Address& address                = require_active(handle);
        const std::size_t address_index = static_cast<std::size_t>(&address - addresses_.data());
        for (std::uint32_t slot = slot_begin; slot < slot_end; ++slot) {
            const std::uint32_t block = slot + block_shift;
            if (block >= page_capacity_) {
                throw std::invalid_argument("KVMem shifted block exceeds the address space");
            }
            block_pages_[address_index * page_capacity_ + block] = membership(address, slot);
            block_k_baked_at_[address_index * page_capacity_ + block] =
                block * static_cast<std::uint32_t>(kPagedKVPageSize);
        }
    }

    // Assemble a COMPACT window from an ordered list of source pages: slot i of the window holds
    // ordered_pages[i]. The address's whole membership is replaced and exactly the written range is
    // republished, so the window is [0, pages) and nothing beyond it belongs to this address.
    //
    // THREE PROPERTIES ARE DELIBERATE -- the stage-0 smoke measured the absence of each and every one
    // of them to be a silent-wrong-answer hazard (see exp/kvmem-stage0/REPORT.md):
    //   1. `0` IS NOT "EMPTY". Pool page 0 is a legitimate page that really does appear in windows, so
    //      validity comes from handle IDENTITY (pages_->valid), never from comparing a value to zero.
    //   2. SLOTS BEYOND THE NEW LENGTH ARE CLEARED. publish() is a pure overwrite, so a partially
    //      reassembled row would otherwise still read as a complete window. Clearing is also what keeps
    //      a later ensure_mapped_to_tokens() legal: it requires its destination slots to be empty.
    //   3. RETAIN BEFORE RELEASE. One page may occupy two slots of the same window (aliasing), which
    //      needs the total reference count raised before the active count is allowed to pass it.
    [[nodiscard]] KVWindowAssembly
    assemble_window(KVAddressSpaceHandle handle, std::span<const LogicalKVPageHandle> ordered_pages,
                    std::uint32_t tail_columns, cudaStream_t stream = nullptr) {
        Address& address          = require_active(handle);
        const std::uint32_t count = static_cast<std::uint32_t>(ordered_pages.size());
        if (count == 0) {
            throw std::invalid_argument("KV window assembly needs at least one page");
        }
        if (count > page_capacity_) {
            throw std::invalid_argument("KV window assembly exceeds the address page capacity");
        }
        if (count > entitlement(address)) {
            throw std::invalid_argument("KV window assembly exceeds the active entitlement");
        }
        const std::uint32_t kvmem_entitlement_before = entitlement(address);
        if (tail_columns == 0 || tail_columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
            throw std::invalid_argument("KV window assembly tail columns are out of range");
        }
        for (std::uint32_t page = 0; page < count; ++page) {
            const LogicalKVPageHandle logical = ordered_pages[page];
            if (!pages_->valid(logical)) {
                throw std::invalid_argument(
                    "KV window assembly got an unset or stale page handle at slot " +
                    std::to_string(page));
            }
            if (!pages_->device_resident(logical)) {
                throw std::logic_error(
                    "KV window assembly needs device-resident pages (stage them in first), slot " +
                    std::to_string(page));
            }
        }

        const std::uint32_t previous = address.page_count;
        // ★ 换序修复（2026-09-24，run6D 实测 refs=2/active=1/writer=1/pins=0 三跑一致）：先撤旧槽、
        // 再占新槽。原顺序（先 retain 后 release）给移动页开了瞬态双活动名额（retain_reference
        // refs→2），而旧槽退役只还 active 不还 ref ⇒ 重排过的页永久停在 refs=2，
        // exclusively_owned_page（refs==1 才算独占）对所有重排过的页全灭——重烘前的独占检查
        // 从未真实通过过（moved>0 首次走到它就 500）。换序后移动页回到原生签名
        // refs=1/active=1/writer=1；真别名页（前缀 fork，refs=2）仍被拒（谓词不动，负控不变）。
        // Retire the old slots and keep their handles. `displaced_pages` is the ONLY record of what
        // the old window held: once membership is rewritten below, an old member is unreachable
        // through this address, and its device replica would stay allocated forever. That gap is
        // exactly the residency a bounded pool must reclaim (last_assembly_dropped_pages()).
        std::vector<LogicalKVPageHandle> displaced_pages(previous);
        for (std::uint32_t page = 0; page < previous; ++page) {
            const LogicalKVPageHandle displaced = membership(address, page);
            displaced_pages[page]                = displaced;
            membership(address, page)            = {};
            if (pages_->valid(displaced)) { pages_->release_active_reference(displaced); }
        }

        // STORAGE DECOUPLING (2026-09-26, class A / narrow form): pick the stage-out candidates HERE,
        // from the OLD member snapshot -- not from `membership`, which by now holds only the new
        // window (walking it made every page look "still in the window" and produced an empty
        // inventory: measured [kvmem-inv] window=24 dropped=0).
        //
        // A candidate must satisfy all three:
        //  * not a member of the NEW window (ordered_pages) -- the window is what the next forward
        //    reads through, so its pages must stay resident;
        //  * `page + 1 < count` -- it lies entirely BELOW the new committed frontier. Only pages in
        //    the changed range receive another coverage commit (commit_frontier commits
        //    min(page_size, frontier - begin) columns to the pages of that range), and a demoted page
        //    comes back with committed_columns == 0, so demoting a frontier page would let a later
        //    commit overwrite it;
        //  * complete coverage -- the page is done growing.
        // The writer flag is then RELEASED: it exists only to license coverage commits
        // (commit_coverage requires writer_references == 1), and this page can receive no further
        // commit. That release is the narrow form of the class-A decoupling. The in-flight tail keeps
        // its flag and therefore stays resident, which the coverage contract requires.
        last_assembly_dropped_.clear();
        last_assembly_releasable_.clear();
        {
            std::size_t valid_old = 0, in_new = 0, below = 0, full = 0, pinned = 0;
            for (std::uint32_t page = 0; page < previous; ++page) {
                const LogicalKVPageHandle d = displaced_pages[page];
                if (!pages_->valid(d) || !pages_->device_resident(d)) { continue; }
                ++valid_old;
                bool iw = false;
                for (std::uint32_t slot = 0; slot < count && !iw; ++slot) {
                    iw = ordered_pages[slot] == d;
                }
                if (iw) { ++in_new; continue; }
                if (page + 1U >= count) { continue; }
                ++below;
                if (pages_->committed_columns(d) < static_cast<std::uint32_t>(kPagedKVPageSize)) {
                    continue;
                }
                ++full;
                if (pages_->can_pin_source(d)) { ++pinned; }
            }
            std::fprintf(stderr,
                         "[kvmem-inv] prev=%u count=%u old_valid=%zu in_new=%zu below=%zu full=%zu "
                         "pin_ok=%zu\n",
                         previous, count, valid_old, in_new, below, full, pinned);
            {
                std::size_t new_in_old = 0;
                std::size_t old_in_new = 0;
                for (std::uint32_t slot = 0; slot < count; ++slot) {
                    for (std::uint32_t page = 0; page < previous; ++page) {
                        if (ordered_pages[slot] == displaced_pages[page]) { ++new_in_old; break; }
                    }
                }
                for (std::uint32_t page = 0; page < previous; ++page) {
                    for (std::uint32_t slot = 0; slot < count; ++slot) {
                        if (displaced_pages[page] == ordered_pages[slot]) { ++old_in_new; break; }
                    }
                }
                std::fprintf(stderr,
                             "[kvmem-inv] new_in_old=%zu old_in_new=%zu (count=%u previous=%u)\n",
                             new_in_old, old_in_new, count, previous);
            }

            for (std::uint32_t page = 0; page < previous; ++page) {
                const LogicalKVPageHandle displaced = displaced_pages[page];
                if (!pages_->valid(displaced)) { continue; }
                if (!pages_->device_resident(displaced)) { continue; }
                bool in_new_window = false;
                for (std::uint32_t slot = 0; slot < count && !in_new_window; ++slot) {
                    in_new_window = ordered_pages[slot] == displaced;
                }
                // NOTE: an earlier revision gated this on `page + 1 < count`, which compares an OLD
                // slot index against the NEW window's block count -- two different quantities. It
                // excluded exactly the pages we need (measured: 25 old members not in the window, yet
                // below=0). "Is this page finished?" is answered by coverage alone: commit_frontier
                // only touches pages inside its changed range and a demoted page comes back with
                // committed_columns == 0, so a FULLY COVERED page outside the window is done.
                if (in_new_window) { continue; }
                if (pages_->committed_columns(displaced) <
                    static_cast<std::uint32_t>(kPagedKVPageSize)) {
                    continue;
                }
                if (pages_->writer_references(displaced) != 0) {
                    pages_->set_writer(displaced, false);
                }
                if (pages_->can_pin_source(displaced)) {
                    last_assembly_dropped_.push_back(displaced);
                }
            }
        }
        for (std::uint32_t page = 0; page < count; ++page) {
            const LogicalKVPageHandle logical = ordered_pages[page];
            if (pages_->active_address_references(logical) >= pages_->address_references(logical)) {
                pages_->retain_reference(logical, false);
            }
            pages_->retain_active_reference(logical);
        }
        // KVMem P1f: keep the list of displaced-but-still-referenced history pages exact.
        {
            const std::size_t retained_index =
                static_cast<std::size_t>(&address - addresses_.data());
            if (kvmem_retained_.size() <= retained_index) {
                kvmem_retained_.resize(addresses_.size());
            }
            std::vector<LogicalKVPageHandle>& retained = kvmem_retained_[retained_index];
            const auto in_window = [&](LogicalKVPageHandle handle) {
                for (std::uint32_t slot = 0; slot < count; ++slot) {
                    if (ordered_pages[slot] == handle) { return true; }
                }
                return false;
            };
            std::erase_if(retained, in_window);
            for (const LogicalKVPageHandle displaced : displaced_pages) {
                if (!pages_->valid(displaced) || in_window(displaced)) { continue; }
                if (std::find(retained.begin(), retained.end(), displaced) != retained.end()) {
                    continue;
                }
                retained.push_back(displaced);
            }
        }
        for (std::uint32_t page = 0; page < count; ++page) {
            membership(address, page) = ordered_pages[page];
        }
        // STORAGE DECOUPLING (2026-09-26): release the active reference of every page that is NOT in
        // the new window. The earlier release loop ran BEFORE the window was known, so a page that
        // survived into the window also got released, while a page squeezed out of it kept its
        // reference. active_references != 0 is precisely what can_drop_device_replica() refuses, so
        // those pages could never be staged out (measured: releasable=16 staged=0 free_after=0).
        // Releasing here instead makes the accounting exact: the window holds exactly `count` active
        // references and everything else is free to demote.
        for (const LogicalKVPageHandle displaced : displaced_pages) {
            if (!pages_->valid(displaced)) { continue; }
            if (pages_->active_address_references(displaced) == 0) { continue; }
            bool still_in_window = false;
            for (std::uint32_t slot = 0; slot < count && !still_in_window; ++slot) {
                still_in_window = membership(address, slot) == displaced;
            }
            if (!still_in_window) { pages_->release_active_reference(displaced); }
        }

        // Coverage only ever GROWS: commit_coverage() refuses to shrink, and a page that is already at
        // full coverage is already at least as covered as a partial tail requires. A read-only page
        // (writer_references != 1) cannot take a coverage commit at all, so it is left as it is.
        const LogicalKVPageHandle tail = ordered_pages[count - 1U];
        if (tail_columns > pages_->committed_columns(tail) &&
            pages_->writer_references(tail) == 1) {
            pages_->commit_coverage(tail, tail_columns);
        }

        address.page_count = count;
        address.compacted = true;
        // KVMem P1f-2: hand the displaced slots back to the growth entitlement.
        if (address.reservation.valid() && entitlement(address) < kvmem_entitlement_before) {
            const std::uint32_t target =
                std::min(kvmem_entitlement_before, page_capacity_) - address.page_count;
            if (pages_->physical_pool().can_resize_reservation(address.reservation, target)) {
                pages_->physical_pool().resize_reservation(address.reservation, target);
            }
        }
        last_window_members_.assign(ordered_pages.begin(), ordered_pages.end());
        std::fprintf(stderr, "[kvmem-inv] window=%zu dropped=%zu releasable=%zu\n",
                     last_window_members_.size(), last_assembly_dropped_.size(),
                     last_assembly_releasable_.size());
        address.committed_frontier =
            (count - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize) + tail_columns;
        // Publishing exactly [0, page_count) is the whole point: a caller that reads the execution row
        // can never see a slot this call did not decide.
        publish_membership(address, stream);

        // STORAGE DECOUPLING (2026-09-26): release the DEVICE residency of every page the new window
        // dropped. Until now assembly only re-pointed membership and references, so pages squeezed out
        // of the window stayed allocated -- fatal with a bounded pool, where the pool then never
        // recovers a free page and the next prefill chunk cannot be mapped ("could not admit the next
        // chunk, free=0").
        //
        // ORDER: this runs AFTER publish_membership, i.e. once the window is complete. A page may
        // legitimately occupy two slots of the window (aliasing), so a handle displaced from one slot
        // can still be live in another; dropping it during the loop above would have removed residency
        // the window still reads.
        //
        // It is a DROP, never a rebuild: can_drop_device_replica() is the whole guard -- it requires a
        // CURRENT Host replica (content_epoch + committed_columns match) plus no writer, no source pin,
        // no active reference and not destination-pinned. A page that fails any check keeps its
        // residency, so this can never discard a copy the sequence still needs.
        for (std::uint32_t page = 0; page < previous; ++page) {
            const LogicalKVPageHandle displaced = displaced_pages[page];
            if (!pages_->valid(displaced)) { continue; }
            bool still_in_window = false;
            for (std::uint32_t slot = 0; slot < count && !still_in_window; ++slot) {
                still_in_window = membership(address, slot) == displaced;
            }
            if (still_in_window) { continue; }
            (void)pages_->drop_device_replica(displaced);
        }
        // KVMem kv7 (kvmem_kv7_free_empty_displaced, P3d-2): a displaced page with NO committed
        // columns is the look-ahead page mapped one past the chunk. It holds no tokens, so keeping it
        // as retained history only pins a pool page forever (writer flag set -> never staged out or
        // dropped). Free it and forget it in the retained list and the block registry.
        {
            const std::size_t kv7_ai = static_cast<std::size_t>(&address - addresses_.data());
            std::uint32_t kv7_freed = 0;
            for (std::uint32_t page = 0; page < previous; ++page) {
                const LogicalKVPageHandle displaced = displaced_pages[page];
                if (!pages_->valid(displaced)) { continue; }
                if (pages_->committed_columns(displaced) != 0) { continue; }
                bool still_in_window = false;
                for (std::uint32_t slot = 0; slot < count && !still_in_window; ++slot) {
                    still_in_window = membership(address, slot) == displaced;
                }
                if (still_in_window) { continue; }
                if (!pages_->can_dematerialize(displaced)) { continue; }
                if (kv7_ai < kvmem_retained_.size()) {
                    std::erase(kvmem_retained_[kv7_ai], displaced);
                }
                if (block_pages_.size() >= (kv7_ai + 1U) * page_capacity_) {
                    for (std::size_t block = 0; block < page_capacity_; ++block) {
                        if (block_pages_[kv7_ai * page_capacity_ + block] == displaced) {
                            block_pages_[kv7_ai * page_capacity_ + block] = {};
                        }
                    }
                }
                pages_->dematerialize(displaced, address.reservation);
                ++kv7_freed;
            }
            if (kv7_freed != 0) {
                std::fprintf(stderr, "[kvmem-kv7] freed %u empty displaced page(s)\n", kv7_freed);
            }
        }
        return KVWindowAssembly{count, address.committed_frontier};
    }

    // STORAGE DECOUPLING (2026-09-26): after a window assembly, the pages that are device-resident
    // but no longer reachable from the window. They are the stage-out candidates: the prefill
    // trigger copies them to the Host tier and then drops their device replica, which is what makes
    // the pool recover a free page and lets a bounded pool survive a long prefill.
    [[nodiscard]] std::span<const LogicalKVPageHandle> last_assembly_dropped_pages() const noexcept {
        return last_assembly_dropped_;
    }
    void clear_last_assembly_dropped_pages() noexcept { last_assembly_dropped_.clear(); }

    // Fully-committed, out-of-window pages that still carried the write-target flag. Returned
    // separately so the caller can drop the flag (the page is done growing) and then stage it out.
    [[nodiscard]] std::span<const LogicalKVPageHandle> last_assembly_releasable_pages() const noexcept {
        return last_assembly_releasable_;
    }
    void clear_last_assembly_releasable_pages() noexcept { last_assembly_releasable_.clear(); }
    // ★ 块→页注册表（2026-09-26 语义档修复）：membership 是"窗口槽→页"（装配时整行重写、压缩后
    //   还换键制），检索要的却是**块自己的页**（V 在页里、K 原地重烘）。这张表按块号登记、活过装配；
    //   k_baked_at = 这块的 K 现在烘在哪一 token 位（首写 = 块号*64，每次重烘 = 新窗位*64）。
    [[nodiscard]] LogicalKVPageHandle block_page(KVAddressSpaceHandle handle,
                                                 std::uint32_t block) const {
        const Address& address = require_active(handle);
        return block_page(address, block);
    }
    [[nodiscard]] std::uint32_t block_k_baked_at(KVAddressSpaceHandle handle,
                                                  std::uint32_t block) const {
        const Address& address = require(handle);
        return block_k_baked_at(address, block);
    }
    void set_block_k_baked_at(KVAddressSpaceHandle handle, std::uint32_t block,
                              std::uint32_t position) {
        Address& address = require_active(handle);
        set_block_k_baked_at(address, block, position);
    }
    // ★ 恢复用预订单（2026-09-26）：reserve_device_replica → materialize_one(reservation) 从预订单
    //   配额取页；默认构造的预订单没挂在池上（resize 直接 bad allocation）。引擎自己的恢复事务用的
    //   就是地址的预订单（program_impl.h:4980-4982 page_reservation）。本 accessor 把它借给 stage-in。
    [[nodiscard]] DeviceKVPageReservation& page_reservation(KVAddressSpaceHandle handle) {
        return require_active(handle).reservation;
    }

    [[nodiscard]] std::span<const LogicalKVPageHandle> last_window_members() const noexcept {
        return last_window_members_;
    }

    // KVMem P3a-b: does window slot `slot` hold logical block `block` (block->page registry)?
    [[nodiscard]] bool kvmem_slot_holds_block(KVAddressSpaceHandle handle, std::uint32_t slot,
                                              std::uint32_t block) const {
        const Address& address = require(handle);
        if (slot >= address.page_count || block >= page_capacity_) { return false; }
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        if (block_pages_.size() < (index + 1U) * page_capacity_) { return false; }
        const LogicalKVPageHandle page = block_page(address, block);
        return pages_->valid(page) && membership(address, slot) == page;
    }
    // KVMem P3a: pages of a (parked) address space -- window members, and retained history.
    [[nodiscard]] std::vector<LogicalKVPageHandle>
    kvmem_member_pages(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        std::vector<LogicalKVPageHandle> out;
        out.reserve(address.page_count);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            out.push_back(membership(address, page));
        }
        return out;
    }
    [[nodiscard]] std::vector<LogicalKVPageHandle>
    kvmem_retained_pages(KVAddressSpaceHandle handle) const {
        (void)require(handle);
        if (handle.index_ < kvmem_retained_.size()) { return kvmem_retained_[handle.index_]; }
        return {};
    }
    // KVMem P3a: move a parked (inactive) compact window into a freshly activated, empty address
    // space. Every window page must already be device resident. The source is released (empty).
    void kvmem_adopt(KVAddressSpaceHandle destination_handle, KVAddressSpaceHandle source_handle,
                     cudaStream_t stream) {
        Address& destination = require_active(destination_handle);
        Address& source      = require(source_handle);
        if (&destination == &source) { throw std::logic_error("KVMem adopt: same address"); }
        if (source.active || source.row || source.reservation.valid()) {
            throw std::logic_error("KVMem adopt: source is still active");
        }
        if (destination.page_count != 0 || destination.committed_frontier != 0) {
            throw std::logic_error("KVMem adopt: destination is not empty");
        }
        // KVMem P3a-f: no entitlement pre-check; the staged-in window was drawn from our reservation.
        const std::uint32_t kvmem_entitlement_before = entitlement(destination);
        for (std::uint32_t page = 0; page < source.page_count; ++page) {
            if (!pages_->device_resident(membership(source, page))) {
                throw std::logic_error("KVMem adopt: window page is not device resident");
            }
        }
        const std::size_t di = static_cast<std::size_t>(&destination - addresses_.data());
        const std::size_t si = static_cast<std::size_t>(&source - addresses_.data());
        const std::uint32_t moved = source.page_count;
        for (std::uint32_t page = 0; page < moved; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            membership(destination, page)      = logical;
            membership(source, page)           = {};
            if (pages_->address_references(logical) == 1) {
                pages_->clear_protection(logical);
                if (pages_->committed_columns(logical) <
                    static_cast<std::uint32_t>(kPagedKVPageSize)) {
                    pages_->set_writer(logical, true);
                }
            }
            pages_->retain_active_reference(logical);
        }
        const std::size_t cap = page_capacity_;
        if (block_pages_.size() >= (std::max(di, si) + 1U) * cap) {
            for (std::size_t block = 0; block < cap; ++block) {
                block_pages_[di * cap + block] = block_pages_[si * cap + block];
                block_pages_[si * cap + block] = {};
            }
        }
        if (block_k_baked_at_.size() >= (std::max(di, si) + 1U) * cap) {
            for (std::size_t block = 0; block < cap; ++block) {
                block_k_baked_at_[di * cap + block] = block_k_baked_at_[si * cap + block];
                block_k_baked_at_[si * cap + block] = std::numeric_limits<std::uint32_t>::max();
            }
        }
        if (kvmem_retained_.size() < addresses_.size()) { kvmem_retained_.resize(addresses_.size()); }
        kvmem_retained_[di] = std::move(kvmem_retained_[si]);
        kvmem_retained_[si].clear();
        destination.page_count          = moved;
        destination.committed_frontier  = source.committed_frontier;
        destination.compacted           = source.compacted;
        destination.mapped_blocks       = source.mapped_blocks;
        destination.checkpoint_frontier = 0;
        source.page_count               = 0;
        source.committed_frontier       = 0;
        source.compacted                = false;
        source.mapped_blocks            = 0;
        source.checkpoint_frontier      = 0;
        last_window_members_.clear();
        for (std::uint32_t page = 0; page < moved; ++page) {
            last_window_members_.push_back(membership(destination, page));
        }
        last_assembly_releasable_.clear();
        last_assembly_dropped_.clear();
        std::fprintf(stderr, "[kvmem-p3a] adopt entitlement %u -> %u (window %u)\n",
                     kvmem_entitlement_before, entitlement(destination), moved);
        publish_membership(destination, stream);
        rebuild_checkpoint_protection();
        if (!release(source_handle)) {
            throw std::logic_error("KVMem adopt: emptied source could not be released");
        }
    }

    // ★ 先 evict 后 stage-in（2026-09-26 语义档修复）：池小窗大时，新窗成员的 stage-in 需要先把
    //   **旧窗里要离开的成员**放掉——它们的 active 引用挂在 membership 行上，不放就永远过不了
    //   can_drop_device_replica()。本函数把该成员从行里摘掉并还 active 引用；后续 assemble_window
    //   的 displace 循环见空槽自动跳过（合法空转）。返回 true 表示确实放掉了一个。
    [[nodiscard]] bool release_stale_member(KVAddressSpaceHandle handle,
                                            LogicalKVPageHandle victim) {
        Address& address = require_active(handle);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (membership(address, page) != victim) { continue; }
            membership(address, page) = {};
            if (pages_->valid(victim)) { pages_->release_active_reference(victim); }
            return true;
        }
        return false;
    }
    // KVMem kv8c-3: switch + "would commit_frontier fail for lack of mapped pages?"
    static bool kvmem_kv8c_commit_on() noexcept {
        static const bool on = [] {
            const char* v = std::getenv("NINFER_TERNARY_KVMEM_KV8C_COMMIT");
            return v == nullptr || v[0] != '0';
        }();
        return on;
    }
    [[nodiscard]] bool kvmem_kv8c_commit_needs_map(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!kvmem_kv8c_commit_on()) { return false; }
        const Address& address = require_active(handle);
        return frontier >= address.committed_frontier &&
               pages_for_tokens(frontier) > mapped_extent(address) &&
               pages_for_tokens(frontier) > address.page_count;
    }

    void commit_frontier(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        // KVMem kv8c-3: pages present but the mapping high-water mark was lowered by a truncate.
        if (kvmem_kv8c_commit_on() && address.compacted && frontier >= address.committed_frontier &&
            pages_for_tokens(frontier) > address.mapped_blocks &&
            pages_for_tokens(frontier) <= address.page_count) {
            const std::uint32_t c8_need = pages_for_tokens(frontier);
            bool c8_all = true;
            for (std::uint32_t b = address.mapped_blocks; b < c8_need; ++b) {
                if (!pages_->valid(membership(address, b))) { c8_all = false; break; }
            }
            if (c8_all) {
                std::fprintf(stderr,
                             "[kvmem-kv8c] commit: mapped_blocks %u -> %u (pages present, frontier=%u "
                             "committed=%u page_count=%u)\n",
                             address.mapped_blocks, c8_need, frontier, address.committed_frontier,
                             address.page_count);
                address.mapped_blocks = c8_need;
            }
        }
        if (frontier < address.committed_frontier ||
            (pages_for_tokens(frontier) > mapped_extent(address))) {
            std::fprintf(stderr,
                         "[kvmem-kv8c] commit_frontier INVALID: frontier=%u committed=%u mapped_blocks=%u "
                         "page_count=%u compacted=%d\n",
                         frontier, address.committed_frontier, address.mapped_blocks,
                         address.page_count, address.compacted ? 1 : 0);
            // STORAGE DECOUPLING (2026-09-26): after compaction the frontier legitimately exceeds the
            // window's page count -- pages beyond it live in the Host tier. Only the monotonic half of
            // the check still applies.
            throw std::invalid_argument("KV committed frontier is invalid");
        }
        if (frontier == address.committed_frontier) { return; }
        const std::uint32_t page_size          = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t first_changed_page = address.committed_frontier / page_size;
        const std::uint32_t final_changed_page = (frontier - 1U) / page_size;
        for (std::uint32_t page = first_changed_page; page <= final_changed_page; ++page) {
            const std::uint32_t begin   = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t columns = std::min(page_size, frontier - begin);
            // A page that was staged out (its window block lives in the Host tier now) must be skipped:
            // it is not device-resident, so it has no coverage to commit, and touching its handle only
            // throws "logical KV page handle is stale" (measured right after a successful stage-out at
            // cursor=3072, where slots [0,24) are the demoted window blocks). Coverage is restored when
            // the page is staged back in.
            const LogicalKVPageHandle target = membership(address, page);
            if (!pages_->valid(target) || !pages_->device_resident(target)) { continue; }
            if (columns > pages_->committed_columns(target)) {
                pages_->commit_coverage(target, columns);
            }
        }
        address.committed_frontier = frontier;
    }

    void destructive_truncate(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier > address.committed_frontier) {
            throw std::invalid_argument("KV destructive truncate extends the frontier");
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        if (frontier == address.committed_frontier && target == address.page_count) { return; }
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            if (!pages_->can_dematerialize(membership(address, page))) {
                throw std::logic_error("KV truncate would partially release a protected page");
            }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            if (!pages_->can_destructive_truncate(membership(address, target - 1U), columns)) {
                throw std::logic_error("KV truncate would overwrite protected coverage");
            }
        }
        while (address.page_count > target) {
            const std::uint32_t index  = --address.page_count;
            LogicalKVPageHandle page   = membership(address, index);
            membership(address, index) = {};
            pages_->release_active_reference(page);
            pages_->dematerialize(page, address.reservation);
        }
        kvmem_kv8_forget_blocks_from(address, target);   // KVMem P4 (kv8) fix
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            pages_->destructive_truncate(membership(address, target - 1U), columns);
        }
        address.committed_frontier = frontier;
    }

    [[nodiscard]] bool can_destructive_truncate_inactive(
        KVAddressSpaceHandle handle, std::uint32_t frontier,
        bool tail_host_replica_will_be_released = false) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid() ||
            frontier > address.committed_frontier) {
            return false;
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->can_release_reference(logical, false)) { return false; }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            const LogicalKVPageHandle tail = membership(address, target - 1U);
            if (columns != pages_->committed_columns(tail) &&
                !(tail_host_replica_will_be_released
                      ? pages_->can_destructive_truncate_inactive_after_host_release(tail, columns)
                      : pages_->can_destructive_truncate_inactive(tail, columns))) {
                return false;
            }
        }
        return true;
    }

    void destructive_truncate_inactive(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!can_destructive_truncate_inactive(handle, frontier)) {
            throw std::logic_error("inactive KV address space is not destructively truncatable");
        }
        Address& address           = require(handle);
        const std::uint32_t target = pages_for_tokens(frontier);
        // Full pages before the selected frontier can remain shared. Only removed suffix
        // references and a changed private partial tail need mutation; protection is rebuilt from
        // the surviving address-space requirements after publication.
        while (address.page_count > target) {
            const std::uint32_t index         = --address.page_count;
            const LogicalKVPageHandle logical = membership(address, index);
            membership(address, index)        = {};
            if (!pages_->release_reference(logical, false)) { std::terminate(); }
        }
        kvmem_kv8_forget_blocks_from(address, target);   // KVMem P4 (kv8) fix
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            const LogicalKVPageHandle tail = membership(address, target - 1U);
            if (columns != pages_->committed_columns(tail)) {
                pages_->destructive_truncate_inactive(tail, columns);
            }
        }
        address.committed_frontier  = frontier;
        address.checkpoint_frontier = 0;
        rebuild_checkpoint_protection();
    }

    void set_checkpoint_requirement(KVAddressSpaceHandle handle, std::uint32_t protected_frontier) {
        Address& address = require(handle);
        if (protected_frontier > address.committed_frontier) {
            throw std::invalid_argument("KV checkpoint requirement exceeds committed frontier");
        }
        address.checkpoint_frontier = protected_frontier;
        rebuild_checkpoint_protection();
    }

    [[nodiscard]] bool can_truncate_inactive_prefix(KVAddressSpaceHandle handle,
                                                    std::uint32_t frontier) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid() ||
            frontier > address.committed_frontier) {
            return false;
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            if (!pages_->can_release_reference(membership(address, page), false)) { return false; }
        }
        return true;
    }

    void truncate_inactive_prefix(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!can_truncate_inactive_prefix(handle, frontier)) {
            throw std::logic_error("inactive KV checkpoint prefix is not truncatable");
        }
        Address& address           = require(handle);
        const std::uint32_t target = pages_for_tokens(frontier);
        while (address.page_count > target) {
            const std::uint32_t index         = --address.page_count;
            const LogicalKVPageHandle logical = membership(address, index);
            membership(address, index)        = {};
            if (!pages_->release_reference(logical, false)) { std::terminate(); }
        }
        address.committed_frontier  = frontier;
        address.checkpoint_frontier = std::min(address.checkpoint_frontier, frontier);
        rebuild_checkpoint_protection();
    }

    [[nodiscard]] std::uint32_t mapped_pages(KVAddressSpaceHandle handle) const {
        return require(handle).page_count;
    }

    [[nodiscard]] std::uint32_t entitlement(KVAddressSpaceHandle handle) const {
        return entitlement(require(handle));
    }

    [[nodiscard]] std::uint32_t committed_frontier(KVAddressSpaceHandle handle) const {
        return require(handle).committed_frontier;
    }

    // 装配臂（exp\mem-arm）用：checkpoint 保护要求的读访问器。压缩窗口会让 committed_frontier
    // 变小，而 checkpoint_frontier > committed_frontier 会让 rebuild_checkpoint_protection()
    // 直接 std::terminate()（logical_kv_store.h:1879-1881）⇒ 装配前必须先读它、装配后必须下调。
    [[nodiscard]] std::uint32_t checkpoint_frontier(KVAddressSpaceHandle handle) const {
        return require(handle).checkpoint_frontier;
    }

    // 装配臂用：这个窗口槽的页是否被本地址**独占**。重烘是原地改写 K，共享页（前缀 fork 的别名页、
    // 检查点引用页）会被改坏，而这一点从外面看不出来 ⇒ 用它做前置拒绝。
    [[nodiscard]] bool exclusively_owned_page(KVAddressSpaceHandle handle,
                                              std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        const LogicalKVPageHandle logical = membership(address, logical_page);
        // ★ 2026-09-26 精确化：writer == 1 不是"独占"的必要条件。写者旗只是"还能收 coverage 提交"
        //   的许可；从 host 恢复的**满页**（cols==64）本就 writer==0（换出时窄形态特意释放：它永不再
        //   收提交）。独占的风险来自**引用数**（前缀 fork 的别名页 refs==2、检查点引用），与写者旗
        //   无关 ⇒ 条件放宽为 writer <= 1（writer==2 的真共享照拒）。实测依据：SEM15
        //   `rebake would rewrite … (slot 22 block 313 refs=1 active=1 writer=0 pins=0 cols=64)`。
        return pages_->valid(logical) && pages_->address_references(logical) == 1 &&
               pages_->active_address_references(logical) == 1 &&
               pages_->writer_references(logical) <= 1;
    }

    // 观测点（2026-09-24）：exclusively_owned_page 拒答时把四个计数带进原文——区分"真被别人
    // 共享"与"装配自身的引用计数语义"（run5D 实测：移动页首次走到该检查即全灭，需看数定性）。
    [[nodiscard]] std::string page_ownership_debug(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        const LogicalKVPageHandle logical = membership(address, logical_page);
        if (!pages_->valid(logical)) { return std::string("page invalid"); }
        return std::string("refs=") + std::to_string(pages_->address_references(logical)) +
               " active=" + std::to_string(pages_->active_address_references(logical)) +
               " writer=" + std::to_string(pages_->writer_references(logical)) +
               " pins=" + std::to_string(pages_->source_pins(logical)) +
               " cols=" + std::to_string(pages_->committed_columns(logical));
    }

    [[nodiscard]] std::int32_t bound_row(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle) || !addresses_[handle.index_].row) { return -1; }
        return addresses_[handle.index_].row->row_index();
    }

    [[nodiscard]] bool active(KVAddressSpaceHandle handle) const noexcept {
        return valid(handle) && addresses_[handle.index_].active;
    }

    [[nodiscard]] const KVExecutionRowLease& execution_row(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row) {
            throw std::logic_error("KV address space has no execution row");
        }
        return *address.row;
    }

    [[nodiscard]] DeviceKVPageHandle physical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->physical(membership(address, logical_page));
    }

    [[nodiscard]] std::uint64_t content_epoch(KVAddressSpaceHandle handle,
                                              std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->content_epoch(membership(address, logical_page));
    }

    [[nodiscard]] LogicalKVPageHandle logical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return membership(address, logical_page);
    }

    [[nodiscard]] bool has_active_reference(LogicalKVPageHandle page) const noexcept {
        return pages_->valid(page) && pages_->active_address_references(page) != 0;
    }

    [[nodiscard]] bool can_release(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid()) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference(membership(address, page), false)) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool can_release_after_deactivate(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (!address.active) { return can_release(handle); }
        if (!address.row) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference_after_active_reference(membership(address, page))) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool release_after_deactivate(KVAddressSpaceHandle handle) noexcept {
        if (!can_release_after_deactivate(handle)) { return false; }
        try {
            if (addresses_[handle.index_].active) { deactivate(handle); }
        } catch (...) { std::terminate(); }
        if (!release(handle)) { std::terminate(); }
        return true;
    }

    [[nodiscard]] bool release(KVAddressSpaceHandle handle) noexcept {
        if (!can_release(handle)) { return false; }
        Address& address = addresses_[handle.index_];
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->release_reference(membership(address, page), false)) { std::terminate(); }
            membership(address, page) = {};
        }
        // KVMem P1f: history pages the compact window displaced still hold our reference.
        if (handle.index_ < kvmem_retained_.size()) {
            std::size_t released = 0;
            for (const LogicalKVPageHandle retained : kvmem_retained_[handle.index_]) {
                if (pages_->valid(retained) && pages_->release_reference(retained, false)) {
                    ++released;
                }
            }
            if (!kvmem_retained_[handle.index_].empty()) {
                std::fprintf(stderr, "[kvmem-p1f] released %zu / %zu retained history pages\n",
                             released, kvmem_retained_[handle.index_].size());
            }
            kvmem_retained_[handle.index_].clear();
        }
        const std::uint32_t index      = handle.index_;
        const std::uint32_t generation = next_generation(address.generation);
        address                        = Address{};
        address.generation             = generation;
        free_[free_count_++]           = index;
        rebuild_checkpoint_protection();
        return true;
    }

private:
    struct Address {
        std::uint32_t generation          = 1;
        std::uint32_t page_count          = 0;
        std::uint32_t committed_frontier  = 0;
        // STORAGE DECOUPLING (2026-09-26): true once assemble_window() replaced this
        // address's membership with a compact window. From then on page_count is the
        // number of pages IN THE WINDOW, not the length of a contiguous mapped prefix,
        // so the frontier may legitimately exceed it.
        bool          compacted           = false;
        // Highest logical block index that has a page mapped (1-based count). Distinct from
        // page_count, which after compaction is the WINDOW size rather than a prefix length.
        std::uint32_t mapped_blocks      = 0;
        std::uint32_t checkpoint_frontier = 0;
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        bool occupied = false;
        bool active   = false;
    };

    [[nodiscard]] static std::size_t checked_membership_cells(std::uint32_t addresses,
                                                              std::uint32_t pages) {
        if (pages != 0 && addresses > std::numeric_limits<std::size_t>::max() / pages) {
            throw std::overflow_error("KV ordered-membership capacity overflow");
        }
        return static_cast<std::size_t>(addresses) * pages;
    }

    // STORAGE DECOUPLING (2026-09-26): the number of positions this address has pages for.
    // In prefix mode that is page_count; under position-range mapping (after a window assembly)
    // page_count is the WINDOW size instead, and the mapped extent is tracked separately. Every
    // frontier-coverage check must go through here, otherwise a position-mapped address looks
    // permanently uncovered and commit_frontier() rejects its own frontier.
    [[nodiscard]] static std::uint32_t mapped_extent(const Address& a) noexcept {
        return a.compacted ? a.mapped_blocks : a.page_count;
    }
    [[nodiscard]] static std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
        return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    }

    [[nodiscard]] static std::uint32_t next_generation(std::uint32_t generation) noexcept {
        ++generation;
        return generation == 0 ? 1 : generation;
    }

    void rebuild_checkpoint_protection() noexcept {
        try {
            for (const Address& address : addresses_) {
                if (!address.occupied) { continue; }
                for (std::uint32_t page = 0; page < address.page_count; ++page) {
                    pages_->set_protected_coverage(membership(address, page), 0);
                }
            }
            for (const Address& address : addresses_) {
                if (!address.occupied || address.checkpoint_frontier == 0) { continue; }
                if (address.checkpoint_frontier > address.committed_frontier ||
                    pages_for_tokens(address.checkpoint_frontier) > mapped_extent(address)) {
                    std::terminate();
                }
                for (std::uint32_t page = 0; page < pages_for_tokens(address.checkpoint_frontier);
                     ++page) {
                    const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
                    pages_->protect_coverage(membership(address, page),
                                             std::min(static_cast<std::uint32_t>(kPagedKVPageSize),
                                                      address.checkpoint_frontier - begin));
                }
            }
        } catch (...) { std::terminate(); }
    }

    [[nodiscard]] Address& require(KVAddressSpaceHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    [[nodiscard]] const Address& require(KVAddressSpaceHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    void require_prefix_fork(const KVPrefixForkReservation& fork) const {
        if (fork.owner_ != this || !valid(fork.source_) || !valid(fork.destination_)) {
            throw std::logic_error("KV prefix-fork reservation is stale");
        }
    }

    void require_active_snapshot(const KVActiveSnapshotReservation& snapshot) const {
        if (snapshot.owner_ != this || !valid(snapshot.source_) || !valid(snapshot.destination_)) {
            throw std::logic_error("active KV snapshot reservation is stale");
        }
    }

    [[nodiscard]] Address& require_active(KVAddressSpaceHandle handle) {
        Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] const Address& require_active(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] std::uint32_t entitlement(const Address& address) const noexcept {
        return address.page_count +
               (address.reservation.valid() ? address.reservation.pages() : 0U);
    }

    [[nodiscard]] LogicalKVPageHandle& membership(Address& address, std::uint32_t page) noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return memberships_[index * page_capacity_ + page];
    }

    [[nodiscard]] const LogicalKVPageHandle& membership(const Address& address,
                                                        std::uint32_t page) const noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return memberships_[index * page_capacity_ + page];
    }

    // KVMem P4 (kv8) fix, 2026-09-29: keep block_pages_ / mapped_blocks consistent with a truncate of
    // an uncompacted address (block == slot there). Without it the re-prefilled tail block is never
    // re-mapped: ensure_mapped_positions() sees end_block <= mapped_blocks and returns, the chunk is
    // written through the stale table entry into a RELEASED page, and the boundary PROMOTE throws
    // "logical KV page handle is stale" (measured: identical re-sent 4743-token request, kv7).
    void kvmem_kv8_forget_blocks_from(Address& address, std::uint32_t target) noexcept {
        // Works for both address kinds: a window assembly marks the address compacted even when the
        // window is the identity (W == F), so the test is per block: an entry at/above the new
        // frontier whose page was just released (handle no longer valid) is forgotten. The
        // high-water mark only drops when nothing live is left above the frontier.
        if (address.mapped_blocks <= target) { return; }
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        const std::uint32_t end = std::min(address.mapped_blocks, page_capacity_);
        if (block_pages_.size() < (index + 1U) * page_capacity_) { return; }
        std::uint32_t cleared = 0, live = 0;
        for (std::uint32_t block = target; block < end; ++block) {
            LogicalKVPageHandle& page = block_pages_[index * page_capacity_ + block];
            if (!page.valid()) { continue; }
            if (pages_->valid(page)) { ++live; continue; }
            page = {};
            if (block_k_baked_at_.size() >= (index + 1U) * page_capacity_) {
                block_k_baked_at_[index * page_capacity_ + block] = ~std::uint32_t{0};
            }
            ++cleared;
        }
        const std::uint32_t before = address.mapped_blocks;
        if (live == 0) { address.mapped_blocks = target; }
        std::fprintf(stderr,
                     "[kvmem-kv8fix] truncate: target_block=%u mapped_blocks %u -> %u cleared=%u live=%u "
                     "compacted=%d\n",
                     target, before, address.mapped_blocks, cleared, live, address.compacted ? 1 : 0);
    }

    [[nodiscard]] LogicalKVPageHandle block_page(const Address& address,
                                                 std::uint32_t block) const noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return block_pages_[index * page_capacity_ + block];
    }

    [[nodiscard]] std::uint32_t block_k_baked_at(const Address& address,
                                                  std::uint32_t block) const noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        return block_k_baked_at_[index * page_capacity_ + block];
    }

    void set_block_k_baked_at(const Address& address, std::uint32_t block,
                              std::uint32_t position) noexcept {
        const std::size_t index = static_cast<std::size_t>(&address - addresses_.data());
        block_k_baked_at_[index * page_capacity_ + block] = position;
    }

    void publish_membership(const Address& address, cudaStream_t stream = nullptr) {
        if (!address.row) { throw std::logic_error("KV address space has no execution row"); }
        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(address, page)));
        }
        tables_->publish(address.row->handle(), 0, publish_scratch_, stream);
    }


    LogicalKVPageStore* pages_    = nullptr;
    KVExecutionTablePool* tables_ = nullptr;
    std::uint32_t page_capacity_  = 0;
    std::vector<LogicalKVPageHandle> last_assembly_dropped_;
    // THE single source of truth for "what is in the window right now" (2026-09-26). The inventory
    // criterion, the stage-out exclusion and any later consumer must all read THIS, not a re-derived
    // membership row: the row is rewritten during assembly and the ordered_pages argument is gone by
    // then. Without one authoritative set, a page the window still needs can be staged out
    // (measured: releasable=24 for a 24-block window -> "logical KV page handle is stale").
    std::vector<LogicalKVPageHandle> last_window_members_;
    std::vector<LogicalKVPageHandle> last_assembly_releasable_;
    std::vector<Address> addresses_;
    std::vector<std::uint32_t> free_;
    std::vector<LogicalKVPageHandle> memberships_;
    // ★ 块→页注册表 + K 烘位出处（2026-09-26）：与 memberships_ 同布局（地址×page_capacity），
    //   按**块号**索引、**换出/装配不清**（页带宿主副本存活）。k_baked_at = UINT32_MAX 表示未知 ⇒ 必重烘。
    std::vector<LogicalKVPageHandle> block_pages_;
    // KVMem P1f: per address space, history pages displaced by a compact window that still hold
    // this address space's reference (disjoint from the current members).
    std::vector<std::vector<LogicalKVPageHandle>> kvmem_retained_;
    std::vector<std::uint32_t> block_k_baked_at_;
    std::vector<DeviceKVPageHandle> publish_scratch_;
    std::uint32_t free_count_ = 0;
};

inline KVPrefixForkReservation::~KVPrefixForkReservation() {
    if (owner_ != nullptr) { owner_->abort_prefix_fork(*this); }
}

inline KVActiveSnapshotReservation::~KVActiveSnapshotReservation() {
    if (owner_ != nullptr) { owner_->abort_active_snapshot(*this); }
}

} // namespace ninfer::targets::qwen3_6::detail
