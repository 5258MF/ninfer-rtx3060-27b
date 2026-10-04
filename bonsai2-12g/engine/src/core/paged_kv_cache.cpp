#include "core/paged_kv_cache.h"

#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/host_worker_pool.h"
#include "core/paged_kv_gather.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ninfer {
namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

std::size_t checked_table_bytes(const KVExecutionTableSpec& spec) {
    const std::size_t logical =
        checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    const std::size_t rows = static_cast<std::size_t>(spec.table_rows);
    if (logical > std::numeric_limits<std::size_t>::max() / rows / sizeof(std::int32_t)) {
        throw std::overflow_error("Paged KV execution table size overflow");
    }
    return logical * rows * sizeof(std::int32_t);
}

void validate_geometry(const KVPageGeometry& geometry) {
    if (geometry.page_tokens != static_cast<std::uint32_t>(kPagedKVPageSize)) {
        throw std::invalid_argument("Paged KV device geometry requires 64-token pages");
    }
    if (geometry.planes.empty()) { throw std::invalid_argument("Paged KV geometry has no planes"); }
    for (const KVPlaneGeometry& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Paged KV plane extents must be positive");
        }
    }
}

void increment_generation(std::uint32_t& generation) noexcept {
    ++generation;
    if (generation == 0) { ++generation; }
}

} // namespace

PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .storage       = cache.storage,
    };
}

DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                const DeviceKVPagePoolSpec& spec) {
    const std::int32_t physical_pages =
        checked_i32(spec.page_group_count, "Paged KV physical page count");
    validate_geometry(spec.geometry);

    DeviceKVPagePoolLayout layout;
    layout.spec = spec;
    layout.planes.reserve(spec.geometry.planes.size());
    for (std::size_t index = 0; index < spec.geometry.planes.size(); ++index) {
        const KVPlaneGeometry& plane = spec.geometry.planes[index];
        DeviceKVPlaneLayout planned;
        planned.geometry        = plane;
        const std::string label = "Paged KV plane " + std::to_string(index);
        if (spec.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, plane.head_extent, physical_pages},
                plane.alignment, label);
        } else {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, physical_pages, plane.head_extent},
                plane.alignment, label);
        }
        layout.planes.push_back(planned);
    }
    return layout;
}

KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                const KVExecutionTableSpec& spec) {
    (void)checked_table_bytes(spec);
    KVExecutionTableLayout layout;
    layout.spec         = spec;
    layout.block_tables = builder.add_tensor(
        DType::I32,
        {checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity"),
         spec.table_rows},
        256, "Paged KV execution tables");
    return layout;
}

std::size_t DeviceKVPagePoolLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const DeviceKVPlaneLayout& plane : planes) { total += plane.storage.region.bytes; }
    return total;
}

std::size_t KVExecutionTableLayout::metadata_bytes() const noexcept {
    return block_tables.region.bytes;
}

DeviceKVPageLease::~DeviceKVPageLease() { (void)release(); }

DeviceKVPageLease::DeviceKVPageLease(DeviceKVPageLease&& other) noexcept
    : owner_(other.owner_), index_(other.index_), generation_(other.generation_) {
    other.owner_      = nullptr;
    other.index_      = -1;
    other.generation_ = 0;
}

DeviceKVPageLease& DeviceKVPageLease::operator=(DeviceKVPageLease&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_            = other.owner_;
    index_            = other.index_;
    generation_       = other.generation_;
    other.owner_      = nullptr;
    other.index_      = -1;
    other.generation_ = 0;
    return *this;
}

DeviceKVPageHandle DeviceKVPageLease::handle() const noexcept {
    return valid() ? DeviceKVPageHandle(owner_, index_, generation_) : DeviceKVPageHandle();
}

bool DeviceKVPageLease::belongs_to(const DeviceKVPagePool& pool) const noexcept {
    return owner_ == &pool;
}

bool DeviceKVPageLease::release() noexcept {
    if (!valid()) { return false; }
    DeviceKVPagePool* owner        = owner_;
    const std::int32_t index       = index_;
    const std::uint32_t generation = generation_;
    owner_                         = nullptr;
    index_                         = -1;
    generation_                    = 0;
    if (!owner->valid_handle(DeviceKVPageHandle(owner, index, generation))) { return false; }
    owner->release_page(index, generation);
    return true;
}

DeviceKVPageReservation::~DeviceKVPageReservation() { release(); }

DeviceKVPageReservation::DeviceKVPageReservation(DeviceKVPageReservation&& other) noexcept
    : owner_(other.owner_), pages_(other.pages_) {
    other.owner_ = nullptr;
    other.pages_ = 0;
}

DeviceKVPageReservation&
DeviceKVPageReservation::operator=(DeviceKVPageReservation&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    owner_       = other.owner_;
    pages_       = other.pages_;
    other.owner_ = nullptr;
    other.pages_ = 0;
    return *this;
}

bool DeviceKVPageReservation::belongs_to(const DeviceKVPagePool& pool) const noexcept {
    return owner_ == &pool;
}

void DeviceKVPageReservation::clear() noexcept {
    if (!valid() || pages_ == 0) { return; }
    owner_->release_reservation(pages_);
    pages_ = 0;
}

void DeviceKVPageReservation::release() noexcept {
    if (!valid()) { return; }
    owner_->release_reservation(pages_);
    owner_ = nullptr;
    pages_ = 0;
}

DeviceKVPagePool::DeviceKVPagePool(DeviceSpan backing, const DeviceKVPagePoolLayout& layout)
    : spec_(layout.spec) {
    validate_geometry(spec_.geometry);
    if (layout.planes.size() != spec_.geometry.planes.size() || layout.planes.empty()) {
        throw std::invalid_argument("Paged KV device layout plane inventory is inconsistent");
    }

    const std::int32_t physical_pages =
        checked_i32(spec_.page_group_count, "Paged KV physical page count");
    planes_.reserve(layout.planes.size());
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const DeviceKVPlaneLayout& planned = layout.planes[index];
        const KVPlaneGeometry& expected    = spec_.geometry.planes[index];
        if (planned.geometry != expected) {
            throw std::logic_error("Paged KV device plane layout does not match its geometry");
        }
        Tensor plane = planned.storage.bind(backing);
        if (plane.dtype != expected.dtype || plane.ne[0] != expected.leading_extent ||
            plane.ne[1] != kPagedKVPageSize) {
            throw std::logic_error("Paged KV device plane tensor is inconsistent");
        }
        if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            if (plane.ne[2] != expected.head_extent || plane.ne[3] != physical_pages) {
                throw std::logic_error("Paged KV PageMajor plane shape is inconsistent");
            }
        } else if (plane.ne[2] != physical_pages || plane.ne[3] != expected.head_extent) {
            throw std::logic_error("Paged KV HeadMajor plane shape is inconsistent");
        }
        planes_.push_back(plane);
    }

    free_page_runs_.reserve(spec_.page_group_count);
    page_generations_.assign(spec_.page_group_count, 1);
    page_allocated_.assign(spec_.page_group_count, false);
    validation_marks_.assign(spec_.page_group_count, 0);
    free_page_runs_.push_back(FreePageRun{.begin = 0, .count = spec_.page_group_count});
}

std::uint32_t DeviceKVPagePool::capacity_pages() const noexcept {
    // Address-space capacity (defaults to the residency capacity => unchanged for every existing
    // construction site). See DeviceKVPagePoolSpec::address_page_group_count.
    return spec_.address_page_group_count != 0 ? spec_.address_page_group_count
                                               : spec_.page_group_count;
}

std::uint32_t DeviceKVPagePool::residency_capacity_pages() const noexcept {
    return spec_.page_group_count;
}

std::uint32_t DeviceKVPagePool::allocated_pages() const noexcept { return allocated_pages_; }

std::uint32_t DeviceKVPagePool::reserved_pages() const noexcept { return reserved_pages_; }

std::uint32_t DeviceKVPagePool::available_pages() const noexcept {
    // STORAGE DECOUPLING (2026-09-26): only ACTUAL occupancy consumes the pool. A granted reservation
    // is an address-space entitlement, not a promise that every one of its pages is resident -- the
    // virtual-memory model. Counting it here would let a long sequence's own reservation starve that
    // very sequence, which is the coupling this split exists to remove. The bound is residency.
    const std::uint32_t resident_capacity = residency_capacity_pages();
    return allocated_pages_ >= resident_capacity
               ? 0U
               : static_cast<std::uint32_t>(resident_capacity - allocated_pages_);
}

std::size_t DeviceKVPagePool::plane_count() const noexcept { return planes_.size(); }

const Tensor& DeviceKVPagePool::plane(std::size_t index) const { return planes_.at(index); }

std::uint32_t
DeviceKVPagePool::contiguous_run_count(std::span<const DeviceKVPageHandle> pages) const {
    if (pages.empty()) { return 0; }
    std::uint32_t runs    = 0;
    std::int32_t previous = -2;
    for (const DeviceKVPageHandle page : pages) {
        const std::int32_t current = physical_index(page);
        if (runs == 0 || current != previous + 1) { ++runs; }
        previous = current;
    }
    return runs;
}

std::optional<DeviceKVPageReservation> DeviceKVPagePool::reserve(std::uint32_t pages) noexcept {
    if (pages == 0 || pages > available_pages()) { return std::nullopt; }
    reserved_pages_ += pages;
    return DeviceKVPageReservation(*this, pages);
}

bool DeviceKVPagePool::can_resize_reservation(const DeviceKVPageReservation& reservation,
                                              std::uint32_t new_reserved_pages) const noexcept {
    // STORAGE DECOUPLING (2026-09-26): a reservation is GRANTED, not OCCUPIED. The address space may
    // legitimately exceed the pool (a 295K sequence over a 529-page pool), so this no longer compares
    // the granted total against the pool -- that comparison was the coupling. It only has to keep the
    // bookkeeping self-consistent; the physical limit is enforced where pages are actually taken, in
    // materialize(), which names the residency numbers when it refuses.
    (void)new_reserved_pages;
    return reservation.belongs_to(*this) && reservation.pages_ <= reserved_pages_;
}

void DeviceKVPagePool::resize_reservation(DeviceKVPageReservation& reservation,
                                          std::uint32_t new_reserved_pages) {
    if (!can_resize_reservation(reservation, new_reserved_pages)) { throw std::bad_alloc(); }
    reserved_pages_    = reserved_pages_ - reservation.pages_ + new_reserved_pages;
    reservation.pages_ = new_reserved_pages;
}

void DeviceKVPagePool::materialize(DeviceKVPageReservation& reservation,
                                   std::uint32_t target_page_count,
                                   std::vector<DeviceKVPageLease>& destination,
                                   std::optional<DeviceKVPageHandle> preferred_predecessor) {
    static_assert(std::is_nothrow_move_constructible_v<DeviceKVPageLease>);
    if (!reservation.belongs_to(*this) || target_page_count < destination.size()) {
        throw std::invalid_argument("Paged KV materialization has an invalid owner or extent");
    }
    const std::uint32_t old_count = static_cast<std::uint32_t>(destination.size());
    const std::uint32_t count     = target_page_count - old_count;
    if (count > reservation.pages_ || target_page_count > destination.capacity()) {
        throw std::invalid_argument("Paged KV materialization exceeds reserved capacity");
    }
    for (const DeviceKVPageLease& page : destination) {
        if (!page.belongs_to(*this) || !valid_handle(page.handle())) {
            throw std::invalid_argument("Paged KV materialization destination is stale");
        }
    }
    if (count == 0) { return; }
    // STORAGE DECOUPLING (2026-09-26): this is now the ONE loud device-residency guard. A caller whose
    // reservation was granted against the (larger) address space still cannot materialise beyond the
    // physical pool, so the failure lands here -- naming the residency limit, the outstanding
    // allocations and the request -- rather than silently handing out a dead page.
    if (count > available_pages()) {
        std::fprintf(stderr,
                     "[kvmem-resid] BLOCKED requested=%u available=%u allocated=%u reserved=%u "
                     "residency=%u\n",
                     count, available_pages(), allocated_pages_, reserved_pages_,
                     residency_capacity_pages());        throw std::logic_error(
            "Paged KV device residency is exhausted (an over-committed address space must bound its "
            "own working set): requested=" +
            std::to_string(count) + " available=" + std::to_string(available_pages()) +
            " allocated=" + std::to_string(allocated_pages_) +
            " reserved=" + std::to_string(reserved_pages_) +
            " residency_capacity=" + std::to_string(residency_capacity_pages()));
    }

    std::optional<std::int32_t> preferred;
    if (preferred_predecessor) {
        preferred = physical_index(*preferred_predecessor) + 1;
    } else if (!destination.empty()) {
        preferred = destination.back().index_ + 1;
    }

    std::size_t selected        = free_page_runs_.size();
    std::int32_t selected_begin = 0;
    if (preferred && *preferred >= 0) {
        const auto upper = std::upper_bound(
            free_page_runs_.begin(), free_page_runs_.end(), *preferred,
            [](std::int32_t page, const FreePageRun& run) { return page < run.begin; });
        if (upper != free_page_runs_.begin()) {
            const auto candidate = upper - 1;
            const std::uint64_t run_end =
                static_cast<std::uint64_t>(candidate->begin) + candidate->count;
            const std::uint64_t requested_end = static_cast<std::uint64_t>(*preferred) + count;
            if (*preferred >= candidate->begin && requested_end <= run_end) {
                selected       = static_cast<std::size_t>(candidate - free_page_runs_.begin());
                selected_begin = *preferred;
            }
        }
    }
    if (selected == free_page_runs_.size()) {
        const auto contiguous =
            std::find_if(free_page_runs_.begin(), free_page_runs_.end(),
                         [count](const FreePageRun& run) { return run.count >= count; });
        if (contiguous != free_page_runs_.end()) {
            selected       = static_cast<std::size_t>(contiguous - free_page_runs_.begin());
            selected_begin = contiguous->begin;
        }
    }

    const auto append_page = [&](std::int32_t page) {
        destination.push_back(
            DeviceKVPageLease(*this, page, page_generations_[static_cast<std::size_t>(page)]));
        page_allocated_[static_cast<std::size_t>(page)] = true;
    };
    if (selected != free_page_runs_.size()) {
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            append_page(selected_begin + static_cast<std::int32_t>(offset));
        }
        consume_free_run(selected, selected_begin, count);
    } else {
        std::uint32_t remaining   = count;
        std::size_t consumed_runs = 0;
        for (std::size_t run_index = 0; remaining != 0; ++run_index) {
            FreePageRun& run         = free_page_runs_[run_index];
            const std::uint32_t take = std::min(remaining, run.count);
            for (std::uint32_t offset = 0; offset < take; ++offset) {
                append_page(run.begin + static_cast<std::int32_t>(offset));
            }
            remaining -= take;
            if (take == run.count) {
                ++consumed_runs;
            } else {
                run.begin += static_cast<std::int32_t>(take);
                run.count -= take;
            }
        }
        free_page_runs_.erase(free_page_runs_.begin(),
                              free_page_runs_.begin() + static_cast<std::ptrdiff_t>(consumed_runs));
    }
    allocated_pages_ += count;
    reserved_pages_ -= count;
    reservation.pages_ -= count;
}

DeviceKVPageLease DeviceKVPagePool::materialize_one(DeviceKVPageReservation& reservation) {
    if (!reservation.belongs_to(*this) || reservation.pages_ == 0) {
        throw std::invalid_argument("Paged KV single-page materialization exceeds reservation");
    }
    if (free_page_runs_.empty()) {
        throw std::logic_error("Paged KV reservation invariant was violated");
    }
    FreePageRun& run        = free_page_runs_.front();
    const std::int32_t page = run.begin++;
    if (--run.count == 0) { free_page_runs_.erase(free_page_runs_.begin()); }
    page_allocated_[static_cast<std::size_t>(page)] = true;
    ++allocated_pages_;
    --reserved_pages_;
    --reservation.pages_;
    return DeviceKVPageLease(*this, page, page_generations_[static_cast<std::size_t>(page)]);
}

void DeviceKVPagePool::dematerialize(DeviceKVPageReservation& reservation,
                                     std::uint32_t target_page_count,
                                     std::vector<DeviceKVPageLease>& source) {
    if (!reservation.belongs_to(*this) || target_page_count > source.size()) {
        throw std::invalid_argument("Paged KV dematerialization has an invalid owner or extent");
    }
    for (const DeviceKVPageLease& page : source) {
        if (!page.belongs_to(*this) || !valid_handle(page.handle())) {
            throw std::invalid_argument("Paged KV dematerialization source is stale");
        }
    }
    const std::uint32_t released = static_cast<std::uint32_t>(source.size() - target_page_count);
    if (released == 0) { return; }
    if (released > std::numeric_limits<std::uint32_t>::max() - reservation.pages_) {
        throw std::overflow_error("Paged KV reservation size overflow");
    }

    // Shrinking a vector of nothrow-destructible leases cannot fail. Each destructor first returns
    // its physical page; the capacity is then adopted by the same reservation before this method
    // becomes observable to its single owner.
    source.resize(target_page_count);
    reserved_pages_ += released;
    reservation.pages_ += released;
}

void DeviceKVPagePool::dematerialize_one(DeviceKVPageReservation& reservation,
                                         DeviceKVPageLease&& page) {
    if (!reservation.belongs_to(*this) || !page.belongs_to(*this) || !valid_handle(page.handle()) ||
        reservation.pages_ == std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Paged KV single-page dematerialization is invalid");
    }
    if (!page.release()) {
        throw std::logic_error("Paged KV single-page dematerialization lost its lease");
    }
    ++reserved_pages_;
    ++reservation.pages_;
}

bool DeviceKVPagePool::valid_handle(DeviceKVPageHandle handle) const noexcept {
    if (handle.owner_ != this || handle.index_ < 0 ||
        handle.index_ >= static_cast<std::int32_t>(capacity_pages())) {
        return false;
    }
    const std::size_t index = static_cast<std::size_t>(handle.index_);
    return page_allocated_[index] && page_generations_[index] == handle.generation_;
}

std::int32_t DeviceKVPagePool::physical_index(DeviceKVPageHandle handle) const {
    if (!valid_handle(handle)) { throw std::invalid_argument("Paged KV page handle is stale"); }
    return handle.index_;
}

void DeviceKVPagePool::validate_distinct_pages(std::span<const DeviceKVPageHandle> pages,
                                               const char* duplicate_message) const {
    ++validation_stamp_;
    if (validation_stamp_ == 0) {
        std::fill(validation_marks_.begin(), validation_marks_.end(), 0);
        validation_stamp_ = 1;
    }
    for (const DeviceKVPageHandle page : pages) {
        const std::size_t index = static_cast<std::size_t>(physical_index(page));
        if (validation_marks_[index] == validation_stamp_) {
            throw std::invalid_argument(duplicate_message);
        }
        validation_marks_[index] = validation_stamp_;
    }
}

void DeviceKVPagePool::release_page(std::int32_t index, std::uint32_t generation) noexcept {
    if (index < 0 || index >= static_cast<std::int32_t>(capacity_pages())) { return; }
    const std::size_t position = static_cast<std::size_t>(index);
    if (!page_allocated_[position] || page_generations_[position] != generation) { return; }
    page_allocated_[position] = false;
    increment_generation(page_generations_[position]);
    release_free_page(index);
    --allocated_pages_;
}

void DeviceKVPagePool::consume_free_run(std::size_t run_index, std::int32_t begin,
                                        std::uint32_t count) noexcept {
    if (run_index >= free_page_runs_.size() || count == 0) { std::terminate(); }
    FreePageRun& run                = free_page_runs_[run_index];
    const std::int64_t run_end      = static_cast<std::int64_t>(run.begin) + run.count;
    const std::int64_t consumed_end = static_cast<std::int64_t>(begin) + count;
    if (begin < run.begin || consumed_end > run_end) { std::terminate(); }
    if (begin == run.begin && consumed_end == run_end) {
        free_page_runs_.erase(free_page_runs_.begin() + static_cast<std::ptrdiff_t>(run_index));
        return;
    }
    if (begin == run.begin) {
        run.begin += static_cast<std::int32_t>(count);
        run.count -= count;
        return;
    }
    if (consumed_end == run_end) {
        run.count = static_cast<std::uint32_t>(begin - run.begin);
        return;
    }
    const FreePageRun right{.begin = static_cast<std::int32_t>(consumed_end),
                            .count = static_cast<std::uint32_t>(run_end - consumed_end)};
    run.count = static_cast<std::uint32_t>(begin - run.begin);
    free_page_runs_.insert(free_page_runs_.begin() + static_cast<std::ptrdiff_t>(run_index + 1),
                           right);
}

void DeviceKVPagePool::release_free_page(std::int32_t index) noexcept {
    const auto next = std::lower_bound(
        free_page_runs_.begin(), free_page_runs_.end(), index,
        [](const FreePageRun& run, std::int32_t page) { return run.begin < page; });
    const bool joins_right = next != free_page_runs_.end() && index + 1 == next->begin;
    const bool joins_left =
        next != free_page_runs_.begin() &&
        static_cast<std::int64_t>((next - 1)->begin) + (next - 1)->count == index;
    if (joins_left && joins_right) {
        auto& left = *(next - 1);
        left.count += 1U + next->count;
        free_page_runs_.erase(next);
    } else if (joins_left) {
        ++(next - 1)->count;
    } else if (joins_right) {
        next->begin = index;
        ++next->count;
    } else {
        free_page_runs_.insert(next, FreePageRun{.begin = index, .count = 1});
    }
}

void DeviceKVPagePool::release_reservation(std::uint32_t pages) noexcept {
    if (pages > reserved_pages_) { std::terminate(); }
    reserved_pages_ -= pages;
}

void DeviceKVPagePool::zero_pages(std::span<const DeviceKVPageHandle> pages,
                                  cudaStream_t stream) const {
    validate_distinct_pages(pages, "Paged KV zero destination contains duplicate pages");
    std::size_t begin = 0;
    while (begin < pages.size()) {
        std::size_t end = begin + 1;
        while (end < pages.size() && pages[end].index_ == pages[end - 1].index_ + 1) { ++end; }
        const std::int32_t first = pages[begin].index_;
        const std::int32_t count = static_cast<std::int32_t>(end - begin);
        for (const Tensor& plane : planes_) {
            auto* base = static_cast<unsigned char*>(plane.data);
            if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemsetAsync(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                                           static_cast<std::size_t>(count) * plane.nb[3], stream));
            } else {
                CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::int64_t>(first) * plane.nb[2],
                                             plane.nb[3], 0,
                                             static_cast<std::size_t>(count) * plane.nb[2],
                                             static_cast<std::size_t>(plane.ne[3]), stream));
            }
        }
        begin = end;
    }
}

void DeviceKVPagePool::copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                                 cudaStream_t stream) const {
    const std::int32_t source_index      = physical_index(source);
    const std::int32_t destination_index = physical_index(destination);
    if (source_index == destination_index) { return; }
    for (const Tensor& plane : planes_) {
        auto* base = static_cast<unsigned char*>(plane.data);
        if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            CUDA_CHECK(
                cudaMemcpyAsync(base + static_cast<std::int64_t>(destination_index) * plane.nb[3],
                                base + static_cast<std::int64_t>(source_index) * plane.nb[3],
                                plane.nb[3], cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                base + static_cast<std::int64_t>(destination_index) * plane.nb[2], plane.nb[3],
                base + static_cast<std::int64_t>(source_index) * plane.nb[2], plane.nb[3],
                plane.nb[2], static_cast<std::size_t>(plane.ne[3]), cudaMemcpyDeviceToDevice,
                stream));
        }
    }
}

void DeviceKVPagePool::copy_to_host(std::span<const DeviceKVPageHandle> source,
                                    HostKVAllocationView destination, cudaStream_t stream) const {
    if (!destination.valid() || destination.page_count() != source.size() ||
        destination.layout().geometry != geometry()) {
        throw std::invalid_argument("Paged KV D2H geometry or extent is inconsistent");
    }
    for (DeviceKVPageHandle page : source) { (void)physical_index(page); }

    const HostKVPageLayout& host = destination.layout();
    std::size_t begin            = 0;
    while (begin < source.size()) {
        std::size_t end = begin + 1;
        while (end < source.size() && source[end].index_ == source[end - 1].index_ + 1) { ++end; }
        const std::size_t count  = end - begin;
        const std::int32_t first = source[begin].index_;
        for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
            const Tensor& plane                 = planes_[plane_index];
            const HostKVPlaneLayout& host_plane = host.planes[plane_index];
            auto* host_base = destination.data() + begin * host.page_stride + host_plane.offset;
            const auto* device_base = static_cast<const unsigned char*>(plane.data);
            if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    host_base, host.page_stride,
                    device_base + static_cast<std::int64_t>(first) * plane.nb[3], plane.nb[3],
                    host_plane.page_payload_bytes, count, cudaMemcpyDeviceToHost, stream));
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        host_base + static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                        host.page_stride,
                        device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                            static_cast<std::int64_t>(first) * plane.nb[2],
                        plane.nb[2], host_plane.head_payload_bytes, count, cudaMemcpyDeviceToHost,
                        stream));
                }
            }
        }
        begin = end;
    }
}

void DeviceKVPagePool::copy_from_host(HostKVAllocationConstView source,
                                      std::span<const DeviceKVPageHandle> destination,
                                      cudaStream_t stream) const {
    if (!source.valid() || source.page_count() != destination.size() ||
        source.layout().geometry != geometry()) {
        throw std::invalid_argument("Paged KV H2D geometry or extent is inconsistent");
    }
    validate_distinct_pages(destination, "Paged KV H2D destination contains duplicate pages");

    const HostKVPageLayout& host = source.layout();
    std::size_t begin            = 0;
    while (begin < destination.size()) {
        std::size_t end = begin + 1;
        while (end < destination.size() &&
               destination[end].index_ == destination[end - 1].index_ + 1) {
            ++end;
        }
        const std::size_t count  = end - begin;
        const std::int32_t first = destination[begin].index_;
        for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
            const Tensor& plane                 = planes_[plane_index];
            const HostKVPlaneLayout& host_plane = host.planes[plane_index];
            const auto* host_base = source.data() + begin * host.page_stride + host_plane.offset;
            auto* device_base     = static_cast<unsigned char*>(plane.data);
            if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    device_base + static_cast<std::int64_t>(first) * plane.nb[3], plane.nb[3],
                    host_base, host.page_stride, host_plane.page_payload_bytes, count,
                    cudaMemcpyHostToDevice, stream));
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                            static_cast<std::int64_t>(first) * plane.nb[2],
                        plane.nb[2],
                        host_base + static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                        host.page_stride, host_plane.head_payload_bytes, count,
                        cudaMemcpyHostToDevice, stream));
                }
            }
        }
        begin = end;
    }
}

namespace {

// 3060 batched H2D gather (see paged_kv_gather.cu). Two mapped pinned staging buffers are filled
// alternately by host workers while the kernel drains the other one. Process lifetime: the buffers
// are never freed because static destruction may run after the CUDA context is gone.
struct KvGatherStaging {
    static constexpr std::size_t kBufferBytes = std::size_t{32} << 20;
    std::mutex mutex;
    bool tried = false;
    bool ok    = false;
    int device = -1;
    std::byte* host[2]{};
    std::byte* mapped[2]{};
    cudaEvent_t done[2]{};
    bool pending[2]{};
    std::unique_ptr<HostWorkerPool> workers;
    std::uint32_t worker_count = 0;
};

KvGatherStaging& kv_gather_staging() {
    static auto* staging = new KvGatherStaging();
    return *staging;
}

bool kv_gather_disabled() {
    static const bool disabled = std::getenv("KV9_NOGATHER") != nullptr;
    return disabled;
}

bool kv_gather_init(KvGatherStaging& s) {
    if (s.tried) { return s.ok; }
    s.tried = true;
    if (cudaGetDevice(&s.device) != cudaSuccess) { return false; }
    for (int b = 0; b < 2; ++b) {
        void* host = nullptr;
        if (cudaHostAlloc(&host, KvGatherStaging::kBufferBytes,
                          cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        void* mapped = nullptr;
        if (cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.done[b], cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        s.host[b]   = static_cast<std::byte*>(host);
        s.mapped[b] = static_cast<std::byte*>(mapped);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    // Packing is bound by soft faults on the pageable Host KV arena; 4 -> 8 threads gained ~12%.
    s.worker_count    = std::max(1U, std::min(8U, hw > 2 ? hw / 2 : 1U));
    if (const char* v = std::getenv("KV9_GATHER_THREADS")) {
        s.worker_count = std::max(1U, std::min(16U, static_cast<unsigned>(std::strtoul(v, nullptr, 10))));
    }
    s.workers         = std::make_unique<HostWorkerPool>(s.worker_count, 64);
    s.ok              = true;
    return true;
}

constexpr bool aligned16(std::uintptr_t value) noexcept { return (value & 15U) == 0; }

} // namespace

bool DeviceKVPagePool::gather_from_host(std::span<const std::byte* const> records,
                                        std::span<const DeviceKVPageHandle> destination,
                                        const HostKVPageLayout& layout, cudaStream_t stream) const {
    if (kv_gather_disabled() || destination.empty() || records.size() != destination.size() ||
        layout.geometry != geometry() || layout.planes.size() != planes_.size() || planes_.empty()) {
        return false;
    }
    const bool head_major = geometry().device_plane_order == PagedKVPlaneOrder::HeadMajor;
    const std::size_t stride = layout.page_stride;
    if (!aligned16(stride)) { return false; }
    std::vector<detail::KvGatherPlane> table;
    table.reserve(planes_.size());
    for (std::size_t p = 0; p < planes_.size(); ++p) {
        const Tensor& plane           = planes_[p];
        const HostKVPlaneLayout& host = layout.planes[p];
        const auto heads              = static_cast<std::int64_t>(head_major ? plane.ne[3] : plane.ne[2]);
        if (heads <= 0 || host.head_payload_bytes * static_cast<std::size_t>(heads) != host.page_payload_bytes ||
            !aligned16(reinterpret_cast<std::uintptr_t>(plane.data)) ||
            !aligned16(static_cast<std::uintptr_t>(plane.nb[2])) ||
            !aligned16(static_cast<std::uintptr_t>(plane.nb[3])) || !aligned16(host.offset) ||
            !aligned16(host.head_payload_bytes) || !aligned16(host.page_payload_bytes) ||
            host.offset + host.page_payload_bytes > stride) {
            return false;
        }
        if (!head_major && static_cast<std::size_t>(plane.nb[3]) < host.page_payload_bytes) { return false; }
        if (head_major && static_cast<std::size_t>(plane.nb[2]) < host.head_payload_bytes) { return false; }
        table.push_back(detail::KvGatherPlane{
            .base        = static_cast<unsigned char*>(plane.data),
            .nb2         = plane.nb[2],
            .nb3         = plane.nb[3],
            .host_offset = static_cast<long long>(host.offset),
            .head_bytes  = static_cast<long long>(host.head_payload_bytes),
            .page_bytes  = static_cast<long long>(host.page_payload_bytes),
            .heads       = static_cast<int>(heads),
            .pad         = 0,
        });
    }
    for (const std::byte* record : records) {
        if (record == nullptr || !aligned16(reinterpret_cast<std::uintptr_t>(record))) { return false; }
    }
    const std::size_t table_bytes = (table.size() * sizeof(detail::KvGatherPlane) + 255) & ~std::size_t{255};
    if (table_bytes + 256 + stride > KvGatherStaging::kBufferBytes) { return false; }
    std::size_t per_batch = (KvGatherStaging::kBufferBytes - table_bytes - 256) / (stride + sizeof(int));
    while (per_batch > 0 &&
           table_bytes + ((per_batch * sizeof(int) + 255) & ~std::size_t{255}) + per_batch * stride >
               KvGatherStaging::kBufferBytes) {
        --per_batch;
    }
    if (per_batch == 0) { return false; }

    KvGatherStaging& s = kv_gather_staging();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!kv_gather_init(s)) { return false; }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != s.device) { return false; }

    validate_distinct_pages(destination, "Paged KV H2D destination contains duplicate pages");
    std::vector<int> indices(destination.size());
    for (std::size_t i = 0; i < destination.size(); ++i) { indices[i] = physical_index(destination[i]); }

    static const bool gather_trace = std::getenv("KV9_XFER_TRACE") != nullptr;
    using GatherClock = std::chrono::steady_clock;
    const auto gather_begin = GatherClock::now();
    double pack_ms = 0, wait_ms = 0;
    const std::size_t index_bytes = (per_batch * sizeof(int) + 255) & ~std::size_t{255};
    std::size_t begin = 0;
    int buffer        = 0;
    while (begin < destination.size()) {
        const std::size_t count = std::min(per_batch, destination.size() - begin);
        if (s.pending[buffer]) {
            const auto w0 = GatherClock::now();
            CUDA_CHECK(cudaEventSynchronize(s.done[buffer]));
            wait_ms += std::chrono::duration<double, std::milli>(GatherClock::now() - w0).count();
            s.pending[buffer] = false;
        }
        std::byte* base = s.host[buffer];
        std::memcpy(base, table.data(), table.size() * sizeof(detail::KvGatherPlane));
        std::memcpy(base + table_bytes, indices.data() + begin, count * sizeof(int));
        std::byte* pages = base + table_bytes + index_bytes;
        // Pack whole page records with the host workers (pageable -> pinned).
        const auto p0 = GatherClock::now();
        const std::uint32_t parts =
            static_cast<std::uint32_t>(std::min<std::size_t>(s.worker_count, count));
        std::vector<std::future<void>> futures;
        futures.reserve(parts);
        for (std::uint32_t part = 0; part < parts; ++part) {
            const std::size_t first = count * part / parts;
            const std::size_t last  = count * (part + 1) / parts;
            futures.push_back(s.workers->submit([=, &records] {
                for (std::size_t i = first; i < last; ++i) {
                    std::memcpy(pages + i * stride, records[begin + i], stride);
                }
            }));
        }
        for (auto& future : futures) { future.get(); }
        pack_ms += std::chrono::duration<double, std::milli>(GatherClock::now() - p0).count();
        std::byte* mapped = s.mapped[buffer];
        const cudaError_t launched = detail::launch_kv_page_gather(
            reinterpret_cast<const detail::KvGatherPlane*>(mapped), static_cast<int>(table.size()),
            reinterpret_cast<const int*>(mapped + table_bytes), mapped + table_bytes + index_bytes,
            stride, static_cast<int>(count), head_major, stream);
        if (launched != cudaSuccess) {
            throw std::runtime_error(std::string("Paged KV H2D gather launch failed: ") +
                                     cudaGetErrorString(launched));
        }
        CUDA_CHECK(cudaEventRecord(s.done[buffer], stream));
        s.pending[buffer] = true;
        begin += count;
        buffer ^= 1;
    }
    if (gather_trace) {
        std::fprintf(stderr,
                     "KVGATHER pages=%zu use=1 threads=%u batch=%zu pack_ms=%.3f wait_ms=%.3f "
                     "submit_ms=%.3f\n",
                     destination.size(), s.worker_count, per_batch, pack_ms, wait_ms,
                     std::chrono::duration<double, std::milli>(GatherClock::now() - gather_begin).count());
    }
    return true;
}

std::vector<DeviceKVPageReservation>
reserve_device_kv_page_bundle(std::span<const DeviceKVPageReservationRequest> requests) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const DeviceKVPageReservationRequest& request = requests[index];
        if (request.pool == nullptr || request.pages == 0) {
            throw std::invalid_argument("Paged KV bundle reservation is empty");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (requests[previous].pool == request.pool) {
                throw std::invalid_argument("Paged KV bundle names the same pool twice");
            }
        }
        if (request.pages > request.pool->available_pages()) { throw std::bad_alloc(); }
    }

    std::vector<DeviceKVPageReservation> reservations;
    reservations.reserve(requests.size());
    for (const DeviceKVPageReservationRequest& request : requests) {
        std::optional<DeviceKVPageReservation> reservation = request.pool->reserve(request.pages);
        if (!reservation) {
            throw std::logic_error("Paged KV bundle prevalidation was not stable");
        }
        reservations.push_back(std::move(*reservation));
    }
    return reservations;
}

KVExecutionRowLease::~KVExecutionRowLease() { (void)release(); }

KVExecutionRowLease::KVExecutionRowLease(KVExecutionRowLease&& other) noexcept
    : owner_(other.owner_), row_(other.row_), generation_(other.generation_) {
    other.owner_      = nullptr;
    other.row_        = -1;
    other.generation_ = 0;
}

KVExecutionRowLease& KVExecutionRowLease::operator=(KVExecutionRowLease&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_            = other.owner_;
    row_              = other.row_;
    generation_       = other.generation_;
    other.owner_      = nullptr;
    other.row_        = -1;
    other.generation_ = 0;
    return *this;
}

KVExecutionRowHandle KVExecutionRowLease::handle() const noexcept {
    return valid() ? KVExecutionRowHandle(owner_, row_, generation_) : KVExecutionRowHandle();
}

bool KVExecutionRowLease::belongs_to(const KVExecutionTablePool& pool) const noexcept {
    return owner_ == &pool;
}

bool KVExecutionRowLease::release() noexcept {
    if (!valid()) { return false; }
    KVExecutionTablePool* owner    = owner_;
    const std::int32_t row         = row_;
    const std::uint32_t generation = generation_;
    owner_                         = nullptr;
    row_                           = -1;
    generation_                    = 0;
    return owner->release_row(row, generation);
}

KVExecutionTablePool::KVExecutionTablePool(DeviceSpan backing, const KVExecutionTableLayout& layout,
                                           const DeviceKVPagePool& pages)
    : spec_(layout.spec), pages_(&pages), block_tables_(layout.block_tables.bind(backing)),
      host_shadow_(checked_table_bytes(layout.spec)),
      row_in_use_(static_cast<std::size_t>(layout.spec.table_rows), false),
      row_generations_(static_cast<std::size_t>(layout.spec.table_rows), 1) {
    if (block_tables_.dtype != DType::I32 ||
        block_tables_.ne[0] !=
            checked_i32(spec_.logical_page_capacity, "Paged KV logical page capacity") ||
        block_tables_.ne[1] != spec_.table_rows) {
        throw std::logic_error("Paged KV execution-table layout is inconsistent");
    }
}

std::uint32_t KVExecutionTablePool::logical_page_capacity() const noexcept {
    return spec_.logical_page_capacity;
}

std::int32_t KVExecutionTablePool::row_count() const noexcept { return spec_.table_rows; }

KVExecutionRowLease KVExecutionTablePool::acquire(std::int32_t row_index) {
    if (row_index < 0 || row_index >= row_count()) {
        throw std::out_of_range("Paged KV execution row is out of range");
    }
    const std::size_t index = static_cast<std::size_t>(row_index);
    if (row_in_use_[index]) { throw std::logic_error("Paged KV execution row is already bound"); }
    row_in_use_[index] = true;
    return KVExecutionRowLease(*this, row_index, row_generations_[index]);
}

bool KVExecutionTablePool::valid_handle(KVExecutionRowHandle handle) const noexcept {
    if (handle.owner_ != this || handle.row_ < 0 || handle.row_ >= row_count()) { return false; }
    const std::size_t row_index = static_cast<std::size_t>(handle.row_);
    return row_in_use_[row_index] && row_generations_[row_index] == handle.generation_;
}

bool KVExecutionTablePool::release_row(std::int32_t row_index, std::uint32_t generation) noexcept {
    if (row_index < 0 || row_index >= row_count()) { return false; }
    const std::size_t index = static_cast<std::size_t>(row_index);
    if (!row_in_use_[index] || row_generations_[index] != generation) { return false; }
    row_in_use_[index] = false;
    increment_generation(row_generations_[index]);
    return true;
}

void KVExecutionTablePool::publish(KVExecutionRowHandle row_handle, std::uint32_t logical_begin,
                                   std::span<const DeviceKVPageHandle> page_handles,
                                   cudaStream_t stream) {
    if (!valid_handle(row_handle) || logical_begin > logical_page_capacity() ||
        page_handles.size() > logical_page_capacity() - logical_begin) {
        throw std::invalid_argument("Paged KV mapping publication is outside its execution row");
    }
    auto* shadow = static_cast<std::int32_t*>(host_shadow_.data()) +
                   static_cast<std::size_t>(row_handle.row_) * logical_page_capacity() +
                   logical_begin;
    for (std::size_t index = 0; index < page_handles.size(); ++index) {
        shadow[index] = pages_->physical_index(page_handles[index]);
    }
    publish_indices(row_handle, logical_begin,
                    std::span<const std::int32_t>(shadow, page_handles.size()), stream);
}

void KVExecutionTablePool::publish(KVExecutionRowHandle row_handle, std::uint32_t logical_begin,
                                   std::span<const DeviceKVPageLease> page_leases,
                                   cudaStream_t stream) {
    if (!valid_handle(row_handle) || logical_begin > logical_page_capacity() ||
        page_leases.size() > logical_page_capacity() - logical_begin) {
        throw std::invalid_argument("Paged KV mapping publication is outside its execution row");
    }
    auto* shadow = static_cast<std::int32_t*>(host_shadow_.data()) +
                   static_cast<std::size_t>(row_handle.row_) * logical_page_capacity() +
                   logical_begin;
    for (std::size_t index = 0; index < page_leases.size(); ++index) {
        if (!page_leases[index].belongs_to(*pages_)) {
            throw std::invalid_argument("Paged KV execution mapping names another page pool");
        }
        shadow[index] = pages_->physical_index(page_leases[index].handle());
    }
    publish_indices(row_handle, logical_begin,
                    std::span<const std::int32_t>(shadow, page_leases.size()), stream);
}

void KVExecutionTablePool::publish_repeated(KVExecutionRowHandle row_handle,
                                            DeviceKVPageHandle page, std::uint32_t count,
                                            cudaStream_t stream) {
    if (!valid_handle(row_handle) || count > logical_page_capacity()) {
        throw std::invalid_argument("Repeated Paged KV mapping is outside its execution row");
    }
    const std::int32_t physical = pages_->physical_index(page);
    auto* shadow                = static_cast<std::int32_t*>(host_shadow_.data()) +
                   static_cast<std::size_t>(row_handle.row_) * logical_page_capacity();
    std::fill_n(shadow, count, physical);
    publish_indices(row_handle, 0, std::span<const std::int32_t>(shadow, count), stream);
}

void KVExecutionTablePool::publish_indices(KVExecutionRowHandle row_handle,
                                           std::uint32_t logical_begin,
                                           std::span<const std::int32_t> indices,
                                           cudaStream_t stream) {
    if (indices.empty()) { return; }
    Tensor destination_row = row(row_handle);
    auto* destination      = static_cast<std::int32_t*>(destination_row.data) + logical_begin;
    CUDA_CHECK(cudaMemcpyAsync(destination, indices.data(), indices.size_bytes(),
                               cudaMemcpyHostToDevice, stream));
}

Tensor KVExecutionTablePool::row(KVExecutionRowHandle handle) const {
    if (!valid_handle(handle)) { throw std::invalid_argument("Paged KV execution row is stale"); }
    return block_tables_.slice(1, handle.row_, 1)
        .view({static_cast<std::int32_t>(logical_page_capacity())});
}

} // namespace ninfer
