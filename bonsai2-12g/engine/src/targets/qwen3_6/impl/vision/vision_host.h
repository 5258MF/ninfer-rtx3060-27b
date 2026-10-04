#pragma once

// 3060 local patch (2026-09-30): NINFER_TERNARY_VISION_HOST=1 keeps the vision tower weights
// (~280 MiB) in host RAM instead of the device weight arena. During VisionContext::encode each
// chunk (patch stage, one ViT layer, merger A, merger B) is copied into one small device slot
// on the encode stream right before it is used, so stream order guarantees the previous chunk's
// kernels finished before the slot is overwritten. Default off: nothing changes when unset.

#include "artifact/binder.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/vision.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct VisionHostStore {
    struct Chunk {
        std::size_t begin = 0;
        std::size_t end   = 0;
    };
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
    // Chunk order: 0 = patch stage, 1..layers = ViT layers, then merger A, merger B.
    static constexpr std::size_t chunk_pre = 0;
    static constexpr std::size_t chunk_layer0 = 1;
    static constexpr std::size_t chunk_merger_a = 1 + VisionBackboneConfig::layers;
    static constexpr std::size_t chunk_merger_b = 2 + VisionBackboneConfig::layers;

    bool enabled = false;
    std::vector<std::byte> blob;
    std::vector<std::size_t> offset_of; // ObjectHandle::index -> blob offset (npos if absent)
    std::vector<Chunk> chunks;
    void* slot             = nullptr;
    std::size_t slot_bytes = 0;
    bool pinned            = false;

    [[nodiscard]] std::byte* host_ptr(artifact::ObjectHandle handle);
    [[nodiscard]] bool in_blob(const void* p) const {
        if (p == nullptr || blob.empty()) { return false; }
        const auto* b = reinterpret_cast<const unsigned char*>(blob.data());
        const auto* q = static_cast<const unsigned char*>(p);
        return q >= b && q < b + blob.size();
    }
};

[[nodiscard]] VisionHostStore& vision_host_store();
[[nodiscard]] bool vision_host_requested();

// Copies every vision payload out of the artifact mapping into the host blob, grouped by chunk.
void vision_host_collect(const artifact::Binder& binder, const VisionBackbonePlan& backbone,
                         const VisionMergerInputPlan& merger_input,
                         const VisionMergerNormPlan& merger_norm,
                         artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias);

// Weight/Tensor views whose pointers name the host blob (never dereferenced on the device).
[[nodiscard]] VisionCommonWeights materialize_vision_common_host(
    const VisionBackbonePlan& backbone, const VisionMergerInputPlan& merger_input,
    const VisionMergerNormPlan& merger_norm);

// Pins the blob (best effort) and allocates the device slot. Call at load, before KV pools.
void vision_host_prepare_device();

inline Weight vision_host_reloc(const VisionHostStore& s, const Weight& w, std::ptrdiff_t delta) {
    Weight out = w;
    const auto move = [&](const void*& p) {
        if (s.in_blob(p)) { p = static_cast<const unsigned char*>(p) + delta; }
    };
    move(out.payload);
    move(out.qdata);
    move(out.qhigh);
    move(out.scales);
    return out;
}

inline Tensor vision_host_reloc(const VisionHostStore& s, const Tensor& t, std::ptrdiff_t delta) {
    Tensor out = t;
    if (s.in_blob(out.data)) { out.data = static_cast<unsigned char*>(out.data) + delta; }
    return out;
}

} // namespace ninfer::targets::qwen3_6
