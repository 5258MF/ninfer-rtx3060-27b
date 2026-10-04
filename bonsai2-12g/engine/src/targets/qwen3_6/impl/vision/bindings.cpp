#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "targets/qwen3_6/impl/vision/vision_host.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::Q6G64_F16S,
                               {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
                                        {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
                                        {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", NumericFormat::W8G32_F16S,
                         {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
    };
}

namespace {

template <class WeightFn, class TensorFn>
VisionCommonWeights build_vision_common(WeightFn&& W, TensorFn&& T,
                                              const VisionBackbonePlan& backbone,
                                        const VisionMergerInputPlan& merger_input,
                                        const VisionMergerNormPlan& merger_norm) {
    using artifact::NumericFormat;

    VisionCommonWeights out;
    out.patch_embedding = W(backbone.patch_embedding, NumericFormat::Q6G64_F16S,
        VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim);
    out.patch_embedding_bias =
        T(backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding = T(backbone.position_embedding, NumericFormat::BF16,
        {VisionBackboneConfig::hidden, VisionBackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv                    = W(source.qkv, NumericFormat::Q4G64_F16S, 3 * VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.qkv_bias = T(source.qkv_bias, NumericFormat::BF16, {3 * VisionBackboneConfig::hidden});
        target.output = W(source.output, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.output_bias = T(source.output_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.fc1 = W(source.fc1, NumericFormat::Q4G64_F16S, VisionBackboneConfig::intermediate,
            VisionBackboneConfig::hidden);
        target.fc1_bias =
            T(source.fc1_bias, NumericFormat::BF16,
                                          {VisionBackboneConfig::intermediate});
        target.fc2 = W(source.fc2, NumericFormat::Q5G64_F16S, VisionBackboneConfig::hidden,
            VisionBackboneConfig::intermediate);
        target.fc2_bias = T(source.fc2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight = T(source.norm1_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias = T(source.norm1_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight = T(source.norm2_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias = T(source.norm2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }

    out.merger_fc1 = W(merger_input.fc1, NumericFormat::W8G32_F16S,
        VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden);
    out.merger_fc1_bias =
        T(merger_input.fc1_bias, NumericFormat::BF16,
                                      {VisionBackboneConfig::merger_hidden});
    out.merger_norm_weight = T(merger_norm.weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.merger_norm_bias = T(merger_norm.bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    return out;
}


} // namespace

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm) {
    return build_vision_common(
        [&](artifact::ObjectHandle h, artifact::NumericFormat f, std::int32_t rows,
            std::int32_t cols) { return artifact::materialized_weight(materialized, h, f, rows, cols); },
        [&](artifact::ObjectHandle h, artifact::NumericFormat f,
            std::initializer_list<std::int32_t> shape) {
            return artifact::materialized_tensor(materialized, h, f, shape);
        },
        backbone, merger_input, merger_norm);
}

// ---- 3060 local patch: vision weights in host RAM, streamed per chunk (NINFER_TERNARY_VISION_HOST=1)

VisionHostStore& vision_host_store() {
    static VisionHostStore store;
    return store;
}

bool vision_host_requested() {
    const char* v = std::getenv("NINFER_TERNARY_VISION_HOST");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

std::byte* VisionHostStore::host_ptr(artifact::ObjectHandle handle) {
    if (handle.index >= offset_of.size() || offset_of[handle.index] == npos) {
        throw std::logic_error("vision host store has no payload for object " +
                               std::to_string(handle.index));
    }
    return blob.data() + offset_of[handle.index];
}

void vision_host_collect(const artifact::Binder& binder, const VisionBackbonePlan& backbone,
                         const VisionMergerInputPlan& merger_input,
                         const VisionMergerNormPlan& merger_norm,
                         artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias) {
    std::vector<std::vector<artifact::ObjectHandle>> groups;
    groups.push_back(
        {backbone.patch_embedding, backbone.patch_embedding_bias, backbone.position_embedding});
    for (const VisionLayerPlan& l : backbone.layers) {
        groups.push_back({l.norm1_weight, l.norm1_bias, l.qkv, l.qkv_bias, l.output,
                          l.output_bias, l.norm2_weight, l.norm2_bias, l.fc1, l.fc1_bias, l.fc2,
                          l.fc2_bias});
    }
    groups.push_back({merger_norm.weight, merger_norm.bias, merger_input.fc1, merger_input.fc1_bias});
    groups.push_back({merger_fc2, merger_fc2_bias});
    if (groups.size() != VisionHostStore::chunk_merger_b + 1) {
        throw std::logic_error("vision host chunk layout mismatch");
    }
    constexpr std::size_t kAlign = 256;
    const auto up = [](std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; };
    std::size_t total     = 0;
    std::size_t max_index = 0;
    for (const auto& g : groups) {
        total = up(total);
        for (const auto h : g) {
            total     = up(total) + binder.payload(h).data.size();
            max_index = std::max(max_index, h.index);
        }
    }
    VisionHostStore& s = vision_host_store();
    s.blob.assign(total, std::byte{0});
    s.offset_of.assign(max_index + 1, VisionHostStore::npos);
    s.chunks.clear();
    std::size_t pos = 0;
    for (const auto& g : groups) {
        pos = up(pos);
        VisionHostStore::Chunk chunk{pos, pos};
        for (const auto h : g) {
            pos             = up(pos);
            const auto data = binder.payload(h).data;
            std::memcpy(s.blob.data() + pos, data.data(), data.size());
            s.offset_of[h.index] = pos;
            pos += data.size();
        }
        chunk.end = pos;
        s.chunks.push_back(chunk);
    }
    s.enabled = true;
}

VisionCommonWeights materialize_vision_common_host(const VisionBackbonePlan& backbone,
                                                   const VisionMergerInputPlan& merger_input,
                                                   const VisionMergerNormPlan& merger_norm) {
    VisionHostStore& s = vision_host_store();
    return build_vision_common(
        [&](artifact::ObjectHandle h, artifact::NumericFormat f, std::int32_t rows,
            std::int32_t cols) { return artifact::weight_at(s.host_ptr(h), f, rows, cols); },
        [&](artifact::ObjectHandle h, artifact::NumericFormat f,
            std::initializer_list<std::int32_t> shape) {
            return artifact::tensor_at(s.host_ptr(h), f, shape);
        },
        backbone, merger_input, merger_norm);
}

void vision_host_prepare_device() {
    VisionHostStore& s = vision_host_store();
    if (!s.enabled || s.slot != nullptr) { return; }
    std::size_t largest = 0;
    for (const auto& c : s.chunks) { largest = std::max(largest, c.end - c.begin); }
    const cudaError_t pin = cudaHostRegister(s.blob.data(), s.blob.size(), cudaHostRegisterDefault);
    s.pinned              = pin == cudaSuccess;
    if (!s.pinned) { (void)cudaGetLastError(); }
    const cudaError_t alloc = cudaMalloc(&s.slot, largest);
    if (alloc != cudaSuccess) {
        (void)cudaGetLastError();
        s.slot = nullptr;
        throw std::runtime_error("vision host slot allocation failed (" + std::to_string(largest) +
                                 " bytes): " + cudaGetErrorString(alloc));
    }
    s.slot_bytes = largest;
    std::fprintf(stderr,
                 "[vision-host] vision weights in host RAM: %.1f MiB (%s), device slot %.1f MiB, "
                 "%zu chunks\n",
                 static_cast<double>(s.blob.size()) / 1048576.0, s.pinned ? "pinned" : "pageable",
                 static_cast<double>(largest) / 1048576.0, s.chunks.size());
}

} // namespace ninfer::targets::qwen3_6
