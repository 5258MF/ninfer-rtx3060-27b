#include "ops/linear/ternary/ternary_rotation.h"

#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kActivationBytesPerElement = 2; // BF16

bool ternary_qtype(QType qtype) noexcept {
    return qtype == QType::PTQ1_0_G128 || qtype == QType::PQ2_0_G128;
}

} // namespace

bool ternary_weight_is_folded(const Weight& weight) noexcept {
    return ternary_qtype(weight.qtype) && weight.hadamard_signs != nullptr &&
           weight.hadamard_n_blk > 0;
}

bool ternary_rotation_enabled() {
    // Read once: the value must not change between graph construction and graph replay, because
    // the rotation decides whether a kernel appears in the captured graph at all.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_HADAMARD");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

bool ternary_gdn_perm_enabled() {
    // Default OFF: this port's packer already normalized every GDN tensor to the grouped order,
    // so the reference's feature permutation would be a second permutation. See the header.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_GDN_PERM");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

std::size_t ternary_rotation_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    if (k <= 0 || tokens <= 0) { return 0; }
    // The rotation buffer plus the int8 rung's activation-quantization scratch (one int8 code row
    // per token, one fp32 scale per token, and a little slack for the 256-byte arena alignment).
    // They are declared together because this function IS the capacity statement for the ternary
    // linear op (see linear.cpp), so anything the op allocates from the arena must be counted here.
    // The quantization pass itself only runs for T >= kTernaryS8MinTokens, but the capacity is the
    // max over token counts the plan allows, which is what the planner asks for.
    const std::size_t rotated = static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) *
                                kActivationBytesPerElement;
    const std::size_t s8_scratch =
        ternary_s8_codes_bytes(k, tokens) + ternary_s8_scales_bytes(tokens) + 512u;
    return rotated + s8_scratch;
}

std::size_t ternary_projection_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                               std::int32_t tokens) {
    return ternary_rotation_workspace_bytes(output_rows, tokens) +
           ternary_rotation_workspace_bytes(input_rows, tokens);
}

Tensor folded_activation(const Tensor& x, const Weight& weight, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    if (!ternary_rotation_enabled()) { return x; }
    if (!ternary_weight_is_folded(weight)) {
        throw std::invalid_argument(
            "folded ternary weight has no sign block; the artifact must carry "
            "text/hadamard_signs and text/hadamard_widths");
    }
    const DeviceSpan span =
        workspace.alloc_bytes(ternary_rotation_workspace_bytes(weight.k, x.ne[1]));
    Tensor rotated(span.data, DType::BF16, {weight.k, x.ne[1]});
    launch_ternary_rotation(x, rotated, weight, stream);
    return rotated;
}

} // namespace ninfer::ops::detail
