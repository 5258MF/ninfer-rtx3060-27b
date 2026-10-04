#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Row slice of a grouped row-split *ternary* parent (PTQ1_0_G128 / PQ2_0_G128).
//
// The fused families store one parent matrix that the graph splits into several logical
// projections (the GDN value/z parent carries [value; z], the attention query/key parent carries
// [query; key]). The binder's own row_view() slices the same planes the same way, so this mirrors
// it exactly: the codes plane, then the high plane, then the scales plane, each laid out
// row-major over [rows, groups].
//
// Plane geometry is per-format: PQ2_0 carries 32 code bytes per 128-weight group and no high
// plane, PTQ1_0 carries 24 code bytes plus 2 high bytes. Assuming the Q4/Q5 32/0 split for PTQ1_0
// would slice the wrong byte ranges, which produces plausible-looking garbage rather than an
// error -- so the geometry is derived from qtype, never from a default.
[[nodiscard]] inline Weight ternary_row_view(const Weight& block, std::int32_t row_begin,
                                            std::int32_t row_count) {
    const bool ptq1 = block.qtype == QType::PTQ1_0_G128;
    if (!ptq1 && block.qtype != QType::PQ2_0_G128) {
        throw std::invalid_argument("ternary row view: parent is not a ternary format");
    }
    if (block.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("ternary row view: parent is not row-split");
    }
    if (block.group <= 0 || block.scales == nullptr || block.qdata == nullptr) {
        throw std::invalid_argument("ternary row view: parent geometry is incomplete");
    }
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n) {
        throw std::invalid_argument("ternary row view: row range is out of bounds");
    }
    const std::uint64_t groups     = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group  = ptq1 ? 24 : 32;
    const std::uint64_t high_group = ptq1 ? 2 : 0;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * groups * low_group;
    out.qhigh = high_group == 0 ? nullptr
                                : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * groups * high_group;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * groups * 2;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

} // namespace ninfer::ops::detail
