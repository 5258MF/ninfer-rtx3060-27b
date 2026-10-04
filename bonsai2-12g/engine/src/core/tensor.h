#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QType : std::uint16_t {
    Q4G64_F16S           = 0,
    Q5G64_F16S           = 1,
    Q6G64_F16S           = 2,
    W8G32_F16S           = 3,
    BF16_CTRL            = 4,
    FP32_CTRL            = 5,
    I32_CTRL             = 6,
    NVFP4                = 7,
    FP8_E4M3FN_ROW_BF16S = 8,
    // Prism-private ternary (Bonsai 2 27B): group 128, base-3 / 2-bit codes.
    PTQ1_0_G128          = 9,
    PQ2_0_G128           = 10,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4G64_F16S;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // Prism ternary weights are stored folded into a rotated basis: the model computes
    // y = W' * (H * (s * P * x)), so every activation feeding one of the folded weights must
    // be mapped by (signs, then normalized Sylvester-Hadamard) before the matmul, and the
    // token-embedding lookup must be mapped by (rotate, then signs) afterwards.
    //
    // hadamard_signs points at the sign block for THIS weight's input width (each width owns
    // k/1024 rows of 1024 F32 +-1 values, and row b within the width is selected by the
    // activation's 1024-block index). nullptr means no transform. hadamard_n_blk = k/1024.
    const float* hadamard_signs = nullptr;
    std::int32_t hadamard_n_blk = 0;

    // Folded-basis feature permutation P, applied BEFORE the signs and the rotation:
    //
    //     x.view(perm_hd, perm_nk, perm_rep) -> swap the last two axes -> flatten
    //
    // Only the GDN output projection stores grouped columns while the runtime produces tiled
    // v-heads, so perm_rep > 1 enables it and everything else leaves it at 1. Geometry comes
    // from the model's head counts: perm_hd = K/n_v, perm_nk = n_k, perm_rep = n_v/n_k.
    std::int32_t hadamard_perm_hd  = 0;
    std::int32_t hadamard_perm_nk  = 0;
    std::int32_t hadamard_perm_rep = 1;
};

} // namespace ninfer
