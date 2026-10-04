#include "ops/linear/ternary/ternary_rotation.h"

#include "core/device.h"
#include "ops/linear/ternary/ternary_rotation_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

static_assert(sizeof(__nv_bfloat16) == 2, "the rotation workspace math assumes a 2-byte BF16");

// Shared validation for both directions: the transform width, the matching sign block, and the
// [k, tokens] BF16 activation.
void require_rotation_operands(const Tensor& x, const Weight& weight, const char* what) {
    const std::int32_t k      = weight.k;
    const std::int32_t tokens = x.ne[1];

    if (k <= 0 || tokens <= 0) {
        throw std::invalid_argument(std::string(what) + ": K and tokens must be positive");
    }
    if ((k % kBlockSize) != 0) {
        throw std::invalid_argument(std::string(what) +
                                    ": K must be a multiple of the block size");
    }
    if (weight.hadamard_signs == nullptr || weight.hadamard_n_blk <= 0) {
        throw std::invalid_argument(std::string(what) + ": weight carries no sign block");
    }
    if (weight.hadamard_n_blk != k / kBlockSize) {
        throw std::invalid_argument(std::string(what) +
                                    ": sign row count does not match the input width");
    }
    if (x.dtype != DType::BF16 || x.data == nullptr) {
        throw std::invalid_argument(std::string(what) + ": activation must be BF16");
    }
    if (x.ne[0] != k) {
        throw std::invalid_argument(std::string(what) +
                                    ": activation width does not match K");
    }
}

unsigned rotation_grid(std::int32_t k, std::int32_t tokens) {
    const std::int64_t warps =
        static_cast<std::int64_t>(k / kBlockSize) * static_cast<std::int64_t>(tokens);
    return static_cast<unsigned>((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

} // namespace

void launch_ternary_rotation(const Tensor& x, Tensor& out, const Weight& weight,
                             cudaStream_t stream) {
    require_rotation_operands(x, weight, "ternary rotation");
    if (out.dtype != DType::BF16 || out.data == nullptr) {
        throw std::invalid_argument("ternary rotation: out must be BF16");
    }
    if (out.ne[0] != weight.k || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("ternary rotation: expected [K,T] x and [K,T] out");
    }

    // The permutation is only admitted when its geometry actually spans the input width; a
    // partially declared permutation would silently transform a malformed basis. It is applied only
    // when the artifact is known to be tile-ordered (see ternary_gdn_perm_enabled()).
    const bool permuted = ternary_gdn_perm_enabled() && weight.hadamard_perm_rep > 1;
    if (permuted) {
        const std::int64_t span = static_cast<std::int64_t>(weight.hadamard_perm_hd) *
                                  weight.hadamard_perm_nk * weight.hadamard_perm_rep;
        if (weight.hadamard_perm_hd <= 0 || weight.hadamard_perm_nk <= 0 ||
            span != weight.k) {
            throw std::invalid_argument(
                "ternary rotation: folded permutation geometry does not span K");
        }
    }
    const std::int32_t perm_hd  = permuted ? weight.hadamard_perm_hd : 0;
    const std::int32_t perm_nk  = permuted ? weight.hadamard_perm_nk : 0;
    const std::int32_t perm_rep = permuted ? weight.hadamard_perm_rep : 1;

    ternary_rotate_bf16_kernel<<<rotation_grid(weight.k, x.ne[1]),
                                 kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data),
        weight.hadamard_signs, weight.hadamard_n_blk, weight.k, x.ne[1], perm_hd, perm_nk,
        perm_rep, /*inverse=*/0);
    CUDA_CHECK(cudaGetLastError());
}

void launch_ternary_rotation_inverse_inplace(Tensor& data, const Weight& weight,
                                             cudaStream_t stream) {
    require_rotation_operands(data, weight, "ternary rotation (inverse)");
    // The token-embedding table is the only inverse-mapped weight and carries no permutation
    // (llama-model.cpp gates P on `.ssm_out.` alone), so one declared here is a contract error.
    if (weight.hadamard_perm_rep > 1) {
        throw std::invalid_argument(
            "ternary rotation: the embedding path does not carry a folded permutation");
    }

    ternary_rotate_inverse_inplace_bf16_kernel<<<rotation_grid(weight.k, data.ne[1]),
                                                 kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
        static_cast<__nv_bfloat16*>(data.data), weight.hadamard_signs, weight.hadamard_n_blk,
        weight.k, data.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
