#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Folded (rotated) ternary basis support for the Prism/Bonsai ports.
//
// The ternary linear weights are stored folded into a rotated basis, so the model computes
//
//     y = W' * (H * (s * P * x))
//
// where P is an optional column permutation (only the GDN output projection carries one), s is
// an explicit +-1 diagonal, and H is the normalized Sylvester-Hadamard transform over blocks of
// 1024. Everything below is driven by data the binder already attached to Weight
// (hadamard_signs / hadamard_n_blk / hadamard_perm_*), so the op layer never needs to know which
// model layers are folded: `qtype` alone decides.
//
// The transform is applied to the ACTIVATION, never to the weight. Folding H and s back into W
// would make W a dense float matrix (`W = W' * H * s`), which is not ternary representable and
// would re-inflate the artifact from 7 GB to ~17 GB -- the exact failure mode this port exists
// to avoid.

// True when this weight's activation must be mapped into the folded basis. False for every
// non-ternary weight, which is why the non-ternary pipelines are unaffected.
[[nodiscard]] bool ternary_weight_is_folded(const Weight& weight) noexcept;

// NINFER_TERNARY_HADAMARD=0 disables the transform at run time.
//
// This is the bisection switch between the two ways an end-to-end ternary run can be wrong:
// decode errors (wrong codes / wrong scale / wrong row order) and rotation errors (wrong
// butterfly, wrong sign row, wrong permutation). With the transform off, the engine still runs
// the full forward pass and still reports a decode speed, so the M4 throughput number can be
// measured before the transform is trusted; the output is numerically meaningless either way,
// and a perplexity that is *not* in the hundreds is the signal that the rotation is correct.
[[nodiscard]] bool ternary_rotation_enabled();

// Whether the folded GDN feature permutation P is applied.
//
// llama.cpp applies P because ITS runtime keeps the GDN V channels tiled, matching the GGUF's
// `attn_qkv` row order, while the folded `ssm_out` keeps its training (grouped) column order
// (`conversion/qwen.py:617-626`). The GGUF metadata records this as
// `prism.hadamard.gdn_v_grouped = 1` -- "v is already grouped, do not permute a second time".
//
// This port's packer normalizes EVERY GDN tensor to the grouped (HF) order: `gdn/value_z` applies
// tiled->grouped to the attn_qkv V rows and to attn_gate, and `ssm_conv1d`/`ssm_alpha`/`ssm_beta`
// are reordered the same way. The runtime therefore already produces grouped V, so applying P here
// permutes a SECOND time and scrambles 48 of the 64 layers.
//
// Default: off. NINFER_TERNARY_GDN_PERM=1 restores the llama.cpp behaviour, which is what a
// tile-ordered artifact would need.
[[nodiscard]] bool ternary_gdn_perm_enabled();

// Device bytes needed to hold a rotated activation of shape [k, tokens], BF16.
[[nodiscard]] std::size_t ternary_rotation_workspace_bytes(std::int32_t k, std::int32_t tokens);

// Bytes a fused family needs to decompose into a ternary GEMM: a scoped [output_rows, tokens]
// BF16 projection scratch (the GEMM writes [N, T] and the family's epilogue then folds it in) plus
// the [input_rows, tokens] rotation buffer the GEMM maps the activation through.
[[nodiscard]] std::size_t ternary_projection_workspace_bytes(std::int32_t output_rows,
                                                             std::int32_t input_rows,
                                                             std::int32_t tokens);

// Map `x` ([k, tokens], BF16) into the folded basis of `weight`, writing [k, tokens] BF16.
// Applies P, then the signs, then H: out = H * (s * (P * x)).
void launch_ternary_rotation(const Tensor& x, Tensor& out, const Weight& weight,
                             cudaStream_t stream);

// Inverse for the token-embedding path: `out = s * (H * x)`, applied IN PLACE to a [k, tokens]
// BF16 activation.
//
// The embedding table stores its rows folded into the rotated basis, so the looked-up rows must be
// mapped back before they can enter the residual stream. There is no permutation on this path
// (llama-model.cpp gates P on `.ssm_out.` alone), and H and s are both involutions, so the inverse
// is the same transform with the sign multiply on the other side.
//
// In place on purpose: the gathered embedding is a freshly produced activation that nothing else
// reads, so mapping it in place costs no scratch buffer at all -- which matters because the
// embedding op has no workspace parameter.
void launch_ternary_rotation_inverse_inplace(Tensor& data, const Weight& weight,
                                            cudaStream_t stream);

// Convenience wrapper around launch_ternary_rotation for callers that feed several folded weights
// from one activation: allocates the scratch from `workspace` and returns the mapped activation.
//
// Returns `x` unchanged when NINFER_TERNARY_HADAMARD=0, and throws when the rotation is enabled
// but the weight carries no sign block -- a folded weight without signs means the artifact is
// missing text/hadamard_signs, and running it untransformed would produce plausible garbage.
//
// The CALLER owns the workspace scope: the returned Tensor points into the arena and stays valid
// only until that scope ends.
[[nodiscard]] Tensor folded_activation(const Tensor& x, const Weight& weight,
                                       WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
