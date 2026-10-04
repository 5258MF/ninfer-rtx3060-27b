// Blackwell-only (SM120) k8v4/NVFP4 entry points for non-sm_120a builds.
//
// k8v4 uses the mixed 8x4 MMA (kind::f8f6f4) and NVFP4 uses FP4 (e2m1) instructions; neither
// is supported on sm_89. The owning kernels are excluded from non-120 builds (see
// src/CMakeLists.txt). These stubs keep the dispatch layer linkable and turn any use on this
// architecture into a clear runtime error.
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void throw_kv_sm120_only(const char* feature) {
    throw std::runtime_error(
        std::string(feature) +
        ": k8v4/NVFP4 KV cache requires the SM120 (Blackwell) kernels (kind::f8f6f4 / e2m1), "
        "which are not compiled for this architecture; use --kv-dtype bf16|int8|fp8 instead");
}

[[noreturn]] void throw_w4a4_sm120_only(const char* feature) {
    throw std::runtime_error(std::string(feature) +
                             ": NVFP4 W4A4 (TMA) execution requires the SM120 (Blackwell) TMA "
                             "kernels, which are not compiled for this architecture");
}

} // namespace

// KV-cache append (ops/kv_cache/append/launch.h).
void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    throw_kv_sm120_only("NVFP4 KV-cache append");
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, PagedKVBatchLayerView,
                                        cudaStream_t) {
    throw_kv_sm120_only("NVFP4 batch KV-cache append");
}

void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    throw_kv_sm120_only("k8v4 KV-cache append");
}

void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, PagedKVBatchLayerView,
                                       cudaStream_t) {
    throw_kv_sm120_only("k8v4 batch KV-cache append");
}

// Causal attention (ops/softmax_attention/dense/causal_cache/launch.h).
void causal_attention_small_t_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&, float,
                                           PagedKVBatchLayerView,
                                           CausalAttentionExecutionEnvelope, std::int32_t,
                                           std::int32_t, Tensor&, Tensor&, Tensor&, Tensor&,
                                           cudaStream_t) {
    throw_kv_sm120_only("NVFP4 small-T causal attention");
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    throw_kv_sm120_only("NVFP4 cached small-T causal attention");
}

void causal_attention_small_t_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView,
                                          CausalAttentionExecutionEnvelope, std::int32_t,
                                          std::int32_t, Tensor&, Tensor&, Tensor&, Tensor&,
                                          cudaStream_t) {
    throw_kv_sm120_only("k8v4 small-T causal attention");
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&,
                                                 CausalAttentionExecutionEnvelope, Tensor&,
                                                 Tensor&, Tensor&, Tensor&, cudaStream_t) {
    throw_kv_sm120_only("k8v4 cached small-T causal attention");
}

void causal_attention_prompt_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, const Tensor&, float,
                                         PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    throw_kv_sm120_only("k8v4 prompt causal attention");
}

void causal_attention_prompt_k8v4_attention_launch(const Tensor&, const Tensor&, float,
                                                   const PagedKVLayerView&, Tensor&,
                                                   cudaStream_t) {
    throw_kv_sm120_only("k8v4 cached prompt causal attention");
}

// Prompt attention kernel launchers (prompt_nvfp4_non_rdc_launch.h).
void causal_attention_prompt_nvfp4_kernel_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&, cudaStream_t) {
    throw_kv_sm120_only("NVFP4 prompt causal attention kernel");
}

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor&, const Tensor&,
                                                       const Tensor&, const Tensor&, float,
                                                       const PagedKVBatchLayerView&, Tensor&,
                                                       cudaStream_t) {
    throw_kv_sm120_only("NVFP4 batch prompt causal attention kernel");
}

// NVFP4 W4A4 (TMA) launchers (nvfp4_w4a4_tma_launch.h / nvfp4_linear_swiglu_w4a4_tma_launch.h).
void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    throw_w4a4_sm120_only("NVFP4 W4A4 linear");
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, std::int32_t, float,
                                     cudaStream_t) {
    throw_w4a4_sm120_only("NVFP4 W4A4 attention input projection");
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    throw_w4a4_sm120_only("NVFP4 W4A4 GDN input projection");
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    throw_w4a4_sm120_only("NVFP4 W4A4 linear add");
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    throw_w4a4_sm120_only("NVFP4 W4A4 linear SwiGLU");
}

} // namespace ninfer::ops::detail
