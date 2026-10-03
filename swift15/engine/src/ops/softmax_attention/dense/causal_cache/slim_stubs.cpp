// NINFER_SLIM_3060: the FP8, NVFP4 and K8V4 KV attention units are not built. Their entry
// points stay so the dispatchers link; serve refuses those KV dtypes before any of these runs.

#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

namespace {
[[noreturn]] void slim_refuse(const char* name) {
    throw std::runtime_error(std::string(name) +
                             ": this NINFER_SLIM_3060 build supports only int8 / rk8v4 KV caches");
}
} // namespace

// NOLINTNEXTLINE
void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    slim_refuse("causal_attention_small_t_fp8_launch");
}

// NOLINTNEXTLINE
void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream) {
    slim_refuse("causal_attention_cached_small_t_fp8_launch");
}

// NOLINTNEXTLINE
void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    slim_refuse("causal_attention_small_t_nvfp4_launch");
}

// NOLINTNEXTLINE
void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream) {
    slim_refuse("causal_attention_cached_small_t_nvfp4_launch");
}

// NOLINTNEXTLINE
void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    slim_refuse("causal_attention_small_t_k8v4_launch");
}

// NOLINTNEXTLINE
void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 Tensor& partial_acc, Tensor& partial_m,
                                                 Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream) {
    slim_refuse("causal_attention_cached_small_t_k8v4_launch");
}

// NOLINTNEXTLINE
void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream) {
    slim_refuse("causal_attention_prompt_fp8_launch");
}

// NOLINTNEXTLINE
void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream) {
    slim_refuse("causal_attention_prompt_fp8_attention_launch");
}

// NOLINTNEXTLINE
void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream) {
    slim_refuse("causal_attention_prompt_nvfp4_launch");
}

// NOLINTNEXTLINE
void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream) {
    slim_refuse("causal_attention_prompt_nvfp4_attention_launch");
}

// NOLINTNEXTLINE
void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream) {
    slim_refuse("causal_attention_prompt_k8v4_launch");
}

// NOLINTNEXTLINE
void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream) {
    slim_refuse("causal_attention_prompt_k8v4_attention_launch");
}

} // namespace ninfer::ops::detail
