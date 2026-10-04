// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attr_bf16);
#if !defined(NINFER_SM89) && !defined(NINFER_RK4_SM86)
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    CUDA_CHECK(attr_i8);
    static const cudaError_t attr_rk8v4 =
        cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    CUDA_CHECK(attr_rk8v4);
#endif

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
#if defined(NINFER_SM89) || defined(NINFER_RK4_SM86)
    // Kernel i8 del fork sergiuszm/ninfer-4090 (modos packed int4 / E8).
    if (cache.storage == KvCacheStorage::RK4V4E8 ||
        cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64 ||
        cache.storage == KvCacheStorage::Int8Group64) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        const auto launch_i8 = [&]<bool PackedV, bool RotateK, bool RotateV, bool PackedK,
                                   bool E8Root>() {
            static const cudaError_t attr_i8 = cudaFuncSetAttribute(
                causal_attention_prompt_i8_kernel<Geometry, PackedV, RotateK, RotateV, PackedK,
                                                  E8Root, Metadata>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
            CUDA_CHECK(attr_i8);
            causal_attention_prompt_i8_kernel<Geometry, PackedV, RotateK, RotateV, PackedK,
                                              E8Root, Metadata>
                <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::int8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data),
                    static_cast<const __half*>(cache_k_scale.data),
                    static_cast<const __half*>(cache_v_scale.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        };
        // rk4v4 y rk4v4-e8 comparten la misma instancia de carga (K i4, V i4, rotaciones);
        // la reticula E8 solo interviene al codificar (append), no al leer.
        if (cache.storage == KvCacheStorage::RK4V4E8 ||
            cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64) {
            launch_i8.template operator()<true, true, true, true, false>();
        } else {
            launch_i8.template operator()<false, false, false, false, false>();
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __half*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
    // Los modos con V rotada (rk4v4 / rk4v4-e8) devuelven el PV al espacio original
    // con la inversa de la rotacion H64 (self-inverse).
    if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueGroup64 ||
        cache.storage == KvCacheStorage::RK4V4E8) {
        kv_cache_inverse_rotate_output_kernel<Geometry::QHeads>
            <<<tokens * Geometry::QHeads * kKVCacheInt8Groups, 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}
#else
    const auto launch_prompt_i8 = [&]<bool PackedValues>() {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        causal_attention_prompt_i8_kernel<Geometry, Metadata, PackedValues>
            <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    if (cache.storage == KvCacheStorage::Int8Group64) {
        launch_prompt_i8.template operator()<false>();
    } else if (cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup32) {
        launch_prompt_i8.template operator()<true>();
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __half*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
