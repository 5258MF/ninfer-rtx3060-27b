#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::ops::detail {

// KVMem uses host-resident FP32 sums and BF16 query rows by default. Setting
// NINFER_TERNARY_KVMEM_CPU_RETRIEVAL=0 selects the existing device implementation.
[[nodiscard]] bool kvmem_cpu_retrieval_enabled() noexcept;

struct KvMemCpuQuery {
    const std::uint16_t* rows = nullptr;
    std::uint32_t tokens = 0;
    std::size_t layer_stride = 0; // BF16 rows between successive attention layers
};

// Eq.10: sum over query tokens of mean over layers/heads of softmax over
// included blocks. Sums are FP32 [layer][block][kv_head][dim]; counts are the
// pre-chunk history fills. Division is applied after the dot product, matching
// the GPU scorer. No additional quantization of the index is introduced.
// Excluded and nonpositive-count blocks receive zero mass. Empty candidates
// produce zero scores. Each output segment contains counts.size() scores.
void kvmem_cpu_scores(const float* sums, std::size_t layer_elements,
                      std::uint32_t layers, std::uint32_t kv_heads,
                      std::uint32_t query_heads, std::uint32_t head_dim,
                      std::span<const std::int32_t> counts,
                      std::span<const std::uint8_t> exclude,
                      std::span<const KvMemCpuQuery> queries,
                      std::vector<std::vector<float>>& scores,
                      bool vectorize = true, std::uint32_t threads = 0);

} // namespace ninfer::ops::detail
