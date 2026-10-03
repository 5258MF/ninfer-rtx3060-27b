// 3060: batched Host -> Device KV page gather.
//
// The KVMem placement promotes several hundred scattered pages per turn. The per-run path issues
// one cudaMemcpy2DAsync per plane (64 planes) per contiguous run from pageable memory, i.e. tens
// of thousands of small driver-staged copies at 3-6 GB/s. Here the host packs whole page records
// into a mapped pinned staging buffer and one kernel per batch reads them over PCIe (zero copy,
// no device staging memory) and scatters every plane slice to its page.
#include "core/paged_kv_gather.h"

#include <cuda_runtime.h>

namespace ninfer::detail {
namespace {

__device__ __forceinline__ void copy16(unsigned char* dst, const unsigned char* src,
                                       long long bytes) {
    const long long n16 = bytes >> 4;
    auto* d4            = reinterpret_cast<uint4*>(dst);
    const auto* s4      = reinterpret_cast<const uint4*>(src);
    for (long long k = threadIdx.x; k < n16; k += blockDim.x) { d4[k] = s4[k]; }
}

// One block per (page in batch, plane in range).
__global__ void kv_page_gather_kernel(const KvGatherPlane* __restrict__ planes, int plane_count,
                                      const int* __restrict__ destination,
                                      const unsigned char* __restrict__ pages,
                                      long long page_stride, int head_major) {
    const int page  = static_cast<int>(blockIdx.x) / plane_count;
    const int plane = static_cast<int>(blockIdx.x) % plane_count;
    const KvGatherPlane p = planes[plane];
    const long long index = destination[page];
    const unsigned char* src = pages + page * page_stride + p.host_offset;
    if (head_major) {
        for (int head = 0; head < p.heads; ++head) {
            copy16(p.base + head * p.nb3 + index * p.nb2, src + head * p.head_bytes, p.head_bytes);
        }
    } else {
        copy16(p.base + index * p.nb3, src, p.page_bytes);
    }
}

} // namespace

cudaError_t launch_kv_page_gather(const KvGatherPlane* planes, int plane_count,
                                  const int* destination, const void* pages,
                                  std::size_t page_stride, int page_count, bool head_major,
                                  cudaStream_t stream) {
    if (plane_count <= 0 || page_count <= 0) { return cudaSuccess; }
    const unsigned blocks = static_cast<unsigned>(plane_count) * static_cast<unsigned>(page_count);
    kv_page_gather_kernel<<<blocks, 256, 0, stream>>>(
        planes, plane_count, destination, static_cast<const unsigned char*>(pages),
        static_cast<long long>(page_stride), head_major ? 1 : 0);
    return cudaGetLastError();
}

} // namespace ninfer::detail
