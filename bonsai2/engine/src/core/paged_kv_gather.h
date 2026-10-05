#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace ninfer::detail {

// One device plane slice of a page, as the gather kernel sees it. Byte counts and offsets are
// multiples of 16 (checked by the caller).
struct KvGatherPlane {
    unsigned char* base;   // device plane base
    long long nb2;         // device stride of axis 2 (HeadMajor: one page of one head)
    long long nb3;         // device stride of axis 3 (PageMajor: one page; HeadMajor: one head)
    long long host_offset; // plane offset inside the packed host page record
    long long head_bytes;  // payload of one head of one page
    long long page_bytes;  // payload of all heads of one page
    int heads;
    int pad;
};

// `planes`, `destination` and `pages` may live in mapped pinned host memory.
cudaError_t launch_kv_page_gather(const KvGatherPlane* planes, int plane_count,
                                  const int* destination, const void* pages,
                                  std::size_t page_stride, int page_count, bool head_major,
                                  cudaStream_t stream);

} // namespace ninfer::detail
