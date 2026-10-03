#pragma once

#include <cstddef>

namespace ninfer {

std::size_t physical_free_device_bytes(std::size_t* out_cuda_free = nullptr,
                                       std::size_t* out_nvml_free = nullptr);

} // namespace ninfer
