#pragma once

#include "core/device.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <mutex>

namespace ninfer {

inline std::size_t physical_free_device_bytes(std::size_t* out_cuda_free = nullptr,
                                              std::size_t* out_nvml_free = nullptr,
                                              bool apply_env_cap         = true) {
    std::size_t cuda_free  = 0;
    std::size_t cuda_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&cuda_free, &cuda_total));
    std::size_t nvml_free = 0;
#if defined(_WIN32)
    struct NvmlMemory {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };
    using NvmlInitFn          = int (*)();
    using NvmlGetByPciBusIdFn = int (*)(const char*, void**);
    using NvmlGetByIndexFn    = int (*)(unsigned int, void**);
    using NvmlGetMemoryInfoFn = int (*)(void*, NvmlMemory*);

    static std::once_flag nvml_once;
    static NvmlGetByPciBusIdFn fn_by_pci  = nullptr;
    static NvmlGetByIndexFn fn_by_index   = nullptr;
    static NvmlGetMemoryInfoFn fn_meminfo = nullptr;
    std::call_once(nvml_once, []() {
        HMODULE mod = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!mod) { mod = LoadLibraryW(L"nvml.dll"); }
        if (!mod) { return; }
        auto fn_init = reinterpret_cast<NvmlInitFn>(GetProcAddress(mod, "nvmlInit_v2"));
        if (!fn_init) { fn_init = reinterpret_cast<NvmlInitFn>(GetProcAddress(mod, "nvmlInit")); }
        if (!fn_init || fn_init() != 0) { return; }
        fn_by_pci = reinterpret_cast<NvmlGetByPciBusIdFn>(
            GetProcAddress(mod, "nvmlDeviceGetHandleByPciBusId_v2"));
        fn_by_index = reinterpret_cast<NvmlGetByIndexFn>(
            GetProcAddress(mod, "nvmlDeviceGetHandleByIndex_v2"));
        fn_meminfo = reinterpret_cast<NvmlGetMemoryInfoFn>(
            GetProcAddress(mod, "nvmlDeviceGetMemoryInfo"));
    });
    if (fn_meminfo != nullptr) {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        void* handle        = nullptr;
        char pci_bus_id[64] = {};
        if (fn_by_pci != nullptr &&
            cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), dev) == cudaSuccess) {
            if (fn_by_pci(pci_bus_id, &handle) != 0) { handle = nullptr; }
        }
        if (handle == nullptr && fn_by_index != nullptr) {
            if (fn_by_index(static_cast<unsigned int>(dev), &handle) != 0) { handle = nullptr; }
        }
        if (handle != nullptr) {
            NvmlMemory mem{};
            if (fn_meminfo(handle, &mem) == 0 && mem.free > 0) {
                nvml_free = static_cast<std::size_t>(mem.free);
            }
        }
    }
#endif
    std::size_t effective = (nvml_free > 0) ? std::min(cuda_free, nvml_free) : cuda_free;
    if (apply_env_cap) {
        if (const char* env_cap = std::getenv("NINFER_FREE_VRAM_MIB");
            env_cap != nullptr && env_cap[0] != '\0') {
            char* end                        = nullptr;
            const unsigned long long cap_mib = std::strtoull(env_cap, &end, 10);
            if (end != env_cap && cap_mib > 0) {
                const std::size_t cap_bytes =
                    static_cast<std::size_t>(cap_mib) * 1024ULL * 1024ULL;
                effective = std::min(effective, cap_bytes);
            }
        }
    }
    if (out_cuda_free != nullptr) { *out_cuda_free = cuda_free; }
    if (out_nvml_free != nullptr) { *out_nvml_free = nvml_free; }
    return effective;
}

} // namespace ninfer
