#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Streaming-multiprocessor count of the device this process runs on, queried once and cached.
// Launch geometry that deliberately fills exactly one resident wave reads the count from here
// instead of transcribing a per-part literal; the product runs one resident model on one device,
// so a single cached query is the whole device set.
int device_sm_count();

// Compile-time mirror of device_sm_count() for the architecture this build targets. __device__
// launch policies cannot query the runtime, and the host launcher that must reproduce such a
// policy exactly has to agree with it at compile time; those two sites use this constant, every
// other site uses device_sm_count().
#if defined(NINFER_SM89)
inline constexpr int kTargetSmCount = 128; // NVIDIA GeForce RTX 4090
#elif defined(NINFER_SM86)
// sm_86 spans a wide SM-count range (3090 = 82, 3080 = 68, 3060 = 28). The two sites that
// consume this constant must agree with the *physical* device, otherwise a cooperative
// launch sized for a larger part fails with cudaErrorCooperativeLaunchTooLarge.
// Override with -DNINFER_TARGET_SM_COUNT=<n> when building for a different sm_86 part.
#if defined(NINFER_TARGET_SM_COUNT)
inline constexpr int kTargetSmCount = NINFER_TARGET_SM_COUNT;
#else
inline constexpr int kTargetSmCount = 28; // NVIDIA GeForce RTX 3060 (GA106)
#endif
#else
#error "NInfer requires NINFER_SM86 or NINFER_SM89"
#endif

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
