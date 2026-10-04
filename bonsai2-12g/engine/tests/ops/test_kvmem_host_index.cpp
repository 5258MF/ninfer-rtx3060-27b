#include "core/arena.h"
#include "core/device.h"
#include "ops/kvmem/mean_k_index.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;

int main() {
    try {
        DeviceContext device(0);
        constexpr int layers = 3, heads = 4, dim = 256, blocks = 8, width = 87;
        ops::MeanKIndex gpu(layers, heads, dim, blocks, false);
        ops::MeanKIndex host(layers, heads, dim, blocks, true);
        DeviceArena input(heads * dim * width * 2U);
        Tensor raw = input.alloc(DType::BF16, {heads * dim, width});
        std::vector<std::uint16_t> keys(raw.numel());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            // Exactly represented, signed, nonuniform BF16 input.
            keys[i] = static_cast<std::uint16_t>(0x3f00U + (i % 127) + ((i % 3 == 0) ? 0x8000U : 0U));
        }
        CUDA_CHECK(cudaMemcpyAsync(raw.data, keys.data(), raw.bytes(), cudaMemcpyHostToDevice, device.stream));
        CUDA_CHECK(gpu.zero(device.stream));
        CUDA_CHECK(host.zero(device.stream));
        for (int layer = 0; layer < layers; ++layer) {
            gpu.append_layer(raw, layer, 5, 0, device.stream);
            host.append_layer(raw, layer, 5, 0, device.stream);
            gpu.append_layer(raw, layer, 17, 0, device.stream);
            host.append_layer(raw, layer, 17, 0, device.stream);
            gpu.append_layer(raw, layer, width, 1, device.stream);
            host.append_layer(raw, layer, width, 1, device.stream);
        }
        std::vector<float> want, got;
        CUDA_CHECK(gpu.save_used(want, device.stream));
        CUDA_CHECK(host.save_used(got, device.stream));
        if (got != want) { throw std::runtime_error("mapped host append differs from device FP32 sums/counts"); }
        float counts[3]{};
        CUDA_CHECK(host.read_counts(1, 0, 3, counts, device.stream));
        if (counts[0] != 22 || counts[1] != 64 || counts[2] != 23) {
            throw std::runtime_error("partial-block counts incorrect");
        }
        CUDA_CHECK(host.zero(device.stream));
        CUDA_CHECK(host.load_used(got, 3, 23, device.stream));
        std::vector<float> restored;
        CUDA_CHECK(host.save_used(restored, device.stream));
        if (restored != want || host.blocks_written() != 3 || host.tail_fill() != 23) {
            throw std::runtime_error("host index snapshot/restore failed");
        }
        // Capture/replay must execute the append each time, rather than doing it
        // once while recording a graph or leaving a pointer to temporary memory.
        CUDA_CHECK(host.zero(device.stream));
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
        host.append_layer(raw, 0, 7, 0, device.stream);
        cudaGraph_t graph{};
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        cudaGraphExec_t exec{};
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaGraphLaunch(exec, device.stream));
        CUDA_CHECK(cudaGraphLaunch(exec, device.stream));
        CUDA_CHECK(host.read_counts(0, 0, 1, counts, device.stream));
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        if (counts[0] != 14) { throw std::runtime_error("host index graph replay lost an append"); }
        std::cout << "Host FP32 index: device parity, partial blocks, reset, snapshot/restore, graph replay PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
