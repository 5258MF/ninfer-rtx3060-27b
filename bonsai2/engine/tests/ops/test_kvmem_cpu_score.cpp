#include "ops/kvmem/kvmem_cpu_score.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using ninfer::ops::detail::KvMemCpuQuery;
using ninfer::ops::detail::kvmem_cpu_scores;

namespace {
float represented(std::uint16_t bits) {
    const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint16_t bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

// Independent FP64 Eq.10 oracle; neither packing nor the production helpers are used.
std::vector<std::vector<double>> oracle(const std::vector<float>& sums,
    std::size_t stride, unsigned layers, unsigned kv, unsigned heads, unsigned dim,
    const std::vector<int>& counts, const std::vector<std::uint8_t>& exclude,
    const std::vector<KvMemCpuQuery>& query) {
    std::vector<std::vector<double>> out(query.size(), std::vector<double>(counts.size()));
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    for (unsigned l = 0; l < layers; ++l) {
        for (std::size_t s = 0; s < query.size(); ++s) {
            for (unsigned r = 0; r < query[s].tokens; ++r) {
                for (unsigned h = 0; h < heads; ++h) {
                    std::vector<double> logit(counts.size(), -INFINITY);
                    double mx = -INFINITY;
                    for (std::size_t b = 0; b < counts.size(); ++b) {
                        if (counts[b] <= 0 || (!exclude.empty() && exclude[b])) { continue; }
                        double dot = 0;
                        for (unsigned d = 0; d < dim; ++d) {
                            const auto qoff = (l * query[s].layer_stride + r) * heads * dim + h * dim + d;
                            const auto koff = l * stride + (b * kv + h / (heads / kv)) * dim + d;
                            dot += static_cast<double>(represented(query[s].rows[qoff])) * sums[koff];
                        }
                        logit[b] = dot * scale / counts[b];
                        mx = std::max(mx, logit[b]);
                    }
                    if (!std::isfinite(mx)) { continue; }
                    double mass = 0;
                    for (const auto x : logit) { mass += std::exp(x - mx); }
                    for (std::size_t b = 0; b < counts.size(); ++b) {
                        out[s][b] += std::exp(logit[b] - mx) / mass / (layers * heads);
                    }
                }
            }
        }
    }
    return out;
}

void run(unsigned blocks, unsigned layers, unsigned kv, unsigned heads, unsigned dim,
         bool all_excluded, bool uniform) {
    const std::size_t stride = static_cast<std::size_t>(blocks + 5) * kv * dim + 19;
    std::vector<float> sums(layers * stride, 123.0F);
    std::vector<int> counts(blocks);
    std::vector<std::uint8_t> exclude(blocks);
    std::mt19937 rng(7301 + blocks + dim);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
    for (unsigned b = 0; b < blocks; ++b) {
        counts[b] = b % 7 == 0 ? 0 : (b % 3 == 0 ? 7 : 64);
        exclude[b] = all_excluded || b % 5 == 1;
        for (unsigned l = 0; l < layers; ++l) {
            for (unsigned k = 0; k < kv * dim; ++k) {
                sums[l * stride + b * kv * dim + k] = uniform ? 0.0F : dist(rng) * counts[b];
            }
        }
    }
    std::vector<std::vector<std::uint16_t>> buffers(3);
    std::vector<KvMemCpuQuery> queries;
    for (unsigned s = 0; s < 3; ++s) {
        const unsigned n = s == 2 ? 0 : (s == 0 ? 3 : 9);
        const unsigned row_stride = n + 5;
        buffers[s].resize(layers * row_stride * heads * dim, bf16(99));
        for (auto& q : buffers[s]) { q = bf16(dist(rng)); }
        queries.push_back({buffers[s].data(), n, row_stride});
    }
    const auto want = oracle(sums, stride, layers, kv, heads, dim, counts, exclude, queries);
    for (const bool vectorize : {false, true}) {
        for (const unsigned threads : {1U, 3U}) {
            std::vector<std::vector<float>> got;
            kvmem_cpu_scores(sums.data(), stride, layers, kv, heads, dim,
                              counts, exclude, queries, got, vectorize, threads);
            for (unsigned s = 0; s < queries.size(); ++s) {
                double total = 0;
                for (unsigned b = 0; b < blocks; ++b) {
                    const double err = std::abs(got[s][b] - want[s][b]);
                    if (!std::isfinite(got[s][b]) || err > 3e-6 + 3e-5 * std::abs(want[s][b])) {
                        throw std::runtime_error("CPU Eq.10 differs from FP64 oracle");
                    }
                    if ((counts[b] <= 0 || exclude[b]) && got[s][b] != 0) {
                        throw std::runtime_error("excluded block received mass");
                    }
                    total += got[s][b];
                }
                const double expected = std::accumulate(want[s].begin(), want[s].end(), 0.0);
                if (std::abs(total - expected) > 2e-5 * (1 + queries[s].tokens)) {
                    throw std::runtime_error("CPU Eq.10 mass is not query token count");
                }
            }
        }
    }
}
} // namespace

int main() {
    try {
        for (unsigned blocks : {1U, 7U, 16U, 17U, 37U, 65U}) {
            run(blocks, 3, 2, 6, 32, false, false);
            run(blocks, 3, 2, 6, 32, true, false);
            run(blocks, 3, 2, 6, 32, false, true);
        }
        run(97, 16, 4, 24, 256, false, false);
        std::cout << "CPU Eq.10: FP64 oracle, BF16 inputs, partial counts, masks, padding,"
                     " segments, scalar/AVX2, one/multiple threads PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
