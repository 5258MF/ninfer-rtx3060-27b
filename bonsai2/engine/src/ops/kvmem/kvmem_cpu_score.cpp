#include "ops/kvmem/kvmem_cpu_score.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define KVMEM_CPU_X86 1
#endif

namespace ninfer::ops::detail {
namespace {

inline float kvmem_cpu_bf16(std::uint16_t v) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

#if KVMEM_CPU_X86
inline bool kvmem_cpu_has_avx2() noexcept {
    static const bool ok = [] {
#if defined(_MSC_VER)
        int r[4] = {};
        __cpuid(r, 0);
        if (r[0] < 7) { return false; }
        __cpuid(r, 1);
        const bool fma = (r[2] & (1 << 12)) != 0, osx = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
        if (!(fma && osx && avx)) { return false; }
        if ((_xgetbv(0) & 6) != 6) { return false; }
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;
#else
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    }();
    return ok;
}

#if defined(__GNUC__) && !defined(_MSC_VER)
#define KVMEM_CPU_AVX2 __attribute__((target("avx2,fma")))
#else
#define KVMEM_CPU_AVX2
#endif

// exp for x <= 0 (softmax after max subtraction); Cephes-style, rel. error ~1e-7, clamps < -87.
KVMEM_CPU_AVX2 inline __m256 kvmem_cpu_exp256(__m256 x) noexcept {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0F));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341F);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375F), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4F), x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4F);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1F));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1F));
    const __m256 xx = _mm256_mul_ps(x, x);
    y = _mm256_fmadd_ps(y, xx, _mm256_add_ps(x, _mm256_set1_ps(1.0F)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

// logits[h][j] = sum_d q[h][d] * kT[d][j] for nh (<= 6) heads, j in [0, bi8) (bi8 % 8 == 0).
KVMEM_CPU_AVX2 inline void kvmem_cpu_logits_avx2(const float* q, std::uint32_t nh, std::uint32_t dim, const float* kT,
                                     std::uint32_t bi8, float* logits) noexcept {
    for (std::uint32_t j = 0; j < bi8; j += 8) {
        __m256 acc[6] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                         _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (std::uint32_t d = 0; d < dim; ++d) {
            const __m256 k = _mm256_loadu_ps(kT + static_cast<std::size_t>(d) * bi8 + j);
            for (std::uint32_t h = 0; h < nh; ++h) {
                acc[h] = _mm256_fmadd_ps(_mm256_set1_ps(q[h * dim + d]), k, acc[h]);
            }
        }
        for (std::uint32_t h = 0; h < nh; ++h) { _mm256_storeu_ps(logits + static_cast<std::size_t>(h) * bi8 + j, acc[h]); }
    }
}

KVMEM_CPU_AVX2 inline float kvmem_cpu_max_avx2(const float* x, std::uint32_t n) noexcept;

// In-place softmax mass over [0, bi) (padding up to bi8 ignored), accumulated as acc[j] += w * p[j].
KVMEM_CPU_AVX2 inline void kvmem_cpu_softmax_acc_avx2(const float* logits, std::uint32_t bi, std::uint32_t bi8,
                                          float w, float* tmp, float* acc) noexcept {
    const float mx = kvmem_cpu_max_avx2(logits, bi);
    const __m256 vmx = _mm256_set1_ps(mx);
    __m256 vs = _mm256_setzero_ps();
    for (std::uint32_t j = 0; j < bi8; j += 8) {
        __m256 e = kvmem_cpu_exp256(_mm256_sub_ps(_mm256_loadu_ps(logits + j), vmx));
        if (j + 8 > bi) {   // zero the padding lanes
            alignas(32) float m[8];
            for (std::uint32_t t = 0; t < 8; ++t) { m[t] = (j + t < bi) ? 1.0F : 0.0F; }
            e = _mm256_mul_ps(e, _mm256_load_ps(m));
        }
        _mm256_storeu_ps(tmp + j, e);
        vs = _mm256_add_ps(vs, e);
    }
    alignas(32) float s8[8];
    _mm256_store_ps(s8, vs);
    const float sum = s8[0] + s8[1] + s8[2] + s8[3] + s8[4] + s8[5] + s8[6] + s8[7];
    if (!(sum > 0.0F)) { return; }
    const __m256 vw = _mm256_set1_ps(w / sum);
    for (std::uint32_t j = 0; j < bi8; j += 8) {
        _mm256_storeu_ps(acc + j, _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j), vw, _mm256_loadu_ps(acc + j)));
    }
}

// out[r][0..16) = sum_d q[r][d] * p[d][0..16) for 6 rows (q row stride D, p packed [D][16]).
// 12 accumulators; the 16 KB p tile stays in L1 while the caller sweeps the row tile.
KVMEM_CPU_AVX2 inline void kvmem_cpu_micro_6x16(const float* q, std::uint32_t dim, const float* p, float* out,
                                    std::size_t ostride) noexcept {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps(),
           c11 = _mm256_setzero_ps(), c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(),
           c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c40 = _mm256_setzero_ps(),
           c41 = _mm256_setzero_ps(), c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    const float* q0 = q;
    const float* q1 = q + dim;
    const float* q2 = q + 2U * dim;
    const float* q3 = q + 3U * dim;
    const float* q4 = q + 4U * dim;
    const float* q5 = q + 5U * dim;
    for (std::uint32_t d = 0; d < dim; ++d) {
        const __m256 b0 = _mm256_loadu_ps(p + static_cast<std::size_t>(d) * 16U);
        const __m256 b1 = _mm256_loadu_ps(p + static_cast<std::size_t>(d) * 16U + 8U);
        __m256 a = _mm256_broadcast_ss(q0 + d);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(q1 + d);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(q2 + d);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(q3 + d);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(q4 + d);
        c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(q5 + d);
        c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }
    _mm256_storeu_ps(out, c00);                _mm256_storeu_ps(out + 8, c01);
    _mm256_storeu_ps(out + ostride, c10);      _mm256_storeu_ps(out + ostride + 8, c11);
    _mm256_storeu_ps(out + 2 * ostride, c20);  _mm256_storeu_ps(out + 2 * ostride + 8, c21);
    _mm256_storeu_ps(out + 3 * ostride, c30);  _mm256_storeu_ps(out + 3 * ostride + 8, c31);
    _mm256_storeu_ps(out + 4 * ostride, c40);  _mm256_storeu_ps(out + 4 * ostride + 8, c41);
    _mm256_storeu_ps(out + 5 * ostride, c50);  _mm256_storeu_ps(out + 5 * ostride + 8, c51);
}

KVMEM_CPU_AVX2 inline float kvmem_cpu_max_avx2(const float* x, std::uint32_t n) noexcept {
    __m256 m = _mm256_set1_ps(-3.0e38F);
    std::uint32_t j = 0;
    for (; j + 8 <= n; j += 8) { m = _mm256_max_ps(m, _mm256_loadu_ps(x + j)); }
    alignas(32) float t[8];
    _mm256_store_ps(t, m);
    float r = std::max({t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]});
    for (; j < n; ++j) { r = std::max(r, x[j]); }
    return r;
}
#endif


} // namespace

bool kvmem_cpu_retrieval_enabled() noexcept {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_CPU_RETRIEVAL");
        if (v != nullptr && v[0] != '\0') { return v[0] != '0'; }
        v = std::getenv("NINFER_TERNARY_KVMEM");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

void kvmem_cpu_scores(const float* sums, std::size_t layer_elements,
                      std::uint32_t layers, std::uint32_t kv_heads,
                      std::uint32_t query_heads, std::uint32_t dim,
                      std::span<const std::int32_t> counts,
                      std::span<const std::uint8_t> exclude,
                      std::span<const KvMemCpuQuery> queries,
                      std::vector<std::vector<float>>& out,
                      bool vectorize, std::uint32_t threads) {
    const auto blocks = static_cast<std::uint32_t>(counts.size());
    out.assign(queries.size(), std::vector<float>(blocks, 0.0F));
    if (blocks == 0 || queries.empty()) { return; }
    if (sums == nullptr || layers == 0 || kv_heads == 0 || query_heads == 0 ||
        dim == 0 || query_heads % kv_heads != 0 ||
        layer_elements < static_cast<std::size_t>(blocks) * kv_heads * dim ||
        (!exclude.empty() && exclude.size() != blocks)) {
        throw std::invalid_argument("KVMem CPU scoring: invalid geometry");
    }
    for (const auto& q : queries) {
        if (q.tokens > q.layer_stride || (q.tokens != 0 && q.rows == nullptr)) {
            throw std::invalid_argument("KVMem CPU scoring: invalid query rows");
        }
    }
    std::vector<std::uint32_t> included;
    included.reserve(blocks);
    for (std::uint32_t b = 0; b < blocks; ++b) {
        if (counts[b] > 0 && (exclude.empty() || exclude[b] == 0)) { included.push_back(b); }
    }
    const auto n = static_cast<std::uint32_t>(included.size());
    if (n == 0) { return; }
    const std::uint32_t nt = (n + 15U) / 16U, padded = nt * 16U;
    const std::uint32_t group = query_heads / kv_heads;
    const std::uint32_t row_elements = query_heads * dim;
    const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
    const float weight = 1.0F / static_cast<float>(layers * query_heads);
    threads = std::min({layers, threads != 0 ? threads : 8U,
                       std::max(1U, std::thread::hardware_concurrency())});
#if KVMEM_CPU_X86
    const bool avx2 = vectorize && kvmem_cpu_has_avx2();
#else
    const bool avx2 = false;
    (void)vectorize;
#endif
    std::vector<std::vector<float>> local(
        threads, std::vector<float>(queries.size() * padded, 0.0F));
    std::vector<std::exception_ptr> errors(threads);
    const auto worker = [&](std::uint32_t tid) {
        try {
            std::vector<float> packed(static_cast<std::size_t>(kv_heads) * nt * dim * 16U);
            std::vector<float> qtile(static_cast<std::size_t>(48) * dim);
            std::vector<float> logits(static_cast<std::size_t>(48) * padded);
            std::vector<float> tmp(padded);
            std::vector<float> logit_scale(padded, 0.0F);
            for (std::uint32_t j = 0; j < n; ++j) {
                logit_scale[j] = scale * (1.0F / static_cast<float>(counts[included[j]]));
            }
            for (std::uint32_t layer = tid; layer < layers; layer += threads) {
                std::fill(packed.begin(), packed.end(), 0.0F);
                const float* layer_sums = sums + static_cast<std::size_t>(layer) * layer_elements;
                for (std::uint32_t j = 0; j < n; ++j) {
                    const float* src = layer_sums + static_cast<std::size_t>(included[j]) * kv_heads * dim;
                    for (std::uint32_t g = 0; g < kv_heads; ++g) {
                        float* dst = packed.data() + ((static_cast<std::size_t>(g) * nt + j / 16U) * dim) * 16U + j % 16U;
                        for (std::uint32_t d = 0; d < dim; ++d) {
                            dst[static_cast<std::size_t>(d) * 16U] = src[static_cast<std::size_t>(g) * dim + d];
                        }
                    }
                }
                for (std::size_t seg = 0; seg < queries.size(); ++seg) {
                    const auto& q = queries[seg];
                    float* acc = local[tid].data() + seg * padded;
                    for (std::uint32_t g = 0; g < kv_heads; ++g) {
                        const std::uint32_t rows = q.tokens * group;
                        const float* pg = packed.data() + static_cast<std::size_t>(g) * nt * dim * 16U;
                        for (std::uint32_t r0 = 0; r0 < rows; r0 += 48U) {
                            const std::uint32_t nr = std::min(48U, rows - r0);
                            const std::uint32_t nr6 = (nr + 5U) / 6U * 6U;
                            for (std::uint32_t r = 0; r < nr6; ++r) {
                                float* dst = qtile.data() + static_cast<std::size_t>(r) * dim;
                                if (r >= nr) { std::fill(dst, dst + dim, 0.0F); continue; }
                                const std::uint32_t tok = (r0 + r) / group;
                                const std::uint32_t h = g * group + (r0 + r) % group;
                                const auto* src = q.rows + (static_cast<std::size_t>(layer) * q.layer_stride + tok) * row_elements + static_cast<std::size_t>(h) * dim;
                                for (std::uint32_t d = 0; d < dim; ++d) { dst[d] = kvmem_cpu_bf16(src[d]); }
                            }
                            for (std::uint32_t tile = 0; tile < nt; ++tile) {
                                const float* pk = pg + static_cast<std::size_t>(tile) * dim * 16U;
                                for (std::uint32_t r = 0; r < nr6; r += 6U) {
                                    float* dst = logits.data() + static_cast<std::size_t>(r) * padded + tile * 16U;
#if KVMEM_CPU_X86
                                    if (avx2) {
                                        kvmem_cpu_micro_6x16(qtile.data() + static_cast<std::size_t>(r) * dim, dim, pk, dst, padded);
                                        continue;
                                    }
#endif
                                    for (std::uint32_t rr = 0; rr < 6U; ++rr) {
                                        for (std::uint32_t j = 0; j < 16U; ++j) {
                                            float dot = 0.0F;
                                            for (std::uint32_t d = 0; d < dim; ++d) {
                                                dot = std::fma(qtile[(static_cast<std::size_t>(r + rr) * dim) + d], pk[static_cast<std::size_t>(d) * 16U + j], dot);
                                            }
                                            dst[static_cast<std::size_t>(rr) * padded + j] = dot;
                                        }
                                    }
                                }
                            }
                            for (std::uint32_t r = 0; r < nr; ++r) {
                                float* lg = logits.data() + static_cast<std::size_t>(r) * padded;
                                for (std::uint32_t j = 0; j < n; ++j) { lg[j] *= logit_scale[j]; }
                                std::fill(lg + n, lg + padded, -INFINITY);
#if KVMEM_CPU_X86
                                if (avx2) {
                                    kvmem_cpu_softmax_acc_avx2(lg, n, padded, weight, tmp.data(), acc);
                                    continue;
                                }
#endif
                                const float mx = *std::max_element(lg, lg + n);
                                double mass = 0.0;
                                for (std::uint32_t j = 0; j < n; ++j) { tmp[j] = std::exp(lg[j] - mx); mass += tmp[j]; }
                                if (!(mass > 0.0)) { continue; }
                                const float w = static_cast<float>(weight / mass);
                                for (std::uint32_t j = 0; j < n; ++j) { acc[j] += tmp[j] * w; }
                            }
                        }
                    }
                }
            }
        } catch (...) { errors[tid] = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    try {
        for (std::uint32_t t = 1; t < threads; ++t) { pool.emplace_back(worker, t); }
        worker(0);
    } catch (...) {
        for (auto& t : pool) { t.join(); }
        throw;
    }
    for (auto& t : pool) { t.join(); }
    for (const auto& error : errors) { if (error) { std::rethrow_exception(error); } }
    for (std::size_t seg = 0; seg < queries.size(); ++seg) {
        for (std::uint32_t t = 0; t < threads; ++t) {
            for (std::uint32_t j = 0; j < n; ++j) { out[seg][included[j]] += local[t][seg * padded + j]; }
        }
    }
}
} // namespace ninfer::ops::detail
