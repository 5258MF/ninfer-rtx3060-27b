#pragma once

// KVMem kv9 port: the official KVMem Eq.10 query-conditioned block scorer, on the host.
// (old engine: ops/kvmem/kvmem_retrieve.h, reference kernels_cuda.cu block_attn_score_softmax_pages)
//
//   For query token m, full-attention layer l, QUERY head h and candidate block b:
//     logit[l,m,h,b] = (q[l,m,h] . kbar[l,b,h/group]) / sqrt(head_dim)
//     mass [l,m,h,b] = softmax OVER THE INCLUDED BLOCKS of logit[l,m,h,.]
//     score[b]       = sum_m mean_l mean_h mass[l,m,h,b]
//   Blocks that are kept anyway (sink, protected new content, recent tail) and blocks without a
//   complete mean are excluded from the softmax, so their mass redistributes onto the middle.
//
// The qz scorer (cosine of the token-averaged query against kbar) averages a needle token's
// signal away; Eq.10 lets every query token pick its own blocks. q rows are the captured
// normalized pre-RoPE BF16 query rows ([layer][row][n_heads * head_dim]).

#include "models/qwen3_5/program/retrieval/block_retrieval.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define KV9_X86 1
#endif

namespace ninfer::models::qwen3_5::detail {

inline bool kv9_env_tiled() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("KV9_EQ10_TILED");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

struct Kv9RowView {
    const std::uint16_t* rows = nullptr;  // bf16; row r of layer l at rows + (l * layer_stride + r) * row_elems
    std::uint32_t n = 0;                  // query tokens
    std::size_t layer_stride = 0;         // rows per layer in this buffer
};

inline float kv9_bf16(std::uint16_t v) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

#if KV9_X86
inline bool kv9_has_avx2() noexcept {
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
#define KV9_AVX2 __attribute__((target("avx2,fma")))
#else
#define KV9_AVX2
#endif

// exp for x <= 0 (softmax after max subtraction); Cephes-style, rel. error ~1e-7, clamps < -87.
KV9_AVX2 inline __m256 kv9_exp256(__m256 x) noexcept {
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
KV9_AVX2 inline void kv9_logits_avx2(const float* q, std::uint32_t nh, std::uint32_t dim, const float* kT,
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

KV9_AVX2 inline float kv9_max_avx2(const float* x, std::uint32_t n) noexcept;

// In-place softmax mass over [0, bi) (padding up to bi8 ignored), accumulated as acc[j] += w * p[j].
KV9_AVX2 inline void kv9_softmax_acc_avx2(const float* logits, std::uint32_t bi, std::uint32_t bi8,
                                          float w, float* tmp, float* acc) noexcept {
    const float mx = kv9_max_avx2(logits, bi);
    const __m256 vmx = _mm256_set1_ps(mx);
    __m256 vs = _mm256_setzero_ps();
    for (std::uint32_t j = 0; j < bi8; j += 8) {
        __m256 e = kv9_exp256(_mm256_sub_ps(_mm256_loadu_ps(logits + j), vmx));
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
KV9_AVX2 inline void kv9_micro_6x16(const float* q, std::uint32_t dim, const float* p, float* out,
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

KV9_AVX2 inline float kv9_max_avx2(const float* x, std::uint32_t n) noexcept {
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

// out[s][b] = Eq.10 score of block b for segment s (0 for excluded blocks).
inline void kv9_eq10_scores(const RetrievalIndex& index, std::span<const Kv9RowView> segs,
                            std::uint32_t q_heads, std::span<const std::uint8_t> include,
                            std::vector<std::vector<float>>& out) {
    const std::uint32_t L = index.layers(), KV = index.kv_heads(), D = index.head_dim();
    const std::uint32_t blocks = index.block_count();
    out.assign(segs.size(), std::vector<float>(blocks, 0.0F));
    if (L == 0 || KV == 0 || D == 0 || q_heads == 0 || q_heads % KV != 0 || segs.empty()) { return; }
    const std::uint32_t group = q_heads / KV;
    const std::uint32_t QW = q_heads * D;
    std::vector<std::uint32_t> inc;
    inc.reserve(blocks);
    for (std::uint32_t b = 0; b < blocks && b < include.size(); ++b) {
        if (!include[b] || !index.block(b).full) { continue; }
        bool ok = true;
        for (std::uint32_t l = 0; l < L && ok; ++l) { ok = index.block_mean(b, l) != nullptr; }
        if (ok) { inc.push_back(b); }
    }
    const auto bi = static_cast<std::uint32_t>(inc.size());
    if (bi == 0) { return; }
    const std::uint32_t bi8 = (bi + 7U) / 8U * 8U;
    const float scale = 1.0F / std::sqrt(static_cast<float>(D));
    const float w = 1.0F / static_cast<float>(L * q_heads);
    const auto nseg = static_cast<std::uint32_t>(segs.size());
#if KV9_X86
#if defined(KV9_NO_AVX)
    const bool avx2 = false;
#else
    const bool avx2 = kv9_has_avx2();
#endif
#else
    const bool avx2 = false;
#endif
    const std::uint32_t hc = std::max(1U, std::thread::hardware_concurrency());
    const std::uint32_t nthreads = std::min({L, 8U, hc});
    std::vector<std::vector<float>> local(nthreads, std::vector<float>(static_cast<std::size_t>(nseg) * bi8, 0.0F));
#if KV9_X86
    // Tiled GEMM path: per layer, kbar is packed per KV head into 16-block tiles [tile][d][16];
    // a 48-row tile of (token, q-head) rows is multiplied against every block tile (6x16 micro
    // kernel), then each row's softmax mass is accumulated. kbar is read from memory once per
    // layer instead of once per query token.
    const std::uint32_t nt = (bi + 15U) / 16U, bi16 = nt * 16U;
    constexpr std::uint32_t kRowTile = 48;
    const auto work_tiled = [&](std::uint32_t t) {
        std::vector<float> pk(static_cast<std::size_t>(KV) * nt * D * 16U);
        std::vector<float> qt(static_cast<std::size_t>(kRowTile) * D);
        std::vector<float> logits(static_cast<std::size_t>(kRowTile) * bi16);
        std::vector<float> tmp(bi16);
        float* acc_all = local[t].data();
        for (std::uint32_t l = t; l < L; l += nthreads) {
            std::fill(pk.begin(), pk.end(), 0.0F);
            for (std::uint32_t j = 0; j < bi; ++j) {
                const float* m = index.block_mean(inc[j], l);
                const std::uint32_t tile = j / 16U, lane = j % 16U;
                for (std::uint32_t g = 0; g < KV; ++g) {
                    float* dst = pk.data() + ((static_cast<std::size_t>(g) * nt + tile) * D) * 16U + lane;
                    const float* src = m + static_cast<std::size_t>(g) * D;
                    for (std::uint32_t d = 0; d < D; ++d) { dst[static_cast<std::size_t>(d) * 16U] = src[d]; }
                }
            }
            for (std::uint32_t s = 0; s < nseg; ++s) {
                const Kv9RowView& v = segs[s];
                float* acc = acc_all + static_cast<std::size_t>(s) * bi8;
                for (std::uint32_t g = 0; g < KV; ++g) {
                    const std::uint32_t rows = v.n * group;
                    const float* pg = pk.data() + static_cast<std::size_t>(g) * nt * D * 16U;
                    for (std::uint32_t r0 = 0; r0 < rows; r0 += kRowTile) {
                        const std::uint32_t nr = std::min(kRowTile, rows - r0);
                        const std::uint32_t nr6 = (nr + 5U) / 6U * 6U;
                        for (std::uint32_t r = 0; r < nr6; ++r) {
                            float* dst = qt.data() + static_cast<std::size_t>(r) * D;
                            if (r >= nr) { std::fill(dst, dst + D, 0.0F); continue; }
                            const std::uint32_t tok = (r0 + r) / group, h = g * group + (r0 + r) % group;
                            const std::uint16_t* src = v.rows + (static_cast<std::size_t>(l) * v.layer_stride + tok) * QW +
                                                       static_cast<std::size_t>(h) * D;
                            for (std::uint32_t d = 0; d < D; ++d) { dst[d] = kv9_bf16(src[d]) * scale; }
                        }
                        for (std::uint32_t tile = 0; tile < nt; ++tile) {
                            const float* ptile = pg + static_cast<std::size_t>(tile) * D * 16U;
                            for (std::uint32_t rg = 0; rg < nr6; rg += 6U) {
                                kv9_micro_6x16(qt.data() + static_cast<std::size_t>(rg) * D, D, ptile,
                                               logits.data() + static_cast<std::size_t>(rg) * bi16 + tile * 16U, bi16);
                            }
                        }
                        for (std::uint32_t r = 0; r < nr; ++r) {
                            kv9_softmax_acc_avx2(logits.data() + static_cast<std::size_t>(r) * bi16, bi, bi8, w,
                                                 tmp.data(), acc);
                        }
                    }
                }
            }
        }
    };
#endif
    const auto work = [&](std::uint32_t t) {
        std::vector<float> kT(static_cast<std::size_t>(KV) * D * bi8, 0.0F);
        std::vector<float> qf(static_cast<std::size_t>(group) * D);
        std::vector<float> logits(static_cast<std::size_t>(group) * bi8);
        std::vector<float> tmp(bi8);
        float* acc_all = local[t].data();
        for (std::uint32_t l = t; l < L; l += nthreads) {
            for (std::uint32_t j = 0; j < bi; ++j) {
                const float* m = index.block_mean(inc[j], l);
                for (std::uint32_t g = 0; g < KV; ++g) {
                    for (std::uint32_t d = 0; d < D; ++d) {
                        kT[(static_cast<std::size_t>(g) * D + d) * bi8 + j] = m[static_cast<std::size_t>(g) * D + d];
                    }
                }
            }
            for (std::uint32_t s = 0; s < nseg; ++s) {
                const Kv9RowView& v = segs[s];
                float* acc = acc_all + static_cast<std::size_t>(s) * bi8;
                for (std::uint32_t r = 0; r < v.n; ++r) {
                    const std::uint16_t* row = v.rows + (static_cast<std::size_t>(l) * v.layer_stride + r) * QW;
                    for (std::uint32_t g = 0; g < KV; ++g) {
                        for (std::uint32_t h = 0; h < group; ++h) {
                            const std::uint16_t* qh = row + static_cast<std::size_t>(g * group + h) * D;
                            for (std::uint32_t d = 0; d < D; ++d) { qf[static_cast<std::size_t>(h) * D + d] = kv9_bf16(qh[d]) * scale; }
                        }
                        const float* kg = kT.data() + static_cast<std::size_t>(g) * D * bi8;
#if KV9_X86
                        if (avx2) {
                            for (std::uint32_t h0 = 0; h0 < group; h0 += 6) {
                                const std::uint32_t nh = std::min(6U, group - h0);
                                kv9_logits_avx2(qf.data() + static_cast<std::size_t>(h0) * D, nh, D, kg, bi8,
                                                logits.data() + static_cast<std::size_t>(h0) * bi8);
                            }
                            for (std::uint32_t h = 0; h < group; ++h) {
                                kv9_softmax_acc_avx2(logits.data() + static_cast<std::size_t>(h) * bi8, bi, bi8, w,
                                                     tmp.data(), acc);
                            }
                            continue;
                        }
#endif
                        for (std::uint32_t h = 0; h < group; ++h) {
                            float* lg = logits.data() + static_cast<std::size_t>(h) * bi8;
                            std::fill(lg, lg + bi, 0.0F);
                            for (std::uint32_t d = 0; d < D; ++d) {
                                const float qd = qf[static_cast<std::size_t>(h) * D + d];
                                const float* kr = kg + static_cast<std::size_t>(d) * bi8;
                                for (std::uint32_t j = 0; j < bi; ++j) { lg[j] += qd * kr[j]; }
                            }
                            float mx = -3.0e38F;
                            for (std::uint32_t j = 0; j < bi; ++j) { mx = std::max(mx, lg[j]); }
                            double sum = 0.0;
                            for (std::uint32_t j = 0; j < bi; ++j) { tmp[j] = std::exp(lg[j] - mx); sum += tmp[j]; }
                            if (!(sum > 0.0)) { continue; }
                            const float k = static_cast<float>(w / sum);
                            for (std::uint32_t j = 0; j < bi; ++j) { acc[j] += tmp[j] * k; }
                        }
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
#if KV9_X86
    if (avx2 && kv9_env_tiled()) {
        for (std::uint32_t t = 1; t < nthreads; ++t) { pool.emplace_back(work_tiled, t); }
        work_tiled(0);
    } else
#endif
    {
        for (std::uint32_t t = 1; t < nthreads; ++t) { pool.emplace_back(work, t); }
        work(0);
    }
    for (auto& th : pool) { th.join(); }
    for (std::uint32_t s = 0; s < nseg; ++s) {
        for (std::uint32_t t = 0; t < nthreads; ++t) {
            const float* src = local[t].data() + static_cast<std::size_t>(s) * bi8;
            for (std::uint32_t j = 0; j < bi; ++j) { out[s][inc[j]] += src[j]; }
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
