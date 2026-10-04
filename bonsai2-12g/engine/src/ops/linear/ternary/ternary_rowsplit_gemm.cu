#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_s8.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_s8x.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_wide_t.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

// Decode (T == 1) takes the warp-per-row GEMV for PQ2_0. K is a whole number of 128-groups for
// every width in this model, so that kernel needs no column guard.
void launch_pq2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((w.k % 128) != 0 || x.ne[1] != 1) {
        throw std::invalid_argument("ternary gemv: expected one token and a whole-group K");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    ternary_pq2_gemv_kernel<<<grid, kGemvWarpsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        groups_per_row);
    CUDA_CHECK(cudaGetLastError());
}

// Small-token-tile GEMV: weights are read once for up to 4 tokens, which is what makes the
// speculative verify pass (T = draft + 1) cheap. Falls back to the reference tiled kernel beyond
// that, and for PTQ1_0 / padded-K weights.
void launch_pq2_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, std::int32_t tokens,
                          cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_pq2_gemv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row);
    } else {
        ternary_pq2_gemv_tile_kernel<4><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row, tokens, out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

// PTQ1_0 twin of the GEMV above: same grid/block policy and the same token-tile size, but the weight
// fetch needs the high plane (w.qhigh) and decodes base-3 trits per column instead of 2-bit slots.
void launch_ptq1_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                           std::int32_t out_row_stride, std::int32_t tokens,
                           cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary gemv: PTQ1_0 needs its high plane");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_ptq1_gemv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row);
    } else {
        ternary_ptq1_gemv_tile_kernel<4><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row, tokens, out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

// One call site for both ternary formats. The kernels key on w.qhigh (PQ2_0 has no high plane), so
// callers stay format-agnostic beyond this dispatch -- which is the whole point of the change.
void launch_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                      std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_gemv_tile(x, w, out, out_row_stride, tokens, stream);
        return;
    }
    launch_pq2_gemv_tile(x, w, out, out_row_stride, tokens, stream);
}

template <class Storage, class Atom, int kTileT>
void launch_gemm(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                 cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const std::int32_t k    = w.k;
    const std::int32_t t    = x.ne[1];
    if (k % Storage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary linear: output row stride is smaller than the tile");
    }
    const std::int32_t groups_per_row = k / Storage::kGroupK;

    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(div_up(t, kTileT)), 1u);
    constexpr dim3 block(Storage::kGroupK, 1u, 1u);

    ternary_rowsplit_gemm_kernel<Storage, Atom, kTileT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, t, groups_per_row, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <int kTileT>
void launch_by_qtype(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                     cudaStream_t stream) {
    switch (w.qtype) {
    case QType::PTQ1_0_G128:
        launch_gemm<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                    stream);
        return;
    case QType::PQ2_0_G128:
        launch_gemm<PQ2RowSplitStorage, PQ2SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                  stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("ternary linear: unsupported weight qtype");
}

} // namespace

// NINFER_TERNARY_PTQ1_FAST=0 puts PTQ1_0 back on the reference path -- the A/B arm, same pattern as
// the mma/s8 switches. Read once: the engine treats env as process-level (ternary_s8_scratch.h).
bool ternary_ptq1_fast_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_TERNARY_PTQ1_FAST");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// A PQ2_0 **or PTQ1_0** weight can take the GEMV family whenever the layout did not pad K past the
// real width -- true for every width in this model (5120/6144/10240/17408 are all whole 128-groups),
// and checked here rather than assumed, because the GEMV reads whole groups without a column guard.
// The two formats are told apart by the high plane: row_split_geometry gives PQ2_0 zero high bytes
// (qhigh == nullptr) and PTQ1_0 two (qhigh != nullptr), which is also what the kernels key on.
bool gemv_admits(const Tensor& x, const Weight& w, std::int32_t max_tokens) {
    const bool pq2  = w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr;
    const bool ptq1 = w.qtype == QType::PTQ1_0_G128 && w.qhigh != nullptr &&
                      ternary_ptq1_fast_enabled();
    return (pq2 || ptq1) && w.padded_shape[1] == w.k && (w.k % 128) == 0 && x.ne[1] >= 1 &&
           x.ne[1] <= max_tokens;
}

void launch_pq2_mma_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                           std::int32_t out_row_stride, std::int32_t tokens,
                           cudaStream_t stream) {
    // Static shared memory again (42.25 KB at BN=64), so no dynamic size and no opt-in.
    const bool token_grid = ternary_token_grid_enabled();
    const dim3 grid(static_cast<unsigned>(div_up(w.n, kTernaryWideRowsPerCta)),
                    token_grid ? static_cast<unsigned>(div_up(tokens, kTernaryWideTokens)) : 1u,
                    1u);
    ternary_wide_t_kernel<false><<<grid, kTernaryWideThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        nullptr, static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), w.n, w.k, tokens, out_row_stride, token_grid);
    CUDA_CHECK(cudaGetLastError());
}

// PTQ1_0 twin: same kernel, same grid, same shared budget -- the only difference is that its staging
// step repacks the 24+2-byte rows into PQ2 code bytes (see the kernel's kPtq1 note).
void launch_ptq1_mma_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, std::int32_t tokens,
                            cudaStream_t stream) {
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary wide-t mma: PTQ1_0 needs its high plane");
    }
    const bool token_grid = ternary_token_grid_enabled();
    const dim3 grid(static_cast<unsigned>(div_up(w.n, kTernaryWideRowsPerCta)),
                    token_grid ? static_cast<unsigned>(div_up(tokens, kTernaryWideTokens)) : 1u,
                    1u);
    ternary_wide_t_kernel<true><<<grid, kTernaryWideThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), w.n, w.k, tokens, out_row_stride, token_grid);
    CUDA_CHECK(cudaGetLastError());
}

// One call site for both ternary formats, exactly like the GEMV dispatcher.
void launch_wide_t(const Tensor& x, const Weight& w, Tensor& out,
                   std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_mma_wide_t(x, w, out, out_row_stride, tokens, stream);
        return;
    }
    launch_pq2_mma_wide_t(x, w, out, out_row_stride, tokens, stream);
}

// Admission for every rung that READS PTQ1_0 and repacks it into the PQ2_0 code layout in shared
// (wide_t, small_t, s8). All three stage the raw 24-byte base rows with 8-byte cp.async copies and
// move a whole chunk's 2 high bytes per row as one more 8-byte copy, so the high plane's BASE has to
// be 8-byte aligned -- the row pitch is aligned by construction, because K is a whole number of
// 512-wide chunks (groups_per_row a multiple of 4). The check is explicit rather than assumed: an
// artifact that violates it falls back to a correct-but-slower rung instead of mis-decoding.
bool ptq1_repack_admits(const Weight& w) {
    return w.qtype == QType::PTQ1_0_G128 && w.qhigh != nullptr && ternary_ptq1_fast_enabled() &&
           (reinterpret_cast<std::uintptr_t>(w.qhigh) & 7u) == 0u;
}

// The wide-token-tile kernel only needs K to be a whole number of 128-wide chunks (one PQ2_0
// group), which is strictly weaker than the small-t path's 512. PTQ1_0 is admitted too: its staging
// step repacks the 24+2-byte rows into the PQ2 code layout in shared (verified bit-exact against
// real weights), so everything downstream of the staging is the same code.
bool mma_wide_admits(const Weight& w, std::int32_t out_row_stride) {
    const bool pq2  = w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr;
    const bool ptq1 = ptq1_repack_admits(w);
    return (pq2 || ptq1) && w.padded_shape[1] == w.k && (w.k % kTernaryWideChunkK) == 0 &&
           out_row_stride >= w.n;
}


// Decode (T == 1): the tensor-core small-T kernel at one staged token row for PQ2_0 (unless
// NINFER_TERNARY_T1_MMA=0), the warp-per-row GEMV otherwise.
//
// sm_86 (RTX 3060, 28 SMs, 2026-09-27) -- WHY THE ROUTING BELOW DIFFERS FROM THE ADA RECORD THAT
// FOLLOWS. Measured DRAM-cold with a sustained clock warm-up on the artifact's projection shapes:
// the GEMV sustains 120-129 GiB/s against a 296 GiB/s streaming-read roof (a no-math probe with the
// GEMV's exact access pattern reaches 285-300, so the gap is SM-side, ~30 instructions per code
// byte), while the register-decode small-T kernel at kActRows == 1 reaches 198-231 GiB/s on the
// N >= 5120 shapes (21.5 KB shared, four CTAs per SM -- not the 38 KB / two CTAs the Ada note below
// is about). This card has ~13 GB/s of DRAM per SM against ~9 on the Ada part, so the SIMT
// GEMV's instruction bill is what binds here. The end-to-end decision is taken on the engine, not
// on this microbenchmark (see the Ada record for why that matters): ninfer_bench decode A/B with
// NINFER_TERNARY_T1_MMA=0/1.
//
// ADA RECORD (the alternative was measured and rejected there) -- routing T = 1 through the
// tensor-core kernel. The synthetic harness (E:/infer-build/tscale_bench.cu) ranks mma at
// 14.56 ms/forward against 19.44 ms for this GEMV at T=1 over the artifact's real shape mix (+34%),
// which is the origin of the "+17% at T=1" claim on the small-T path below. On the actual engine
// the sign flips. A 400-token greedy decode A/B with the arms alternating three times measures
// 58.9 t/s with the tensor core against 60.8 t/s with this GEMV (58.3-59.5 against 60.2-61.2, no
// overlap), and a node-level nsys profile of both arms shows 81402 ternary GEMM calls in each --
// so the branch was genuinely taken, the mma kernel just costs ~7% more per call at T=1
// (36.6 us against 34.1 us).
//
// WHY THE HARNESS DISAGREES, MEASURED (E:/infer-build/t1_flush_probe.cu)
// Not L2 residency, and not the flush failing -- a 256 MB memset does evict this part's 64 MB L2
// (an 8 MiB read goes 7.4 us L2-hot to 13.3 us after a memset, against an 11.4 us DRAM floor).
// The harness's T=1 numbers are dominated by its own subtraction: it reports
//   (mean[256 MB memset + kernel]) - flush_constant
// where the constant is ~419 us and its own repeatability across a run spans 415-445 us
// (7.0%). That 29 us band is 1.2-3.2x the kernels it is used to isolate, and the same binary run
// three times reports this shape's T=1 mma as 901.5 / 292.7 / 357.8 GiB/s. A negative control sits
// inside one run: gemv_tile<kT=4> and gemv_tile<kT=8> are the SAME kernel at T=1 (both fall
// through to ternary_pq2_gemv_kernel) and were reported 344.0 and 487.0 GiB/s. The 901 GiB/s
// reading is also below that shape's 11.35 us DRAM floor, so it was never a bandwidth at all.
// Measured without any flush or subtraction (rotating a 128 MiB working set so every launch is
// DRAM-cold), the mma IS the faster kernel in isolation (~440-560 GiB/s against ~330-460 for this
// GEMV), so what the engine loses is not the kernel's inner loop but its interaction with the
// other ~600 launches of a token -- 38144 B of static shared memory puts it at 2 CTAs/SM where
// this GEMV has no shared memory and no barrier at all.
//
// WHAT THE TOKEN ACTUALLY LOOKS LIKE at T=1 (nsys, 200-token greedy run, 403 ternary GEMM calls
// per step): 13.75 ms of GEMV inside a 16.2 ms token = 85%, i.e. 6.70 GiB / 13.75 ms = 487 GiB/s
// = 71% of this card's 685.5 GiB/s spec peak, and 82% of the 595 GiB/s it actually delivers on a
// plain streaming read. The other ~2.4 ms is ~258 rotation launches (~1.1 ms) plus norms, GDN and
// attention. Repro pack: E:/infer-build/run-t1-ab-long.cmd, run-nsys-t1.cmd, t1_flush_probe.cu.
//
// COMMUNITY CROSS-CHECK (2026-09-20) -- this is not a local anomaly, so do not re-litigate it:
//   * The one other ninfer fork that routed a T=1 projection to mma (UDPSendToFailed/ninfer-4090,
//     commit 39a6f20, "route T=1 draft head projections to double-buffered Ada MMA") measured
//     +3.0% on real-world decode but 0.0% on PURE T=1 auto-regression, and its win is credited to
//     grid/wave geometry (32768 blocks = 256 waves -> 8192 blocks), not to the tensor core.
//   * Upstream PR #292 moves T=1 onto a K-split mma rung only on two N=5120 shapes, "where it beats
//     the dedicated GEMV by 8-11%. N=6144 and N=7168 keep that GEMV, which is still 7-12% faster
//     for them", and reports that ordinary (non-speculative) generation "moves about a percent,
//     which is close to this machine's run-to-run noise".
//   * llama.cpp keeps SIMT matvec up to batch 8 on Ada for the same reason this kernel loses here:
//     it says the deciding factor is that dequantization cost cannot be amortised over one column
//     ("k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner"), and
//     Marlin's own framing is that batch < 64 is memory-bound, so a tensor core at batch 1 is at
//     best "not a loss". Expect 0-3% here, per shape -- never the +17% a micro-benchmark suggests.
bool mma_admits(const Weight& w, std::int32_t out_row_stride);
void launch_small_t(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                    std::int32_t tokens, cudaStream_t stream);

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch /*scratch*/) {
    if (w.qtype == QType::PQ2_0_G128 && x.ne[1] == 1 && ternary_mma_enabled() &&
        ternary_t1_mma_enabled() && mma_admits(w, out_row_stride)) {
        launch_small_t(x, w, out, out_row_stride, 1, stream);
        return;
    }
    if (gemv_admits(x, w, 1)) {
        launch_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    launch_by_qtype<1>(x, w, out, out_row_stride, stream);
}

// Tensor-core path for the small-T regime (T = 2..8) -- the speculative verify pass.
//
// The SIMT tile GEMV reuses each weight across the tile but still pays one FMA per weight PER
// TOKEN, so its cost grows with T while its weight traffic does not: measured on a 322 MB PQ2_0
// payload it takes 0.706 ms at T=1 and 1.533 ms at T=3, i.e. 2.2x the time for byte-identical
// traffic (211 GiB/s against 483 GiB/s). The tensor core removes that per-token bill entirely --
// one mma.m16n8k16 carries 16 rows x 8 tokens x 16 k -- so the whole draft+verify tile rides one
// weight read: measured 0.611 / 0.633 ms at T = 3 / 8 against 1.493 / 1.533 ms for the tile GEMV.
// (The T=1 endpoint of that same measurement, 0.603 against 0.706 ms, did NOT transfer to the real
// engine on Ada; on sm_86 PQ2_0 T=1 is routed here -- see the note on launch_ternary_gemm_t1.)
//
// BOTH formats are admitted. The PTQ1_0 instantiation cp.async's the raw 24+2-byte row into the
// same 144-byte weight rows and repacks it into the PQ2 code layout before the mma (the same
// repack and the same code_of/stage_codes arithmetic the wide-t rung uses, validated bit-exact
// against the real artifact), so everything downstream of the staging is the same code. Its only
// extra layout requirement is that the high plane be 8-byte aligned, and it is checked HERE rather
// than assumed: a failure falls back to the tile GEMV, which is correct and merely slower (see the
// note at the condition), never a silent mis-decode.
bool mma_admits(const Weight& w, std::int32_t out_row_stride) {
    const bool pq2 = w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr;
    const bool ptq1 = w.qtype == QType::PTQ1_0_G128 && w.qhigh != nullptr &&
                      ternary_ptq1_fast_enabled() &&
                      // The staging moves a whole chunk's 4 x 2 high bytes per row as ONE 8-byte
                      // cp.async. The row pitch is 8-byte aligned by construction (groups_per_row
                      // is a multiple of 4 whenever K is a whole number of 512-wide chunks), so
                      // only the plane base is not implied by the geometry.
                      (reinterpret_cast<std::uintptr_t>(w.qhigh) & 7u) == 0u;
    return (pq2 || ptq1) && w.padded_shape[1] == w.k &&
           (w.k % kTernaryMmaChunkK) == 0 && out_row_stride >= w.n;
}

// One launch per call; the instantiation is picked by how many token rows the CTA has to stage
// (min(tokens, 8) -- see TernaryMmaStorage), which is what sets the shared footprint and so the
// resident CTAs per SM. Tokens above 8 walk tiles of 8 inside the kernel.
template <bool kPtq1>
void launch_small_t_rows(const Tensor& x, const Weight& w, const std::uint8_t* high, Tensor& out,
                         std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    // Token tiles above 8 go to grid.y unless NINFER_TERNARY_TOKEN_GRID=0 (see the kernel).
    const bool token_grid = ternary_token_grid_enabled() && tokens > kTernaryMmaTokens;
    const dim3 grid(static_cast<unsigned>(div_up(w.n, kTernaryMmaRowsPerCta)),
                    token_grid ? static_cast<unsigned>(div_up(tokens, kTernaryMmaTokens)) : 1u, 1u);
    const auto launch = [&](auto rows_tag) {
        constexpr int kRows = decltype(rows_tag)::value;
        ternary_mma_small_t_kernel<kPtq1, kRows><<<grid, kTernaryMmaThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            high, static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, w.k, tokens, out_row_stride, token_grid);
    };
    switch (tokens < kTernaryMmaTokens ? tokens : kTernaryMmaTokens) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 5: launch(std::integral_constant<int, 5>{}); break;
    case 6: launch(std::integral_constant<int, 6>{}); break;
    case 7: launch(std::integral_constant<int, 7>{}); break;
    default: launch(std::integral_constant<int, 8>{}); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_pq2_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, std::int32_t tokens,
                            cudaStream_t stream) {
    // The staging is static shared memory (a compile-time 32 KB), so there is no dynamic size to
    // pass and no cudaFuncSetAttribute opt-in to make.
    launch_small_t_rows<false>(x, w, nullptr, out, out_row_stride, tokens, stream);
}

// PTQ1_0 twin: same kernel, same grid, same everything below the repack. Its static shared is
// 46.25 KB instead of 32 KB (three rotating weight-row buffers instead of two -- see
// kTernaryMmaPtq1WeightBufs), still under the 48 KB static cap and still two CTAs per SM.
void launch_ptq1_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                             std::int32_t out_row_stride, std::int32_t tokens,
                             cudaStream_t stream) {
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary small-t mma: PTQ1_0 needs its high plane");
    }
    launch_small_t_rows<true>(x, w, static_cast<const std::uint8_t*>(w.qhigh), out,
                              out_row_stride, tokens, stream);
}

// One call site for both ternary formats, exactly like the GEMV and wide-t dispatchers.
void launch_small_t(const Tensor& x, const Weight& w, Tensor& out,
                    std::int32_t out_row_stride, std::int32_t tokens, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_mma_small_t(x, w, out, out_row_stride, tokens, stream);
        return;
    }
    launch_pq2_mma_small_t(x, w, out, out_row_stride, tokens, stream);
}

// int8 rung (#14): quantize the activation row per token, then run the s8 tensor-core kernel.
// Requires a caller-provided scratch (see TernaryS8Scratch): the quantization pass needs one int8
// code row per token plus one fp32 scale per token.
void launch_pq2_mma_s8(const Tensor& x, const Weight& w, Tensor& out,
                       std::int32_t out_row_stride, std::int32_t tokens, TernaryS8Scratch scratch,
                       cudaStream_t stream) {
    launch_ternary_s8_quantize(static_cast<const __nv_bfloat16*>(x.data), scratch.codes,
                               scratch.scales, tokens, w.k, stream);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kTokens = 64;
    constexpr int kWarps  = 4;
    const bool token_grid = ternary_token_grid_enabled();
    if (token_grid && ternary_s8x_enabled()) {
        launch_pq2_mma_s8x(scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
                           static_cast<const std::uint8_t*>(w.scales),
                           static_cast<__nv_bfloat16*>(out.data), w.n, w.k, tokens, out_row_stride,
                           stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(
        static_cast<unsigned>(div_up(w.n, TernaryS8Storage<kTokens, kWarps>::kRowsPerCta)),
        token_grid ? static_cast<unsigned>(div_up(tokens, kTokens)) : 1u, 1u);
    ternary_pq2_mma_s8_kernel<kTokens, kWarps, 3><<<grid, kWarps * 32, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride, nullptr, token_grid);
    CUDA_CHECK(cudaGetLastError());
}

// PTQ1_0 twin of the int8 rung (2026-09-23). Same activation quantization pass, same mma, same
// staging geometry -- the only difference is that the weight staging cp.async's the raw 24+2-byte
// rows and repacks them into the PQ2_0 code layout in shared (the shared helper in
// ternary_ptq1_repack.cuh, also used by wide_t and small_t). Kernel-level accuracy on real shapes:
// rel_l2 ~9.0e-3, which IS the int8 activation quantization error itself (see the s8 header note),
// i.e. the same accuracy class as the PQ2_0 s8 rung -- this rung is not bit-exact by construction,
// and its gate is the E4 criterion the PQ2_0 rung already passed.
void launch_ptq1_mma_s8(const Tensor& x, const Weight& w, Tensor& out,
                        std::int32_t out_row_stride, std::int32_t tokens, TernaryS8Scratch scratch,
                        cudaStream_t stream) {
    if (w.qhigh == nullptr) {
        throw std::invalid_argument("ternary s8 mma: PTQ1_0 needs its high plane");
    }
    launch_ternary_s8_quantize(static_cast<const __nv_bfloat16*>(x.data), scratch.codes,
                               scratch.scales, tokens, w.k, stream);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kTokens = 64;
    constexpr int kWarps  = 4;
    // Same rows-per-CTA as the PQ2_0 instantiation (it is kTernaryS8RowsPerWarp * kWarps in both),
    // named through the PTQ1 storage so the two can never drift apart silently.
    const bool token_grid = ternary_token_grid_enabled();
    const dim3 grid(
        static_cast<unsigned>(div_up(w.n, TernaryS8Storage<kTokens, kWarps, true>::kRowsPerCta)),
        token_grid ? static_cast<unsigned>(div_up(tokens, kTokens)) : 1u, 1u);
    ternary_pq2_mma_s8_kernel<kTokens, kWarps, 3, true><<<grid, kWarps * 32, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride, static_cast<const std::uint8_t*>(w.qhigh), token_grid);
    CUDA_CHECK(cudaGetLastError());
}

// One call site for both ternary formats, like the GEMV / wide-t / small-t dispatchers.
void launch_s8(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
               std::int32_t tokens, TernaryS8Scratch scratch, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        launch_ptq1_mma_s8(x, w, out, out_row_stride, tokens, scratch, stream);
        return;
    }
    launch_pq2_mma_s8(x, w, out, out_row_stride, tokens, scratch, stream);
}

void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch) {
    // The speculative verify pass runs T = draft + 1 (2..4 here). The reference tiled kernel wastes
    // five of its eight token slots at that size and needs a 128-thread CTA plus seven barriers per
    // output row, which cost more than the whole decode step it was verifying. The tensor-core path
    // is preferred because its cost is flat in T; the small-tile GEMV is the fallback for shapes it
    // cannot take (PTQ1_0, padded K, or K not a whole number of 256-wide chunks).
    //
    // The tensor-core kernel walks the sequence in 8-token tiles, so it now covers PREFILL too, not
    // just the verify pass. Above 8 tokens the old code fell through to the reference kernel, which
    // is where prefill's 55 tok/s (against 412-437 t/s for the non-ternary engine) came from.
    //
    // But 8 tokens per weight pass is also the reason prefill is 3-4x behind the official runtime on
    // this card: this kernel restages the weight window for EVERY tile, so one forward reads the
    // weights ceil(T/8) times (measured: T=62 -> 131 ms = 8 passes x 6.70 GiB at 438 GiB/s, fitted
    // as t ~= 12.31 ms * ceil(T/8) + 44 ms over a 14x range of T). The wide-token-tile kernel keeps
    // the weight window staged across all tokens of its tile, dropping that to ceil(T/64). Rungs:
    //   T <= 8  -> small_t  (one pass already; also the verified MTP-verify path, left untouched)
    //   T >= 9  -> wide_t   (one pass up to 64 tokens: prefill, and batched multi-sequence passes,
    //                        where 11 tokens in one tile is what the 5-lane 25% efficiency costs)
    //   T >= 33 -> s8       (int8 operands: 1.20-1.30x the bf16 wide rung from T=40 up, and the
    //                        first rung that attacks the measured instruction/pipe oversubscription.
    //                        Only when the caller supplied the quantization scratch -- the *_basis
    //                        entry points have no workspace and stay on the bf16 rungs.)
    // Diagnostic for the rung dispatch: which path each call ACTUALLY takes, and why. Gated on an
    // env var and capped, so it costs nothing in normal runs. (The first version of this probe only
    // printed the CONDITIONS, which is how the PPL gate came back bit-identical with the int8 rung
    // enabled and still left the question open -- so this one is called from the branch that was
    // taken, after the decision, and names the rung.)   NINFER_TERNARY_S8_DEBUG=1 -> stderr lines
    //
    // 2026-09-23: the budget is env-tunable and a min-T filter was added, because ONE forward runs
    // ~320 ternary calls, so the fixed 24-line budget was entirely consumed by the very first
    // forward -- which left "which rung does the T=6 verify step of --draft-tokens 5 take"
    // unanswerable, and that gap was already recorded (stale-assets S-23: "make the print budget env
    // adjustable, default 24").  NINFER_TERNARY_S8_DEBUG_BUDGET (default 24);
    // NINFER_TERNARY_S8_DEBUG_MIN_T (default 0) prints only calls with at least that many tokens.
    static const int probe_min_t = []() {
        const char* env = std::getenv("NINFER_TERNARY_S8_DEBUG_MIN_T");
        return env != nullptr ? std::atoi(env) : 0;
    }();
    static int probe_budget = []() {
        const char* env = std::getenv("NINFER_TERNARY_S8_DEBUG_BUDGET");
        return env != nullptr ? std::atoi(env) : 24;
    }();
    const char* const probe_fmt = (w.qtype == QType::PTQ1_0_G128) ? "PTQ1"
                                : (w.qtype == QType::PQ2_0_G128)  ? "PQ2"
                                                                  : "other";
    const auto note_rung = [&](const char* rung) {
        if (probe_budget <= 0 || std::getenv("NINFER_TERNARY_S8_DEBUG") == nullptr) { return; }
        if (static_cast<int>(x.ne[1]) < probe_min_t) { return; }
        --probe_budget;
        std::fprintf(stderr,
                     "[ternary] rung=%s fmt=%s qtype=%d T=%d k=%d n=%d stride=%d wide_admits=%d "
                     "small_admits=%d gemv_admits=%d s8_scratch=%d s8_switch=%d s8_min=%d mma=%d "
                     "ptq1_fast=%d token_grid=%d\n",
                     rung, probe_fmt, static_cast<int>(w.qtype), static_cast<int>(x.ne[1]),
                     static_cast<int>(w.k), static_cast<int>(w.n),
                     static_cast<int>(out_row_stride), mma_wide_admits(w, out_row_stride) ? 1 : 0,
                     mma_admits(w, out_row_stride) ? 1 : 0, gemv_admits(x, w, 4) ? 1 : 0,
                     scratch.codes != nullptr ? 1 : 0, ternary_s8_enabled() ? 1 : 0,
                     ternary_s8_min_tokens(), ternary_mma_enabled() ? 1 : 0,
                     ternary_ptq1_fast_enabled() ? 1 : 0, ternary_token_grid_enabled() ? 1 : 0);
    };
    if (scratch.codes != nullptr && scratch.scales != nullptr && ternary_s8_enabled() &&
        ternary_mma_enabled() &&
        // 2026-09-23: PTQ1_0 is admitted here on the kPtq1 instantiation, which repacks its raw rows
        // into the PQ2_0 code layout in shared first. The old PQ2_0-only guard was correct while that
        // instantiation did not exist: the s8 kernel reads the 2-bit code plane DIRECTLY, so feeding
        // it base-3 bytes would have mis-decoded silently. With the repack in the staging the hazard
        // is gone, and the accuracy class is unchanged (rel_l2 ~9.0e-3 == the int8 activation error
        // itself, same as PQ2_0's s8). This rung is what PQ2_0 prefill has been running all along --
        // PTQ1_0 was simply missing it, which was the whole 2.36x prefill gap (measured on a
        // 4,044-token prompt: 635.7 against 1,500.0 tok/s).
        (w.qtype == QType::PQ2_0_G128 || ptq1_repack_admits(w)) &&
        mma_wide_admits(w, out_row_stride) && x.ne[1] >= ternary_s8_min_tokens()) {
        note_rung("s8");
        launch_s8(x, w, out, out_row_stride, x.ne[1], scratch, stream);
        return;
    }
    if (ternary_mma_enabled() && mma_wide_admits(w, out_row_stride) &&
        x.ne[1] >= ternary_wide_min_tokens()) {
        note_rung("wide_t");
        launch_wide_t(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (ternary_mma_enabled() && mma_admits(w, out_row_stride) && x.ne[1] >= 2) {
        note_rung("small_t");
        launch_small_t(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (gemv_admits(x, w, 4)) {
        note_rung("gemv_tile");
        launch_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    // REFERENCE-PATH TILE WIDTH -- 8, and a wider tile is a MEASURED REGRESSION, not an oversight.
    // 2026-09-23: this fallback was what PTQ1_0 prefill ran on when a wider tile was tried to
    // amortise the seven CTA-wide reductions per tile (the MMA rungs were PQ2-only then; PTQ1_0 has
    // since been admitted to wide_t and small_t, so it is now reached only for shapes the rungs
    // reject -- padded K, a non-512-multiple K, or the gate switches off).
    // kTileT = 32 (threshold 33 tokens) measured **45.2 -> 19.0 t/s** on a 2,536-token prompt
    // (5,036 tokens: 19.0 t/s too), decode unchanged. The reason is occupancy, not overhead: 32
    // accumulators per thread plus a 16 KB partials tile cost resident CTAs, and this kernel is
    // latency-bound (one FMA per weight per token, weight decoded once per group per tile). Do NOT
    // raise kTileT again -- the way to make PTQ1_0 prefill fast is the tensor-core rung, not this.
    note_rung("reference");
    launch_by_qtype<8>(x, w, out, out_row_stride, stream);
}

} // namespace ninfer::ops::detail
