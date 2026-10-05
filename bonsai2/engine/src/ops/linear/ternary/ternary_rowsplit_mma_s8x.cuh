// s8x: re-tiled int8 prefill rung for PQ2_0 (sm_86). Same inputs/outputs and the SAME float
// arithmetic as ternary_pq2_mma_s8_kernel<64,4,3> (int32 group dot -> fmaf(float(dot), w_scale, acc)
// in group order -> bf16(acc * a_scale)), so the output is bit-identical by construction; the oracle
// checks that bitwise.
//
// What changes (all operand movement / scheduling, no numerics):
//   * A fragment built in registers straight from the 2-bit code plane (m16n8k32 s8 A layout:
//     a0=(row g, k 4l..4l+3) a1=(g+8, same k) a2=(g, k+16) a3=(g+8, k+16) -> code byte l / l+4 of
//     the step's 8 bytes) -- no expansion plane, no __syncwarp, no A ldmatrix.
//   * B via ldmatrix_x4 (two 8-token sub-tiles per instruction).
//   * warp tile MT*16 rows x NT*8 tokens (MT=2 shares every B fragment across two A tiles).
//   * S-stage cp.async ring, one __syncthreads per chunk.
//   * weight scales loaded into a register one ring-distance ahead and parked in shared after the
//     mma loop (the shipped rung's synchronous 2-byte load stalls its warp once per chunk).
//   * exact int->float via the 1.5*2^23 magic (|dot| <= 128*128 < 2^22), same value as cvt.rn.
//   * optional token-fastest CTA order (all token tiles of a row block run together -> weights are
//     fetched from DRAM once instead of once per token tile).
#pragma once
// ENGINE COPY of kbench/s8x.cuh (2026-09-28, RTX 3060 prefill work). Oracle: kbench/s8xb.cu checks
// every instantiation bitwise against ternary_pq2_mma_s8_kernel<64,4,3> (ragged rows/tokens incl.).
// Measured on the 3060 (28 SM), per-forward 27B shape mix, gemm only, vs the A3 s8 rung:
//   T=1024 +34%  T=512 +47%  T=256 +32%  T=128 +27%  T=64 +17%   (45 -> 61 effective TOPS)
// NINFER_TERNARY_S8X=0 restores the shipped kernel (read once, graph-capture safe).
#include "ops/linear/ternary/ternary_rowsplit_mma_s8.cuh"

#include <cstdlib>
#include <string>

namespace ninfer::ops::detail::s8x {

constexpr int kCK = 128, kKSteps = 4, kCodePitch = 48;

template <int MT, int NT, int WM, int WN, int S>
struct Cfg {
    static constexpr int kThreads = WM * WN * 32;
    static constexpr int kRows    = WM * MT * 16;
    static constexpr int kTok     = WN * NT * 8;
    static constexpr int kActBytes   = kTok * kCK;
    static constexpr int kCodeBytes  = kRows * kCodePitch;
    static constexpr int kScaleBytes = ((kRows * 2 + 15) / 16) * 16;
    static constexpr int kStage      = kActBytes + kCodeBytes + kScaleBytes;
    static constexpr int kSmem       = S * kStage;
};

__device__ __forceinline__ float exact_i2f(int v) {
    return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

// GROUP_T: CTA order. 0 = row-block fastest (shipped A3 order). g>0 = token tiles fastest inside
// super-rows of `g` row blocks... simplified: 1 = token-tile fastest.
template <int MT, int NT, int WM, int WN, int S, int MINB, int ORDER, int FLAGS, bool XA>
__global__ void __launch_bounds__(WM * WN * 32, MINB)
kernel(const std::int8_t* __restrict__ act_codes, const float* __restrict__ act_scales,
       const std::uint8_t* __restrict__ w_codes, const std::uint8_t* __restrict__ w_scales,
       __nv_bfloat16* __restrict__ out, int rows, int k, int tokens, int out_row_stride) {
    using C = Cfg<MT, NT, WM, WN, S>;
    extern __shared__ __align__(16) std::uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WM, wn = warp / WM;
    const int gid = lane >> 2, lid = lane & 3;

    int rb, tb;
    if constexpr (ORDER == 0) { rb = blockIdx.x; tb = blockIdx.y; }
    else if constexpr (ORDER == 1) { tb = blockIdx.x; rb = blockIdx.y; }
    else {
        // grouped: ORDER row blocks x all token tiles per group, row block fastest in a group
        const int rbs = (rows + C::kRows - 1) / C::kRows, tbs = (tokens + C::kTok - 1) / C::kTok;
        const int id = blockIdx.x, gsz = ORDER * tbs, grp = id / gsz, within = id - grp * gsz;
        const int g0 = grp * ORDER, gw = min(ORDER, rbs - g0);
        rb = g0 + within % gw; tb = within / gw;
    }
    const int row0 = rb * C::kRows, tok0 = tb * C::kTok;
    if (tok0 >= tokens) return;

    const int groups = k / kCK, chunks = groups;
    const std::int64_t code_row_bytes = static_cast<std::int64_t>(groups) * 32;
    const std::int64_t scale_row_bytes = static_cast<std::int64_t>(groups) * 2;

    auto act_sh   = [&](int s) { return reinterpret_cast<std::int8_t*>(smem + s * C::kStage); };
    auto code_sh  = [&](int s) { return smem + s * C::kStage + C::kActBytes; };
    auto scale_sh = [&](int s) {
        return reinterpret_cast<std::uint16_t*>(smem + s * C::kStage + C::kActBytes + C::kCodeBytes);
    };

    // scale ownership: thread tid < kRows owns row tid's scale
    const bool owns_scale = tid < C::kRows;
    const std::int64_t my_scale_row = static_cast<std::int64_t>(row0) + tid;
    const bool scale_valid = owns_scale && my_scale_row < rows;

    auto stage = [&](int s, int chunk) {
        // activations: kTok rows x 8 segments of 16 B, XOR swizzled like the shipped rung
        std::int8_t* a = act_sh(s);
        for (int copy = tid; copy < C::kTok * 8; copy += C::kThreads) {
            const int row = copy >> 3, seg = copy & 7;
            const int phys = (seg ^ (row & 7)) << 4;
            const int gt = tok0 + row;
            std::int8_t* dst = a + row * kCK + phys;
            if (gt < tokens) {
                cp_async<16, Cache::cg>(dst, act_codes + static_cast<std::int64_t>(gt) * k +
                                                 static_cast<std::int64_t>(chunk) * kCK + seg * 16);
            } else {
                store_vec<std::int8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }
        std::uint8_t* c = code_sh(s);
        for (int copy = tid; copy < C::kRows * 2; copy += C::kThreads) {
            const int r = copy >> 1, half = copy & 1;
            const std::int64_t gr = static_cast<std::int64_t>(row0) + r;
            std::uint8_t* dst = c + r * kCodePitch + half * 16;
            if (gr < rows) {
                cp_async<16, Cache::cg>(dst, w_codes + gr * code_row_bytes +
                                                 static_cast<std::int64_t>(chunk) * 32 + half * 16);
            } else {
                store_vec<std::uint8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }
    };
    auto load_scale = [&](int chunk) -> std::uint16_t {
        return scale_valid ? load_vec<std::uint16_t>(w_scales + my_scale_row * scale_row_bytes +
                                                     static_cast<std::int64_t>(chunk) * 2)
                           : static_cast<std::uint16_t>(0);
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.0f;

    // prologue: stages 0..S-2 (scales stored directly -- first use is after a barrier)
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < chunks) {
            stage(s, s);
            if (owns_scale) scale_sh(s)[tid] = load_scale(s);
        }
        cp_commit();
    }

    const int sh = lid * 8;
    const int b_row4 = (lane & 7) + ((lane >> 4) << 3);
    const int b_col  = ((lane >> 3) & 1) * 16;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int buf = chunk % S;
        cp_wait<S - 2>();
        __syncthreads();
        const int nxt = chunk + S - 1;
        std::uint16_t pending_scale = 0;
        if (nxt < chunks) {
            stage(nxt % S, nxt);
            pending_scale = load_scale(nxt);
        }
        cp_commit();

        int tmp[MT][NT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int i = 0; i < 4; ++i) tmp[m][n][i] = 0;

        const std::uint8_t* cs = code_sh(buf);
        const std::int8_t* as = act_sh(buf);
#pragma unroll
        for (int step = 0; step < kKSteps; ++step) {
            unsigned a[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int r = (wm * MT + m) * 16 + gid;
                const uint2 lo = *reinterpret_cast<const uint2*>(cs + r * kCodePitch + step * 8);
                const uint2 hi = *reinterpret_cast<const uint2*>(cs + (r + 8) * kCodePitch + step * 8);
                a[m][0] = ternary_s8_expand_codes(lo.x >> sh);
                a[m][1] = ternary_s8_expand_codes(hi.x >> sh);
                a[m][2] = ternary_s8_expand_codes(lo.y >> sh);
                a[m][3] = ternary_s8_expand_codes(hi.y >> sh);
            }
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                const int row = (wn * NT + np * 2) * 8 + b_row4;
                const int logical = step * 32 + b_col;
                const int phys = (((logical >> 4) ^ (row & 7)) << 4);
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, smem_addr(as + row * kCK + phys));
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma_s8(tmp[m][2 * np][0], tmp[m][2 * np][1], tmp[m][2 * np][2], tmp[m][2 * np][3],
                           a[m][0], a[m][1], a[m][2], a[m][3], b0, b1);
                    mma_s8(tmp[m][2 * np + 1][0], tmp[m][2 * np + 1][1], tmp[m][2 * np + 1][2],
                           tmp[m][2 * np + 1][3], a[m][0], a[m][1], a[m][2], a[m][3], b2, b3);
                }
            }
        }

        const std::uint16_t* ss = scale_sh(buf);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r = (wm * MT + m) * 16 + gid;
            const float w_top = __half2float(__ushort_as_half(ss[r]));
            const float w_bot = __half2float(__ushort_as_half(ss[r + 8]));
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if constexpr ((FLAGS & 1) != 0) {
                    acc[m][n][0] = fmaf(exact_i2f(tmp[m][n][0]), w_top, acc[m][n][0]);
                    acc[m][n][1] = fmaf(exact_i2f(tmp[m][n][1]), w_top, acc[m][n][1]);
                    acc[m][n][2] = fmaf(exact_i2f(tmp[m][n][2]), w_bot, acc[m][n][2]);
                    acc[m][n][3] = fmaf(exact_i2f(tmp[m][n][3]), w_bot, acc[m][n][3]);
                } else {
                    acc[m][n][0] = fmaf(__int2float_rn(tmp[m][n][0]), w_top, acc[m][n][0]);
                    acc[m][n][1] = fmaf(__int2float_rn(tmp[m][n][1]), w_top, acc[m][n][1]);
                    acc[m][n][2] = fmaf(__int2float_rn(tmp[m][n][2]), w_bot, acc[m][n][2]);
                    acc[m][n][3] = fmaf(__int2float_rn(tmp[m][n][3]), w_bot, acc[m][n][3]);
                }
            }
        }
        // park the scale fetched at the top of this iteration (its buffer is free: every warp
        // passed this iteration's barrier, so nobody still reads chunk-1's slot == nxt % S)
        if (nxt < chunks && owns_scale) scale_sh(nxt % S)[tid] = pending_scale;
    }

    if constexpr ((FLAGS & 2) != 0) {
        // smem-staged epilogue: [token][row] bf16 tile, then 16-byte row-contiguous stores
        constexpr int kPitch = C::kRows + 8;
        static_assert(C::kTok * kPitch * 2 <= C::kSmem, "epilogue tile must fit the ring");
        cp_wait<0>();
        __syncthreads();
        __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int rl = (wm * MT + m) * 16 + gid;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int tl = (wn * NT + n) * 8 + 2 * lid;
                const int ta = tok0 + tl;
                const float al = (ta < tokens) ? act_scales[ta] : 0.0f;
                const float ar = (ta + 1 < tokens) ? act_scales[ta + 1] : 0.0f;
                tile[tl * kPitch + rl]           = __float2bfloat16_rn(acc[m][n][0] * al);
                tile[(tl + 1) * kPitch + rl]     = __float2bfloat16_rn(acc[m][n][1] * ar);
                tile[tl * kPitch + rl + 8]       = __float2bfloat16_rn(acc[m][n][2] * al);
                tile[(tl + 1) * kPitch + rl + 8] = __float2bfloat16_rn(acc[m][n][3] * ar);
            }
        }
        __syncthreads();
        constexpr int kVecPerTok = C::kRows / 8;
        const bool aligned = (out_row_stride % 8) == 0 &&
                             (reinterpret_cast<std::uintptr_t>(out) % 16) == 0;
        for (int v = tid; v < C::kTok * kVecPerTok; v += C::kThreads) {
            const int tl = v / kVecPerTok, c8 = (v - tl * kVecPerTok) * 8;
            const int t = tok0 + tl, r = row0 + c8;
            if (t >= tokens) continue;
            __nv_bfloat16* dst = out + static_cast<std::int64_t>(t) * out_row_stride + r;
            const __nv_bfloat16* src = tile + tl * kPitch + c8;
            if (aligned && r + 8 <= rows) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            } else {
                for (int j = 0; j < 8; ++j) if (r + j < rows) dst[j] = src[j];
            }
        }
        return;
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int row_lo = row0 + (wm * MT + m) * 16 + gid;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int ta = tok0 + (wn * NT + n) * 8 + 2 * lid;
            const int tb2 = ta + 1;
            const float al = (ta < tokens) ? act_scales[ta] : 0.0f;
            const float ar = (tb2 < tokens) ? act_scales[tb2] : 0.0f;
            if (row_lo < rows) {
                if (ta < tokens) out[static_cast<std::int64_t>(ta) * out_row_stride + row_lo] = __float2bfloat16_rn(acc[m][n][0] * al);
                if (tb2 < tokens) out[static_cast<std::int64_t>(tb2) * out_row_stride + row_lo] = __float2bfloat16_rn(acc[m][n][1] * ar);
            }
            if (row_hi < rows) {
                if (ta < tokens) out[static_cast<std::int64_t>(ta) * out_row_stride + row_hi] = __float2bfloat16_rn(acc[m][n][2] * al);
                if (tb2 < tokens) out[static_cast<std::int64_t>(tb2) * out_row_stride + row_hi] = __float2bfloat16_rn(acc[m][n][3] * ar);
            }
        }
    }
}

template <int MT, int NT, int WM, int WN, int S, int MINB, int ORDER, int MAGIC>
inline void launch(const std::int8_t* aq, const float* as, const std::uint8_t* wc, const std::uint8_t* ws,
                   __nv_bfloat16* out, int rows, int k, int tokens, int stride, cudaStream_t st = 0) {
    using C = Cfg<MT, NT, WM, WN, S>;
    auto fn = kernel<MT, NT, WM, WN, S, MINB, ORDER, MAGIC, false>;
    static_assert(C::kSmem <= 48 * 1024, "stays under the default dynamic-smem limit (no attribute call, graph safe)");
    const unsigned rbs = (rows + C::kRows - 1) / C::kRows, tbs = (tokens + C::kTok - 1) / C::kTok;
    const dim3 grid = ORDER == 0 ? dim3(rbs, tbs) : ORDER == 1 ? dim3(tbs, rbs) : dim3(rbs * tbs);
    fn<<<grid, C::kThreads, C::kSmem, st>>>(aq, as, wc, ws, out, rows, k, tokens, stride);
}
} // namespace ninfer::ops::detail::s8x

namespace ninfer::ops::detail {
[[nodiscard]] inline bool ternary_s8x_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8X");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}
// the shipped configuration: 32x64 warp tile, 128x64 CTA, 3-stage ring, grouped order (8 row
// blocks x all token tiles), smem-staged epilogue, plain cvt.rn fold.
inline void launch_pq2_mma_s8x(const std::int8_t* aq, const float* as, const std::uint8_t* wc,
                               const std::uint8_t* ws, __nv_bfloat16* out, int rows, int k,
                               int tokens, int stride, cudaStream_t st) {
    s8x::launch<2, 8, 4, 1, 3, 2, 8, 2>(aq, as, wc, ws, out, rows, k, tokens, stride, st);
}
} // namespace ninfer::ops::detail
