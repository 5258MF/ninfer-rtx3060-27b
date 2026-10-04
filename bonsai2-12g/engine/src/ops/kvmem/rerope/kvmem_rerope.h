#pragma once

// KVMem re-RoPE consumer (host-safe contract). Device half: kvmem_rerope_launch.cu.
//
// WHY THIS EXISTS
//   Our engine's visible set is a CONTIGUOUS PREFIX by contract
//   (include/ninfer/ops/softmax_attention.h:118-123: "query head h attends cache rows [0,p]",
//   "live positions are sequential"), and there is no per-cell mask anywhere in the tree. A KVMem
//   block selection therefore cannot be expressed as holes: the selected pages must be ASSEMBLED
//   INTO A COMPACT WINDOW. The K in those pages was baked by RoPE at its ORIGINAL position, so
//   moving a page to a new window position requires re-baking its K at the new one. This file is
//   that consumer. Assembling the bytes at the new window slots (the page-table move) is stage 2a's
//   job, not this file's.
//
// ONE PATH, NOT TWO — the in-place code-rotation path was DROPPED, and here is why
//   The cheap-looking design rotates the stored fp8 CODES in place (de-rotate by the old position,
//   re-rotate by the new one, keep the per-vector scale: R(s*x) = s*R(x), so the scale plane would
//   survive untouched). That reasoning is correct about the SCALE and still WRONG about the CODES:
//   our fp8 K is not stored in the RoPE basis. The append path transforms the row with the
//   normalized Sylvester Hadamard first and quantizes the TRANSFORMED row --
//     ops/kv_cache/append/kernel.cuh:36  normalized_hadamard_d256_inplace(values, lane);  // K only
//     ops/kv_cache/append/kernel.cuh:43  cache_k[...] = kv_cache_fp8_quant_code(values[r], ...);
//   and the attention side transforms Q with the same operator and consumes the stored codes as-is
//     ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh:137 (H on Q), :204 (K codes as stored)
//   so a stored code row is a quantized H*(rope-baked K), not a quantized rope-baked K. H does not
//   commute with the pair rotations of RoPE, so rotating codes in the code domain is not a rotation
//   of K at all. It is not a precision trade-off: it computes the wrong vector.
//   Measured, not asserted -- the standalone verifier keeps the code-domain rotation as a negative
//   control and prints its error against the engine pipeline (see REROPE_SUMMARY, NEG-2). Note the
//   Hadamard-on-K invariant holds for every KV encoding in this tree (fp8, k8v4, nvfp4, i8), so the
//   conclusion is not specific to one dtype.
//
// SO THE ONLY CORRECT PATH IS THE RE-BAKE, and it is also the better one
//   kvmem_rerope_rebake_*: take the RAW (pre-RoPE) K, which the raw-K shadow already keeps
//   (ops/kvmem/raw_k_shadow.*), apply RoPE at the NEW position, narrow to BF16 exactly as the
//   engine's KV tensor would, then run the engine's OWN append pipeline (Hadamard -> row absmax ->
//   fp8 codec) so the bytes are the bytes the engine would have stored had the block lived at the
//   new position all along. Consequences: drift-free by construction (every move is recomputed from
//   the same raw K, so N round trips are byte-identical to a fresh store), and it needs no scale
//   arithmetic of its own -- the engine's codec owns that.
//
// POSITION CONTRACT
//   `win_base` is the ABSOLUTE token position of the block's first token: the token at index `tok`
//   is baked at `win_base + tok` and the bytes are written to the window slots `win_base + tok` of
//   `block_table`. (For a freshly assembled window this is the same number, which is why one base
//   suffices; the block must already sit at those slots.) Addressing goes through the engine's own
//   `paged_kv_element_offset`, so the page layout is never re-derived here.
//   `n_tokens == 0` must produce NO WRITE AT ALL.
//
// GEOMETRY
//   The bake is the engine's fixed Text 1-D D256/R64 theta=1e7 path: dims [0, rope_dim) are rotated
//   in split-half pairs (i, i + rope_dim/2) with kTextRopeInvFrequency[i], dims [rope_dim, 256) are
//   passed through unchanged. kv_heads must be 4 or 2; rope_dim must be even and <= 256.

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kKvMemReropeHeadDim   = 256;
inline constexpr std::int32_t kKvMemReropeRotaryDim = 64;    // Text 1-D D256/R64
inline constexpr float        kKvMemReropeTheta     = 1.0e7F; // Text 1-D

// Which K codec the destination planes use.
//
// The re-bake runs the engine's OWN append pipeline, so it has to write whatever
// the destination KV planes are encoded in. The file header already records that
// the Hadamard-on-K invariant holds for every encoding in this tree
// (fp8, k8v4, nvfp4, i8), which is what makes a second codec a store-side change
// and nothing more: RoPE, the BF16 narrowing and the Hadamard are shared, only
// the final quantize-and-store differs.
//
// Int8Group64 exists because fp8 KV needs sm_89 FP8 tensor cores in attention,
// so on Ampere the only KV encoding that can actually be read back is int8 --
// without this the window-assembly arm is unreachable on that hardware.
enum class KvMemReropeCodec : std::int32_t {
    Fp8E4M3Row256 = 0, // 256 codes + 1 FP16 scale per (token, kv head)
    Int8Group64   = 1, // 256 codes + 4 FP16 group scales per (token, kv head)
};

// Path B, one block. Returns the launch status instead of asserting, so a standalone verifier can
// tell WHICH call failed (same reasoning as mean_k_index_launch.h).
//   raw_k_bf16     : BF16 [head_dim, kv_heads, tokens] -- the harvest/append source layout, i.e.
//                    element (d, head, tok) sits at d + head_dim * (head + kv_heads * tok)
//                    (ops/kv_cache/fp8_e4m3_row_codec.cuh:36-42, kv_cache_fp8_src_index).
//   raw_token_begin: index of the block's first token inside raw_k_bf16 (0 when the slice IS the
//                    block).
//   k_codes/k_scales: the destination K planes (Fp8E4M3Row256: 256 codes + 1 FP16 scale per
//                    (token, kv head)); every code and scale of the block is written.
cudaError_t kvmem_rerope_rebake_launch(const void* raw_k_bf16, std::int32_t raw_token_begin,
                                       void* k_codes, void* k_scales,
                                       const std::int32_t* block_table, std::int32_t kv_heads,
                                       std::int32_t win_base, std::int32_t n_tokens,
                                       std::int32_t rope_dim, float theta, KvMemReropeCodec codec,
                                       cudaStream_t stream);

// Path B for every moved block of a layer in one launch. Per-block arguments are DEVICE arrays of
// length n_blocks; a block whose n_tokens <= 0 is skipped entirely (no write). `max_tokens` bounds
// the launch envelope (the largest n_tokens in the batch).
cudaError_t kvmem_rerope_rebake_batched_launch(const void* raw_k_bf16,
                                               void* k_codes, void* k_scales,
                                               const std::int32_t* block_table,
                                               const std::int32_t* to_base,
                                               const std::int32_t* raw_token_begin,
                                               const std::int32_t* n_tokens,
                                               std::int32_t n_blocks, std::int32_t max_tokens,
                                               std::int32_t kv_heads, std::int32_t rope_dim,
                                               float theta, KvMemReropeCodec codec,
                                               cudaStream_t stream);

} // namespace ninfer::ops
