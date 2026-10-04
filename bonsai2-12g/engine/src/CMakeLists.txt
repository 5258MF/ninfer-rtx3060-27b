function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(ninfer_core STATIC
  core/arena.cu
  core/device.cu
  core/dtype.cpp
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/host_kv_arena.cpp
  core/paged_kv_cache.cpp
  core/paged_kv_gather.cu
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/admission_policy.cpp
  runtime/engine/context_cost.cpp
  runtime/engine/context_cost_defaults.cpp
  runtime/engine/kv_capacity.cpp
  runtime/engine/public_types.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3 Threads::Threads)
if(WIN32)
  # WDDM residency lock (opt-in via --wddm-evictable-budget) uses the D3D12/DXGI driver API.
  target_link_libraries(ninfer_core PRIVATE d3d12 dxgi)
endif()

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from producer
# warpgroups to consumer warpgroups. Relocatable device code makes ptxas discard that contract, so
# these self-contained kernels are intentionally compiled in a non-RDC archive and called through
# host launchers from ninfer_ops.
# They rely on Blackwell TMA instructions and do not compile for sm_89, so the archive is
# only built (and linked) for 120a builds.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^120")
  add_library(ninfer_nvfp4_non_rdc STATIC
    ops/linear/nvfp4/nvfp4_w4a4_tma.cu
    ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu
    ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc.cu)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  set_target_properties(ninfer_nvfp4_non_rdc PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_compile_options(ninfer_nvfp4_non_rdc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)
  set(_ninfer_nvfp4_non_rdc ninfer_nvfp4_non_rdc)
endif()

# SM120-only NVFP4 W4A4 (TMA) kernels: excluded from non-120 builds; nvfp4_w4a4_stubs.cpp
# keeps the dispatch layer linkable with a clear runtime error on this architecture.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^120")
  set(_ninfer_nvfp4_w4a4_sources
    ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
    ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
    ops/linear/nvfp4/nvfp4_w4a4.cu
    ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
    ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu)
else()
  set(_ninfer_nvfp4_w4a4_sources ops/nvfp4_w4a4_stubs.cpp)
endif()

# SM120-only k8v4/NVFP4 KV-cache and attention kernels: the mixed 8x4 MMA uses
# kind::f8f6f4 and NVFP4 uses e2m1 (FP4) instructions, both Blackwell-only. Excluded from
# non-120 builds; sm120_kv_stubs.cpp keeps the dispatch layer linkable with a clear runtime
# error on this architecture.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^120")
  set(_ninfer_sm120_kv_sources
    ops/kv_cache/append/k8v4_launch.cu
    ops/kv_cache/append/nvfp4_launch.cu
    ops/softmax_attention/dense/causal_cache/prompt_k8v4.cu
    ops/softmax_attention/dense/causal_cache/small_t_k8v4.cu
    ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cu)
else()
  set(_ninfer_sm120_kv_sources ops/sm120_kv_stubs.cpp)
endif()

# w8 splitk/small-T GEMM kernels: built for every supported architecture. The kernel and its
# contraction (w8_small_t_mma.cuh) are the single upstream implementation; on sm_89 (and sm_86)
# the schedules in w8_config.h are capped so the static shared layout stays within the 48 KiB
# limit, and sm_120a keeps the high-shared-memory schedules. The former w8_sm120_stubs.cpp is
# no longer referenced.
set(_ninfer_w8_sm120_sources
  ops/linear/w8/w8_small_t.cu
  ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_splitk.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ops/linear_add/w8/w8_linear_add_gemm_splitk.cu
  ops/linear_pair/w8/w8_pair_gemm_splitk.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gelu.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/speculative_round.cu
  ops/launcher/position.cu
  ops/launcher/prepare_ragged_prefix.cu
  ops/launcher/prepare_masked_block.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rmsnorm_pack_tail.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/target_logprobs.cu
  ops/launcher/vision_pos_embed.cu
  ops/rmsnorm_rope/rmsnorm_rope.cpp
  ops/rmsnorm_rope/launch.cu
  ops/context_kv_materialize/context_kv_materialize.cpp
  ops/context_kv_materialize/materialize.cu
  # KVMem raw-K shadow: harvest the pre-RoPE K. Diagnostic, OFF by default (NINFER_TERNARY_KVMEM=1
  # enables it); the host half carries the switch and the capture-checked drain, the launch half the
  # capture-safe bit copy. See src/ops/kvmem/kvmem_shadow.h.
  ops/kvmem/raw_k_shadow.cpp
  ops/kvmem/raw_k_shadow_launch.cu
  # Per-layer staging + the drain driver for the harvest above: one device slot and one pinned host
  # slot per full-attention layer, so a prefill chunk keeps all 16 layers instead of only the last.
  # Allocates at construction, which is why it must be built while the graph is being prepared and
  # never inside a capture. See src/ops/kvmem/raw_k_harvest.h.
  ops/kvmem/raw_k_harvest.cpp
  # KVMem mean-K index (Eq.9): the first CONSUMER of the harvest. Reduces each layer's pre-RoPE
  # keys into one FP16 vector per (layer, logical block, KV head), appended chunk by chunk while the
  # keys are still in hand. The official implementation rejects the alternative -- rebuilding the
  # index from the paged cache after the prefill -- because a block offloaded mid-prefill would never
  # be reachable by a later scan, so the index would silently lose history. See mean_k_index.h.
  ops/kvmem/mean_k_index.cpp
  ops/kvmem/mean_k_index_launch.cu
  # KVMem block selection (reference section 5): which blocks survive into the assembled window. Pure
  # host logic -- one retrieval score per candidate block in, a block-id list out -- with no device
  # state, no switch and no arena, so it can be pinned down before the index and the scorer exist.
  # Its tie rule (ties go to the NEWER block), its zero-budget rule (0 means unlimited, because the
  # caller truncates select_budget/block_tokens) and its "no signal = -inf" encoding are all aligned
  # with the reference; a 3675-case cross-check drives the reference's own KvMemStore and this
  # selector side by side. See src/ops/kvmem/kvmem_select.h.
  ops/kvmem/kvmem_select.cpp
  # KVMem window assembly: the missing path between the selector above and the page table the
  # attention reads -- "block id -> physical page slot -> published at the slot's WINDOW position".
  # The window is a contiguous prefix with sequential live positions (softmax_attention.h:118-123),
  # so its order is CHRONOLOGICAL and NOT the score order; publishing in score order yields a window
  # that is full, valid and silently wrong. A selected block with no live page is a hole and is
  # refused rather than dropped. Host-only (no CUDA, no arena): the physical slots arrive through an
  # abstract source so the placement is checkable with no GPU. See kvmem_window_assembly.h.
  ops/kvmem/kvmem_window_assembly.cpp
  # KVMem retrieval scoring (Eq.10): the dot product between the de-RoPEd queries of the marked span
  # and each block's mean-K vector, reduced to one score per block for the selection above. Every file
  # here is deliberately capture-safe -- no allocation, no memcpy, no stream synchronisation -- because
  # the caller runs it INSIDE the capture window, between the rmsnorm and the rope, where the pre-RoPE
  # q and k are still alive and hot (that is what removes the need for a raw-Q shadow entirely).
  # Checked against the reference scoring kernel on identical inputs: rel_l2 2.2e-07. See
  # src/ops/kvmem/kvmem_retrieve.h.
  ops/kvmem/kvmem_retrieve.cpp
  ops/kvmem/kvmem_retrieve_launch.cu
  # The PORTED reference policy layer (Apache-2.0, verbatim from kvmem-qw3 -- see the attribution
  # block at the top of each file and qw3/LICENSE-kvmem-qw3.txt). It owns the block metadata, the
  # selection, the working-set diff and the remap plan, and it is deliberately engine-independent.
  # It is registered here so the wiring layer can call it; the four upstream host tests still pass
  # against it unchanged (kvmem_store_test / pinned_kv_tier_test / refresh_policy / prefix_reuse).
  ops/kvmem/qw3/kvmem_store.cpp
  ops/kvmem/qw3/kvmem_gc.cpp
  # Bridge between that ported layer and ninfer's own types: geometry (block = one KV page = 64, not
  # the reference's 128), config translation, plan translation plus the invariants that catch a bad
  # plan before it reaches the engine, the engine-independent host hooks, and the explicit list of
  # what is still missing before this can be wired up. See kvmem_port_bridge.h.
  ops/kvmem/kvmem_port_bridge.cpp
  # KVMem re-RoPE consumer (接线的 P2 路径): re-bake the K of a block that has been MOVED to a new
  # window position. WHY THERE IS NO IN-PLACE VARIANT: the fp8 K plane holds quantize(H * bakedK)
  # with H = normalized Sylvester Hadamard D256 (see ops/kv_cache/append/kernel.cuh:36), and H does
  # NOT commute with the paired RoPE rotation -- rotating the stored code in place computes a
  # different vector (measured: 65456/65536 codes differ). So this rebakes from the raw-K shadow:
  # raw K -> RoPE at the new position -> BF16 -> H -> absmax -> the engine's own codec, which is
  # drift-free by construction (8 consecutive re-moves are byte-identical). Requires rope_dim to be
  # a multiple of 64; other geometries are REJECTED rather than silently wrong.
  ops/kvmem/rerope/kvmem_rerope_launch.cu
  ops/candidate_selector/bf16/candidate_selector_path.cu
  ops/candidate_selector/bf16/candidate_selector_path_plan.cpp
  # Softmax Attention and KV-cache state transitions.
  ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
  ops/softmax_attention/dense/causal_cache/small_t.cu
  ops/softmax_attention/dense/causal_cache/small_t_fp8.cu
  ops/softmax_attention/dense/causal_cache/prompt.cu
  ops/softmax_attention/dense/causal_cache/prompt_fp8.cu
  ops/softmax_attention/dense/causal_cache/prompt_nvfp4.cu
  ops/softmax_attention/dense/packed/packed_softmax_attention.cpp
  ops/softmax_attention/dense/packed/launch.cu
  ops/softmax_attention/dense/context/context_softmax_attention.cpp
  ops/softmax_attention/dense/context/launch.cu
  ops/softmax_attention/sliding_window/sliding_window_attention.cpp
  ops/softmax_attention/sliding_window/launch.cu
  ops/kv_cache/append/kv_cache_append.cpp
  ops/kv_cache/append/launch.cu
  # Gated DeltaNet Linear Attention core.
  ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ops/linear_attention/gated_delta_net/replay.cpp
  ops/linear_attention/gated_delta_net/recurrent.cu
  ops/linear_attention/gated_delta_net/chunked/launch.cu
  ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ops/linear_attention/gated_delta_net/chunked/output.cu
  ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a16_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a16_gemm.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ops/attn_input_proj/w8/w8_dflash2_attn_input.cu
  ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_matrix.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/gdn_projected_conv.cu
  ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp
  ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_materialized.cu
  ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.cpp
  ops/linear/bf16/bf16_gemv.cu
  ops/linear/bf16/bf16_gemm_mma.cu
  ops/linear/bf16/bf16_n256_k5120.cu
  ops/linear/bf16/bf16_small_t.cu
  ops/linear/bf16/bf16_dispatch.cpp
  ops/linear/fp8/fp8_format.cpp
  ops/linear/fp8/fp8_gemv.cu
  ops/linear/fp8/fp8_small_t.cu
  ops/linear/fp8/fp8_a16_small_t_mma.cu
  ops/linear/fp8/fp8_a16_gemm_mma.cu
  ops/linear/fp8/fp8_a8.cu
  ops/linear/fp8/fp8_dispatch.cpp
  ops/linear/nvfp4/nvfp4_format.cpp
  ops/linear/nvfp4/nvfp4_gemv.cu
  ops/linear/nvfp4/nvfp4_small_t.cu
  ops/linear/nvfp4/nvfp4_dispatch.cpp
  ops/linear/q4/q4_small_t_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_dispatch.cpp
  ops/linear/q5/q5_dispatch.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear_add/bf16/bf16_linear_add_decode.cu
  ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ops/linear_add/fp8/fp8_linear_add_decode.cu
  ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ops/linear_add/fp8/fp8_linear_add_a8.cu
  ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ops/linear_add/w8/w8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu
  ops/linear_swiglu/w8/w8_dflash2_linear_swiglu.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp
  ops/linear/q6/q6_dispatch.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/ternary/ternary_rowsplit_gemm.cu
  ops/linear/ternary/ternary_dispatch.cpp
  ops/linear/ternary/ternary_rotation.cu
  ops/linear/ternary/ternary_rotation.cpp
  ops/linear/w8/w8_feature.cu
  ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ops/linear/w8/w8_dispatch.cpp
  ops/linear_pair/w8/w8_pair_decode.cu
  ops/linear_pair/w8/w8_pair_gemm_concat.cu
  ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ops/linear_pair/w8/w8_pair_plan.cpp
  ops/linear_topk/linear_topk.cpp
  ops/linear_topk/w8.cu
  ops/linear_topk/w8_m64.cu
  ops/linear_topk/fp8.cu
  ops/linear_topk/fp8_m64.cu
  ops/linear_topk/q4.cu
  ops/linear_topk/q4_m64.cu
  ops/linear_topk/merge.cu
  ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ops/linear/linear.cpp
  ops/wrapper/add_bias.cpp
  ops/wrapper/argmax.cpp
  ops/wrapper/cast.cpp
  ops/wrapper/causal_conv1d_silu.cpp
  ops/wrapper/candidate_selector.cpp
  ops/wrapper/dynamic_grouped_conv.cpp
  ops/wrapper/embedding.cpp
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/speculative_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/prepare_ragged_prefix.cpp
  ops/wrapper/prepare_masked_block.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rmsnorm_pack_tail.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/target_logprobs.cpp
  ops/wrapper/vision_pos_embed.cpp
  ${_ninfer_nvfp4_w4a4_sources}
  ${_ninfer_sm120_kv_sources}
  ${_ninfer_w8_sm120_sources})
ninfer_internal_includes(ninfer_ops)
# The ported KVMem policy layer (ops/kvmem/qw3/, Apache-2.0) is kept BYTE-IDENTICAL to its upstream,
# so its own sources still say #include "qw3/kvmem_store.hpp". Adding src/ops/kvmem is what makes
# that verbatim include resolve without editing the ported files -- kvmem-port's REPORT.md §1 says the
# qw3/ subdirectory exists for exactly this reason, but this include dir was never added to the ninja
# target, so those two TUs failed with C1083 the first time they were configured into b1 (2026-09-22).
target_include_directories(ninfer_ops PRIVATE ${PROJECT_SOURCE_DIR}/src/ops/kvmem)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ${_ninfer_nvfp4_non_rdc})

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)

add_library(ninfer_media_decode STATIC
  media/decode/decode.cpp)
ninfer_internal_includes(ninfer_media_decode)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)
  target_compile_definitions(ninfer_media_decode PRIVATE NINFER_HAVE_FFMPEG)
endif()

if(NINFER_BUILD_MEDIA_ACQUIRE OR NINFER_BUILD_PROMPT_INPUT)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  # fork: 该库本身无外部依赖（curl 链接本就注释掉），但 prompt_input/apps 无条件链接它，
  # 因此关掉 media 时也必须构建，否则 LNK1181 找不到 ninfer_media_acquire.lib。
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  ninfer_internal_includes(ninfer_media_acquire)
  # target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
endif()

if(NINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)
endif()

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(ninfer_product_logging STATIC
    product/logging/logging.cpp
    product/logging/pretty_format.cpp
    product/logging/startup_log.cpp)
  ninfer_internal_includes(ninfer_product_logging)
  target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/contract/sampling.cpp
  runtime/engine/engine.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(targets/qwen3_6)
add_subdirectory(targets/qwen3_6_27b)
add_subdirectory(targets/qwen3_6_35b_a3b)

target_link_libraries(ninfer_engine
  PRIVATE
    ninfer_artifact
    ninfer_core
    ninfer_ops
    ninfer_text
    ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(ninfer_serve STATIC
    serve/http_transport.cpp
    serve/request_validation.cpp
    serve/openai_chat_request.cpp
    serve/openai_chat_response.cpp
    serve/openai_chat_http.cpp
    serve/openai_common.cpp
    serve/openai_responses_request.cpp
    serve/openai_responses_state.cpp
    serve/openai_responses_response.cpp
    serve/anthropic_messages_request.cpp
    serve/anthropic_messages_response.cpp
    serve/anthropic_messages_http.cpp
    serve/openai_responses_store.cpp
    serve/translate.cpp
    serve/generation_service.cpp
    serve/request_events.cpp
    serve/operational_log.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/openai_responses_http.cpp
    serve/serve_options.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire ninfer_product_logging CUDA::cudart)
endif()


