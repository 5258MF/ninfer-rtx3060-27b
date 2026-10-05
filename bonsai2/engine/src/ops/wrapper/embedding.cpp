// ninfer::ops - embedding wrapper: public api validation and qtype dispatch.
#include "ninfer/ops/embedding.h"

#include "ops/common/math.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/launcher/embed_gather.h" // detail::embed_gather_*_launch
#include "core/device.h"
#include "core/weight.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("embedding: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("embedding: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error("embedding: weight payload size overflows uint64");
    }
    return a * b;
}

std::int32_t align_up_i32(std::int32_t x, std::int32_t m) {
    const std::int64_t y = round_up(static_cast<std::int64_t>(x), static_cast<std::int64_t>(m));
    if (y > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("embedding: padded shape overflows int32");
    }
    return static_cast<std::int32_t>(y);
}

void require_ids_shape(const Tensor& ids) {
    if (ids.ne[1] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("embedding: ids must have shape [T]");
    }
}

void require_out_shape(const Tensor& ids, const Tensor& out) {
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("embedding: out must have shape [d,T]");
    }
    if (out.ne[1] != ids.ne[0]) {
        throw std::invalid_argument("embedding: out T dimension must match ids");
    }
}

void require_weight_2d(const Weight& table) {
    if (table.ndim != 2) { throw std::invalid_argument("embedding: table must be 2-D [vocab,d]"); }
    if (table.shape[0] <= 0 || table.shape[1] <= 0) {
        throw std::invalid_argument("embedding: table shape must be positive");
    }
}

void require_dense_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("embedding: BF16_CTRL table must be Contiguous");
    }
    require_weight_2d(table);
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: dense table d must match out.ne[0]");
    }
    if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
        throw std::invalid_argument("embedding: dense table high plane must be null");
    }
    const std::uint64_t expected =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]),
                                        static_cast<std::uint64_t>(table.shape[1])),
                        2);
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: dense payload is too small");
    }
}

void require_q6_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embedding: Q6G64_F16S table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 64 || table.group != 64) {
        throw std::invalid_argument("embedding: Q6G64_F16S table group must be 64");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument("embedding: Q6G64_F16S table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument("embedding: Q6G64_F16S padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: Q6G64_F16S table d must match out.ne[0]");
    }
    const std::uint64_t kg = static_cast<std::uint64_t>(table.padded_shape[1] / 64);
    const std::uint64_t nibble_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 32);
    const std::uint64_t high_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 16);
    const std::uint64_t scale_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 2);
    const std::uint64_t high_plane_off = ((nibble_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t scale_plane_off =
        high_plane_off + ((high_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: Q6G64_F16S payload is too small");
    }
    if (table.qdata == nullptr || table.qhigh == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embedding: Q6G64_F16S planes must be non-null");
    }
    if (table.high_plane_bytes < high_plane_bytes) {
        throw std::invalid_argument("embedding: Q6G64_F16S high plane is too small");
    }
}

void require_w8_metadata(const Weight& table, const Tensor& out) {
    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("embedding: W8G32_F16S table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != 32 || table.group != 32) {
        throw std::invalid_argument("embedding: W8G32_F16S table group must be 32");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument("embedding: W8G32_F16S table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument("embedding: W8G32_F16S padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument("embedding: W8G32_F16S table d must match out.ne[0]");
    }
    const std::uint64_t kg = static_cast<std::uint64_t>(table.padded_shape[1] / 32);
    const std::uint64_t code_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 32);
    const std::uint64_t scale_plane_bytes =
        checked_mul_u64(checked_mul_u64(static_cast<std::uint64_t>(table.shape[0]), kg), 2);
    const std::uint64_t scale_plane_off = ((code_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected        = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument("embedding: W8G32_F16S payload is too small");
    }
    if (table.qdata == nullptr || table.scales == nullptr) {
        throw std::invalid_argument("embedding: W8G32_F16S planes must be non-null");
    }
    if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
        throw std::invalid_argument("embedding: W8G32_F16S high plane must be empty");
    }
}

// Prism ternary tables: group 128, base plane (qs) + optional high plane (qh, PTQ1_0 only) +
// one binary16 scale per group. The plane geometry mirrors row_split_geometry(), whose sizes
// were checked byte for byte against the artifact reader ([248320,5120] -> PTQ1_0 278,118,400 B
// / PQ2_0 337,715,200 B).
void require_ternary_metadata(const Weight& table, const Tensor& out, std::int32_t code_bytes,
                              std::int32_t high_bytes, const char* label) {
    constexpr std::int32_t kGroup = 128;
    const std::string tag         = std::string("embedding: ") + label;

    if (table.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(tag + " table must be RowSplit");
    }
    require_weight_2d(table);
    if (table.group_size != kGroup || table.group != kGroup) {
        throw std::invalid_argument(tag + " table group must be 128");
    }
    if (table.scale_dtype != DType::FP16) {
        throw std::invalid_argument(tag + " table scale dtype must be FP16");
    }
    if (table.padded_shape[0] != table.shape[0] ||
        table.padded_shape[1] != align_up_i32(table.shape[1], 128)) {
        throw std::invalid_argument(tag + " padded shape is invalid");
    }
    if (table.shape[1] != out.ne[0]) {
        throw std::invalid_argument(tag + " table d must match out.ne[0]");
    }

    const std::uint64_t rows = static_cast<std::uint64_t>(table.shape[0]);
    const std::uint64_t kg   = static_cast<std::uint64_t>(table.padded_shape[1] / kGroup);
    const std::uint64_t code_plane_bytes =
        checked_mul_u64(checked_mul_u64(rows, kg), static_cast<std::uint64_t>(code_bytes));
    const std::uint64_t high_plane_bytes =
        checked_mul_u64(checked_mul_u64(rows, kg), static_cast<std::uint64_t>(high_bytes));
    const std::uint64_t scale_plane_bytes = checked_mul_u64(checked_mul_u64(rows, kg), 2);
    const std::uint64_t high_plane_off    = ((code_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t scale_plane_off =
        high_plane_off + ((high_plane_bytes + 255u) / 256u) * 256u;
    const std::uint64_t expected = scale_plane_off + scale_plane_bytes;
    if (table.payload_bytes != 0 && table.payload_bytes < expected) {
        throw std::invalid_argument(tag + " payload is too small");
    }
    if (table.qdata == nullptr || table.scales == nullptr) {
        throw std::invalid_argument(tag + " planes must be non-null");
    }
    if (high_bytes == 0) {
        if (table.qhigh != nullptr || table.high_plane_bytes != 0) {
            throw std::invalid_argument(tag + " high plane must be empty");
        }
    } else {
        if (table.qhigh == nullptr) {
            throw std::invalid_argument(tag + " high plane must be non-null");
        }
        if (table.high_plane_bytes < high_plane_bytes) {
            throw std::invalid_argument(tag + " high plane is too small");
        }
    }
}

void require_fp8_metadata(const Weight& table, const Tensor& out) {
    constexpr std::int32_t kVocabulary = 248320;
    constexpr std::int32_t kHidden     = 5120;
    if (table.n != kVocabulary || table.k != kHidden || out.ne[0] != kHidden) {
        throw std::invalid_argument("embedding: unsupported FP8 table shape");
    }
    if ((reinterpret_cast<std::uintptr_t>(out.data) &
         (static_cast<std::uintptr_t>(alignof(std::uint32_t)) - 1)) != 0) {
        throw std::invalid_argument("embedding: FP8 output must be 4-byte aligned");
    }
    (void)detail::validate_fp8_weight(table, "embedding");
}

bool is_empty_T(const Tensor& ids, const Tensor& out) { return ids.ne[0] == 0 || out.ne[1] == 0; }

void require_non_empty_tensors(const Tensor& ids, const Tensor& out) {
    if (!ids.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("embedding: ids/out must be contiguous");
    }
    if (ids.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("embedding: ids/out data must be non-null");
    }
}

// The Prism ternary embedding table stores its rows folded into the rotated basis, so a gathered
// row is not yet an admissible residual stream: the model computes h = s * (H * z) after the
// lookup, which is the one inverse-mapped transform in the whole model.
//
// Mapping in place costs no scratch buffer, which is what keeps the workspace-free embedding op
// signature intact. With NINFER_TERNARY_HADAMARD=0 the mapping is skipped along with every other
// folded-basis transform.
void unrotate_folded_embedding(Tensor& out, const Weight& table, cudaStream_t stream) {
    if (!detail::ternary_rotation_enabled()) { return; }
    if (!detail::ternary_weight_is_folded(table)) {
        throw std::invalid_argument(
            "embedding: folded ternary table has no sign block; the artifact must carry "
            "text/hadamard_signs and text/hadamard_widths");
    }
    detail::launch_ternary_rotation_inverse_inplace(out, table, stream);
}

// Diagnostic hook: NINFER_TERNARY_DUMP_EMBED=<file> writes the token ids and the FINAL embedding
// activation (after the inverse mapping) so an external oracle can check the whole lookup path
// against the artifact payload. The residual stream enters layer 0 here, so a break at this point
// makes every downstream number meaningless -- which is what a perplexity near uniform looks like.
// Raw BF16 is dumped; the oracle widens it. Output path must be ASCII (a native binary cannot open
// a Chinese path).
void dump_embedding_if_requested(const Tensor& ids, const Tensor& out, cudaStream_t stream,
                                 const char* stage) {
    const char* path = std::getenv("NINFER_TERNARY_DUMP_EMBED");
    if (path == nullptr || *path == '\0') { return; }
    // Graph preparation replays this op while the stream is capturing, and both the device->host
    // copies below and the synchronize that makes the dump meaningful are illegal inside a capture.
    // Skipping capture means the dump lands on the first real replay instead, which is the run whose
    // numbers matter anyway.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return;
    }
    const std::int32_t hidden   = out.ne[0];
    const std::int32_t tokens   = out.ne[1];
    const std::int32_t id_count = static_cast<std::int32_t>(ids.numel());
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(id_count));
    std::vector<std::uint16_t> host_raw(static_cast<std::size_t>(hidden) * tokens);
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), ids.data, host_ids.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_raw.data(), out.data, host_raw.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // One file per (stage, token count), so the gathered rows and the mapped rows can be told apart
    // and a prefill dump is not overwritten by a later decode dump.
    char suffixed[1024];
    std::snprintf(suffixed, sizeof(suffixed), "%s.%s.T%d", path, stage, tokens);
    std::FILE* file = std::fopen(suffixed, "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "embedding: cannot open dump path %s\n", suffixed);
        return;
    }
    const std::int32_t header[3] = {tokens, hidden, id_count};
    std::fwrite(header, sizeof(std::int32_t), 3, file);
    std::fwrite(host_ids.data(), sizeof(std::int32_t), host_ids.size(), file);
    std::fwrite(host_raw.data(), sizeof(std::uint16_t), host_raw.size(), file);
    std::fclose(file);
    std::fprintf(stderr, "embedding: dumped ids=%d hidden=%d tokens=%d -> %s\n", id_count, hidden,
                 tokens, path);
}

} // namespace

void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    if (ids.dtype != DType::I32) { throw std::invalid_argument("embedding: ids must be I32"); }
    if (out.dtype != DType::BF16) { throw std::invalid_argument("embedding: out must be BF16"); }
    if (std::getenv("NINFER_TERNARY_TRACE_EMBED") != nullptr) {
        std::fprintf(stderr,
                     "embedding: qtype=%d ids.ne=[%d,%d] out.ne=[%d,%d] -> axes via ids.ne[0]=%d\n",
                     static_cast<int>(table.qtype), ids.ne[0], ids.ne[1], out.ne[0], out.ne[1],
                     ids.ne[0]);
    }

    (void)numel_allow_zero(ids, "ids");
    (void)numel_allow_zero(out, "out");
    require_ids_shape(ids);
    require_out_shape(ids, out);

    switch (table.qtype) {
    case QType::BF16_CTRL: {
        require_dense_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        if (table.qdata == nullptr) {
            throw std::invalid_argument("embedding: dense table data must be non-null");
        }
        const Tensor dense = as_dense(table);
        detail::embed_gather_dense_launch(ids, dense, out, stream);
    } break;
    case QType::Q6G64_F16S:
        require_q6_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_q6_launch(ids, table, out, stream);
        break;
    case QType::W8G32_F16S:
        require_w8_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_w8_launch(ids, table, out, stream);
        break;
    case QType::FP8_E4M3FN_ROW_BF16S:
        require_fp8_metadata(table, out);
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_fp8_launch(ids, table, out, stream);
        break;
    case QType::PQ2_0_G128:
        require_ternary_metadata(table, out, 32, 0, "PQ2_0_G128");
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_pq2_launch(ids, table, out, stream);
        dump_embedding_if_requested(ids, out, stream, "pre");
        unrotate_folded_embedding(out, table, stream);
        dump_embedding_if_requested(ids, out, stream, "post");
        break;
    case QType::PTQ1_0_G128:
        require_ternary_metadata(table, out, 24, 2, "PTQ1_0_G128");
        if (is_empty_T(ids, out)) { return; }
        require_non_empty_tensors(ids, out);
        detail::embed_gather_ptq1_launch(ids, table, out, stream);
        dump_embedding_if_requested(ids, out, stream, "pre");
        unrotate_folded_embedding(out, table, stream);
        dump_embedding_if_requested(ids, out, stream, "post");
        break;
    default:
        throw std::invalid_argument("embedding: unsupported table qtype");
    }
}

} // namespace ninfer::ops
