#include "models/qwen3_5/program/retrieval/block_retrieval.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace r = ninfer::models::qwen3_5::detail;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

constexpr std::uint32_t kLayers    = 2;
constexpr std::uint32_t kKvHeads   = 2;
constexpr std::uint32_t kHeadDim   = 8;
constexpr std::uint32_t kBlockTok  = 128;

r::RetrievalIndex make_index() { return r::RetrievalIndex(kBlockTok, kLayers, kKvHeads, kHeadDim); }

void test_append_and_truncate() {
    auto index = make_index();
    auto full  = index.append(300);
    expect(index.block_count() == 3 && index.total_tokens() == 300,
           "append splits 300 tokens into three blocks");
    expect(full.size() == 2 && full[0].block_id == 0 && full[1].block_id == 1,
           "append reports exactly the newly full blocks");
    expect(index.block(2).n_tokens == 44 && !index.block(2).full,
           "the trailing partial block holds the remainder");

    index.truncate_to(200);
    expect(index.block_count() == 2 && index.total_tokens() == 200,
           "truncate drops fully-past blocks");

    index.append(56);
    expect(index.block_count() == 2 && index.block(1).n_tokens == 128 && index.block(1).full,
           "re-append refills the shrunk block exactly");
}

void test_scoring() {
    auto index = make_index();
    (void)index.append(256);  // two full blocks

    // Block 0: every head points along +x. Block 1: along -x.
    std::vector<float> mean(index.kv_heads() * index.head_dim(), 0.0F);
    for (std::uint32_t head = 0; head < index.kv_heads(); ++head) {
        mean[head * index.head_dim()] = 1.0F;  // +x on every KV head
    }
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        index.write_block_mean(0, layer, mean);
        std::vector<float> flipped = mean;
        for (std::uint32_t head = 0; head < index.kv_heads(); ++head) {
            flipped[head * index.head_dim()] = -1.0F;
        }
        index.write_block_mean(1, layer, flipped);
    }

    std::vector<float> query(static_cast<std::size_t>(kLayers) * kKvHeads * kHeadDim, 0.0F);
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            query[layer * kKvHeads * kHeadDim + head * kHeadDim] = 1.0F;
        }
    }
    const std::vector<std::uint32_t> counts(kLayers, 7U);

    float aligned = std::numeric_limits<float>::quiet_NaN();
    float opposed = std::numeric_limits<float>::quiet_NaN();
    expect(index.score(0, query, counts, aligned) && aligned > 0.999F,
           "aligned query/block scores near +1");
    expect(index.score(1, query, counts, opposed) && opposed < -0.999F,
           "opposed query/block scores near -1");

    const std::vector<std::uint32_t> zero_counts(kLayers, 0U);
    float unused = 0.0F;
    expect(!index.score(0, query, zero_counts, unused),
           "a layer without query accumulation cannot score");
}

void test_selection() {
    auto index = make_index();
    (void)index.append(6 * kBlockTok);  // six full blocks

    const std::vector<float> scores{0.1F, 0.9F, 0.5F, 0.7F, 0.3F, 0.8F};
    r::BlockSelectionConfig config;
    config.block_tokens  = kBlockTok;
    config.budget_blocks = 3;
    config.sink_blocks   = 1;
    config.recent_blocks = 1;

    const r::BlockSelection selection = r::select_blocks(index, scores, config);
    expect(selection.selected ==
               std::vector<std::uint32_t>({0U, 1U, 5U}),
           "selection keeps sink, recent, and the best remaining score");
    expect(selection.scored_blocks == 4 && selection.mandatory_kept == 0,
           "non-structural blocks are ranked by score");

    const r::BlockSelection mandated =
        r::select_blocks(index, scores, config, std::array<const std::uint32_t, 1>{3U});
    expect(mandated.selected == std::vector<std::uint32_t>({0U, 3U, 5U}),
           "a mandatory block displaces the weakest kept score");
    expect(mandated.mandatory_kept == 1, "mandatory keeps are counted");

    config.budget_blocks = 10;
    const r::BlockSelection everything = r::select_blocks(index, scores, config);
    expect(everything.selected.size() == 6, "an oversized budget selects every block");

    const std::vector<float> with_nan{0.1F, std::numeric_limits<float>::quiet_NaN(),
                                      0.5F, 0.7F, 0.3F, 0.8F};
    const r::BlockSelection skipped = r::select_blocks(index, with_nan, config);
    bool one_absent                 = false;
    for (const std::uint32_t block : skipped.selected) { one_absent |= block == 1U; }
    expect(!one_absent && skipped.scored_blocks == 3, "unscored blocks never win a slot");
}

void test_window_helpers() {
    const auto full = r::prefill_window_page_set(5, 1, 4);
    expect(full == std::vector<std::uint32_t>({0U, 1U, 2U, 3U, 4U}),
           "a covering window keeps every page");

    const auto windowed = r::prefill_window_page_set(20, 1, 4);
    expect(windowed == std::vector<std::uint32_t>({0U, 16U, 17U, 18U, 19U}),
           "a rolling window keeps the sink prefix and newest pages");

    expect(r::prefill_window_page_set(0, 1, 4).empty(), "an empty prefix has no window");

    const auto pages = r::block_pages(std::vector<std::uint32_t>{0U, 2U}, 128);
    expect(pages == std::vector<std::uint32_t>({0U, 1U, 4U, 5U}),
           "blocks expand to their 64-token pages");
}

} // namespace

int main() {
    try {
        test_append_and_truncate();
        test_scoring();
        test_selection();
        test_window_helpers();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
