// Incremental host-encode test. Ported and adapted from dylanbrodiefafard's 48d1857
// ("incremental host encode") to this tree's evolved frontend API:
//  - Frontend::prepare/count_tokens take an EncodedHistoryCache*; the observation travels
//    through the thread-local last_host_encode_observation instead of a prompt field.
//  - try_splice_encoded_chat consumes a RenderedChat plus stored committed-region boundary
//    results instead of a bare checkpoint spec.
//  - The suite runs on the toy 256-byte tokenizer (no BPE merges), so the upstream
//    marked-encode prefix_tokens oracle is replaced by is_encode_loop_pos plus the concat
//    property; the official-tokenizer-only sections (BPE merge concat inequality, official
//    per-byte sweep, prepare bench) run only when NINFER_OFFICIAL_TOKENIZER_DIR provides the
//    tokenizer files.
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/encoded_history_cache.h"
#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "text/unicode.h"

// Compat MSVC: headers Windows (vía el engine) definen macros que rompen
// identificadores de este test: `near` (windef.h, memoria 16-bit) y
// `small` (rpcndr.h, NDR: #define small char).
#undef near
#undef small

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

using Frontend              = ninfer::targets::qwen3_6::Frontend;
using FrontendFactory       = ninfer::targets::qwen3_6::FrontendTestAccess;
using FrontendResources     = ninfer::targets::qwen3_6::FrontendResources;
using EncodedHistoryPrepare = ninfer::targets::qwen3_6::EncodedHistoryPrepare;
namespace fi                = ninfer::targets::qwen3_6::frontend_internal;

inline constexpr std::size_t kNoTokenLimit = std::numeric_limits<std::size_t>::max();

// setenv/unsetenv are not declared by this toolchain's C++ headers; _putenv_s (value nullptr
// removes) is the CRT-internal equivalent and is visible to getenv.
#if defined(_WIN32)
void set_test_env(const char* name, const char* value) { _putenv_s(name, value); }
#else
void set_test_env(const char* name, const char* value) {
    if (value == nullptr) { unsetenv(name); }
    else { setenv(name, value, 1); }
}
#endif

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string read_template_fixture(const char* path) {
    std::string source = read_file(path);
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

const std::string& thinking_toggle_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    return source;
}

const std::string& reasoning_effort_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/reasoning_effort_chat_template.jinja");
    return source;
}

const fi::CompiledChatTemplate& thinking_toggle_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(thinking_toggle_template_source());
    return value;
}

const fi::CompiledChatTemplate& reasoning_effort_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(reasoning_effort_template_source());
    return value;
}

bool official_tokenizer_available() {
    const char* env = std::getenv("NINFER_OFFICIAL_TOKENIZER_DIR");
    if (env == nullptr || env[0] == '\0') { return false; }
    std::ifstream stream(std::string(env) + "/tokenizer.json", std::ios::binary);
    return static_cast<bool>(stream);
}

const std::string& official_tokenizer_dir() {
    static const std::string tokenizer_dir = [] {
        const char* env = std::getenv("NINFER_OFFICIAL_TOKENIZER_DIR");
        if (env == nullptr || env[0] == '\0') {
            throw std::runtime_error("official tokenizer.json was not found");
        }
        return std::string(env);
    }();
    return tokenizer_dir;
}

const fi::Tokenizer& official_tokenizer() {
    static const std::string tokenizer_json =
        read_file((official_tokenizer_dir() + "/tokenizer.json").c_str());
    static const std::string tokenizer_config_json =
        read_file((official_tokenizer_dir() + "/tokenizer_config.json").c_str());
    static const std::string generation_config_json =
        read_file((official_tokenizer_dir() + "/generation_config.json").c_str());
    static const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                          .tokenizer_config_json  = tokenizer_config_json,
                                          .generation_config_json = generation_config_json});
    return tokenizer;
}

constexpr std::string_view kThinkingControlGuidance =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n";

constexpr ninfer::TokenId kFixtureByteTokenBase = 1'000;

constexpr ninfer::TokenId fixture_byte_token(std::uint8_t byte) {
    // Preserve IDs already used by the output-session fixtures. All other bytes live outside the
    // added-token range so the synthetic tokenizer can encode arbitrary UTF-8 test input.
    switch (byte) {
    case static_cast<std::uint8_t>('x'):
        return 0;
    case 0xe4:
        return 10;
    case 0xb8:
        return 11;
    case 0xad:
        return 12;
    case 0x80:
        return 13;
    case 0xe0:
        return 14;
    case 0xed:
        return 15;
    case 0xa0:
        return 16;
    case 0xf4:
        return 17;
    case 0x90:
        return 18;
    case 0xf5:
        return 19;
    case 0xf0:
        return 20;
    case 0x9f:
        return 21;
    case 0x98:
        return 22;
    case 0xc2:
        return 23;
    case 0xa2:
        return 24;
    default:
        return kFixtureByteTokenBase + byte;
    }
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(8, std::string(kThinkingControlGuidance)), added(30, "user\n"),
         added(31, "assistant\n"), added(32, "\n"), added(248045, "<|im_start|>", true),
         added(248046, "<|im_end|>", true), added(248053, "<|vision_start|>", true),
         added(248054, "<|vision_end|>", true), added(248056, "<|image_pad|>", true),
         added(248057, "<|video_pad|>", true), added(248068, "<think>"),
         added(248069, "</think>")});
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        const auto byte                = static_cast<std::uint8_t>(value);
        vocab[byte_level_symbol(byte)] = fixture_byte_token(byte);
    }
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"}, {"vocab", std::move(vocab)}, {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

const fi::Tokenizer& fixture_tokenizer() {
    static const FrontendResources fixture = resources();
    static const fi::Tokenizer tokenizer(
        {.tokenizer_json         = fixture.tokenizer_json,
         .tokenizer_config_json  = fixture.tokenizer_config_json,
         .generation_config_json = fixture.generation_config_json});
    return tokenizer;
}

FrontendResources official_frontend_resources(const std::string& chat_template) {
    FrontendResources result     = resources(chat_template);
    result.tokenizer_json        = read_file((official_tokenizer_dir() + "/tokenizer.json").c_str());
    nlohmann::json config        = nlohmann::json::parse(
        read_file((official_tokenizer_dir() + "/tokenizer_config.json").c_str()));
    config["chat_template"]      = chat_template;
    result.tokenizer_config_json = config.dump();
    result.generation_config_json =
        read_file((official_tokenizer_dir() + "/generation_config.json").c_str());
    return result;
}

fi::ChatMessage chat_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

ninfer::ChatMessage product_message(ninfer::ChatRole role, std::string content) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(content), .media = {}});
    return message;
}

ninfer::PromptInput product_input(std::vector<ninfer::ChatMessage> messages,
                                  ninfer::PromptOptions options = {}) {
    ninfer::PromptInput input;
    input.messages = std::move(messages);
    input.options  = std::move(options);
    return input;
}

fi::RenderedChat text_rendered(std::string text) {
    fi::RenderedChat rendered;
    rendered.text = std::move(text);
    return rendered;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

// The stored committed-region boundary map for a splice at n: every rendered text boundary
// below n, encoded against the committed prefix (results are properties of those bytes alone).
std::vector<fi::CommittedBoundary> committed_boundaries_for(const fi::Tokenizer& tokenizer,
                                                            const fi::RenderedChat& rendered,
                                                            std::size_t n) {
    std::vector<std::size_t> offsets;
    fi::append_rendered_text_boundaries(rendered, offsets);
    std::vector<std::size_t> kept;
    for (const std::size_t offset : offsets) {
        if (offset < n) { kept.push_back(offset); }
    }
    std::vector<fi::CommittedBoundary> result;
    if (kept.empty()) { return result; }
    const fi::BoundaryEncodedText marked =
        tokenizer.encode_with_boundaries(rendered.text.substr(0, n), kept);
    result.reserve(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
        result.push_back(fi::CommittedBoundary{.offset = kept[i], .result = marked.boundaries[i]});
    }
    std::stable_sort(result.begin(), result.end(),
                     [](const fi::CommittedBoundary& lhs, const fi::CommittedBoundary& rhs) {
                         return lhs.offset < rhs.offset;
                     });
    std::vector<fi::CommittedBoundary> unique;
    unique.reserve(result.size());
    for (const fi::CommittedBoundary& boundary : result) {
        if (unique.empty() || unique.back().offset != boundary.offset) {
            unique.push_back(boundary);
        }
    }
    return unique;
}

// At a conservative encode-loop position the committed and suffix token streams join without
// re-segmentation: encode(prefix) ++ encode(suffix) == encode(full).
int check_loop_pos_concat(const fi::Tokenizer& tokenizer, std::string_view text,
                          const char* message) {
    int failures = 0;
    const std::vector<int> cold_ids = tokenizer.encode(text);
    for (std::size_t n = 1; n < text.size(); ++n) {
        if (!tokenizer.is_encode_loop_pos(text, n)) { continue; }
        const std::vector<int> prefix = tokenizer.encode(text.substr(0, n));
        const std::vector<int> suffix = tokenizer.encode(text.substr(n));
        std::vector<int> concat       = prefix;
        concat.insert(concat.end(), suffix.begin(), suffix.end());
        if (concat != cold_ids) {
            std::cerr << message << " n=" << n << " size=" << text.size()
                      << ": concat of a legal cut differs from the cold encode\n";
            ++failures;
            break;
        }
    }
    return failures;
}

void boundary_fail(int& failures, const char* label, std::size_t n, std::size_t size,
                   const char* detail) {
    std::cerr << label << " n=" << n << " size=" << size << ": " << detail << '\n';
    ++failures;
}

int check_every_byte_boundary(const fi::Tokenizer& tokenizer, std::string_view text,
                              const std::optional<fi::RewriteCheckpointByteSpec>& checkpoint,
                              const char* label) {
    if (text.empty()) {
        std::cerr << label << ": empty text\n";
        return 1;
    }
    int failures                  = 0;
    fi::RenderedChat rendered;
    rendered.text                 = std::string(text);
    rendered.rewrite_checkpoint   = checkpoint;
    const fi::EncodedChat cold    = fi::encode_rendered_chat(tokenizer, rendered);
    const std::optional<std::size_t> checkpoint_offset =
        checkpoint ? std::optional<std::size_t>{checkpoint->offset} : std::nullopt;
    std::size_t legal_cuts = 0;

    for (std::size_t n = 0; n <= text.size(); ++n) {
        const bool legal = tokenizer.is_encode_loop_pos(text, n);
        if (legal) { ++legal_cuts; }
        const bool checkpoint_blocks = checkpoint_offset && *checkpoint_offset < n;
        if (!legal) {
            if (fi::try_splice_encoded_chat(tokenizer, std::vector<int>{1}, rendered, n, {},
                                            kNoTokenLimit)) {
                boundary_fail(failures, label, n, text.size(),
                              "illegal tokenizer cut accepted a splice");
                return failures;
            }
            if (n == 0) { continue; }
            fi::EncodedHistoryCache cache;
            cache.insert_committed(std::string(text.substr(0, n)), {1}, {});
            if (cache.copy_longest_prefix(text, tokenizer, checkpoint_offset)) {
                boundary_fail(failures, label, n, text.size(),
                              "illegal exact byte prefix was used as a cache hit");
                return failures;
            }
            continue;
        }

        if (checkpoint_blocks) {
            if (fi::try_splice_encoded_chat(tokenizer, std::vector<int>{1}, rendered, n, {},
                                            kNoTokenLimit)) {
                boundary_fail(failures, label, n, text.size(),
                              "splice accepted when checkpoint.offset < n");
                return failures;
            }
            continue;
        }

        std::vector<int> prefix_ids;
        if (n > 0) {
            prefix_ids = tokenizer.encode(text.substr(0, n));
            const std::vector<int> suffix = tokenizer.encode(text.substr(n));
            std::vector<int> concat       = prefix_ids;
            concat.insert(concat.end(), suffix.begin(), suffix.end());
            if (concat != cold.input_ids) {
                boundary_fail(failures, label, n, text.size(),
                              "concat of a legal cut differs from cold input_ids");
                return failures;
            }
        }
        const std::vector<fi::CommittedBoundary> no_boundaries{};
        const auto spliced =
            fi::try_splice_encoded_chat(tokenizer, prefix_ids, rendered, n, no_boundaries,
                                        kNoTokenLimit);
        if (!spliced) {
            boundary_fail(failures, label, n, text.size(), "legal tokenizer cut refused splice");
            return failures;
        }
        if (spliced->input_ids != cold.input_ids) {
            boundary_fail(failures, label, n, text.size(), "spliced ids differ from cold encode");
            return failures;
        }
        if (spliced->rewrite_checkpoint != cold.rewrite_checkpoint) {
            boundary_fail(failures, label, n, text.size(), "spliced frontier differs from cold");
            return failures;
        }

        if (n == 0) { continue; }
        fi::EncodedHistoryCache cache;
        cache.insert_committed(std::string(text.substr(0, n)), prefix_ids, {});
        const auto hit = cache.copy_longest_prefix(text, tokenizer, checkpoint_offset);
        if (!hit || hit->bytes.size() != n || hit->ids != prefix_ids) {
            boundary_fail(failures, label, n, text.size(),
                          "exact legal byte prefix was not the cache hit");
            return failures;
        }
        std::string flipped(text.substr(0, n));
        flipped.front() = static_cast<char>(static_cast<unsigned char>(flipped.front()) ^ 0xff);
        fi::EncodedHistoryCache scrambled;
        scrambled.insert_committed(std::move(flipped), prefix_ids, {});
        if (scrambled.copy_longest_prefix(text, tokenizer, checkpoint_offset)) {
            boundary_fail(failures, label, n, text.size(),
                          "one-byte mismatch still produced a cache hit");
            return failures;
        }
    }
    if (legal_cuts < 2) {
        std::cerr << label << ": expected at least cuts at 0 and size, got " << legal_cuts << '\n';
        ++failures;
    }
    return failures;
}

int check_special_token_interiors(const fi::Tokenizer& tokenizer, const char* token) {
    const std::string text = token;
    int failures           = 0;
    for (std::size_t n = 1; n < text.size(); ++n) {
        const fi::RenderedChat rendered = text_rendered(text);
        if (tokenizer.is_encode_loop_pos(text, n) ||
            fi::try_splice_encoded_chat(tokenizer, std::vector<int>{1}, rendered, n, {},
                                        kNoTokenLimit)) {
            boundary_fail(failures, token, n, text.size(),
                          "interior of an added special token was treated as a legal cut");
            return failures;
        }
        fi::EncodedHistoryCache cache;
        cache.insert_committed(text.substr(0, n), {1}, {});
        if (cache.copy_longest_prefix(text, tokenizer, std::nullopt)) {
            boundary_fail(failures, token, n, text.size(),
                          "interior special-token bytes were used as a cache hit");
            return failures;
        }
    }
    return failures;
}

int check_splice_matches_cold(const fi::Tokenizer& tokenizer, const fi::RenderedChat& committed,
                              const fi::RenderedChat& full, const char* message) {
    const std::string label = message;
    int failures =
        check(full.text.starts_with(committed.text),
              (label + ": full does not start with committed").c_str());
    if (failures != 0) { return failures; }
    const std::size_t n = committed.text.size();
    failures += check(tokenizer.is_encode_loop_pos(full.text, n),
                      (label + ": committed size is not a loop-pos of full").c_str());
    const std::vector<int> committed_ids = tokenizer.encode(committed.text);
    const std::vector<fi::CommittedBoundary> committed_boundaries =
        committed_boundaries_for(tokenizer, full, n);
    const auto spliced = fi::try_splice_encoded_chat(tokenizer, committed_ids, full, n,
                                                     committed_boundaries, kNoTokenLimit);
    const fi::EncodedChat cold = fi::encode_rendered_chat(tokenizer, full);
    failures += check(spliced.has_value(), (label + ": splice refused").c_str());
    if (!spliced) { return failures; }
    failures += check(spliced->input_ids == cold.input_ids,
                      (label + ": spliced ids differ from cold encode").c_str());
    failures += check(spliced->rewrite_checkpoint == cold.rewrite_checkpoint,
                      (label + ": spliced frontier differs from cold encode").c_str());
    return failures;
}

struct CachedCall {
    ninfer::targets::qwen3_6::PreparedPrompt prompt;
    fi::HostEncodeObservation observation;
};

CachedCall cached_prepare(const Frontend& frontend, fi::EncodedHistoryCache& cache,
                          ninfer::PromptInput input) {
    auto prompt = EncodedHistoryPrepare::prepare(frontend, std::move(input), cache);
    return CachedCall{.prompt = std::move(prompt),
                      .observation = fi::last_host_encode_observation};
}

int expect_match_cold(const Frontend& frontend, const CachedCall& call,
                      ninfer::PromptInput input, bool expect_hit, const char* message) {
    const std::string label = message;
    const auto& got         = FrontendFactory::inspect(call.prompt);
    const auto cold         = frontend.prepare(std::move(input));
    const auto& cold_data   = FrontendFactory::inspect(cold);
    int failures =
        check(call.observation.cache_hit == expect_hit,
              (label + ": unexpected hit flag").c_str());
    failures += check(got.token_ids == cold_data.token_ids,
                      (label + ": ids differ from cold Frontend::prepare").c_str());
    failures += check(got.identity.rewrite_checkpoint == cold_data.identity.rewrite_checkpoint,
                      (label + ": frontier differs from cold prepare").c_str());
    failures += check(got.starts_in_reasoning == cold_data.starts_in_reasoning,
                      (label + ": starts_in_reasoning differs").c_str());
    return failures;
}

bool same_observation(const fi::HostEncodeObservation& a, const fi::HostEncodeObservation& b) {
    return a.cache_hit == b.cache_hit && a.attempted_prefix == b.attempted_prefix &&
           a.verified_mismatch == b.verified_mismatch && a.inserted == b.inserted &&
           a.prefix_bytes == b.prefix_bytes;
}

int test_loop_pos_and_splice() {
    const fi::Tokenizer& tokenizer = fixture_tokenizer();
    int failures                   = 0;

    const std::vector<fi::ChatMessage> user_only{chat_message(ninfer::ChatRole::User, "hello")};
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat committed = thinking_toggle_template().render(user_only, no_generation);
    const fi::RenderedChat full      = thinking_toggle_template().render(user_only, {});
    failures += check(full.text.starts_with(committed.text) &&
                          full.text.substr(committed.text.size()) ==
                              "<|im_start|>assistant\n<think>\n",
                      "thinking-toggle generation suffix is not only the opener");
    failures += check_loop_pos_concat(tokenizer, full.text, "thinking-toggle full loop-pos");
    failures +=
        check_splice_matches_cold(tokenizer, committed, full, "P1-H1/H2/H5 thinking toggle");

    fi::ChatRenderOptions no_think_gen;
    no_think_gen.enable_thinking        = false;
    const fi::RenderedChat no_think_full = thinking_toggle_template().render(user_only, no_think_gen);
    failures += check(no_think_full.text.starts_with(committed.text) &&
                          no_think_full.text.substr(committed.text.size()) ==
                              "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "thinking-toggle disable suffix is not only the opener");
    failures +=
        check_splice_matches_cold(tokenizer, committed, no_think_full, "P1-H8 thinking toggle flip");

    const std::vector<fi::ChatMessage> effort_user{chat_message(ninfer::ChatRole::User, "hello")};
    fi::ChatRenderOptions effort_medium;
    effort_medium.reasoning_effort      = ninfer::ReasoningEffort::Medium;
    fi::ChatRenderOptions effort_committed = effort_medium;
    effort_committed.add_generation_prompt = false;
    const fi::RenderedChat effort_c =
        reasoning_effort_template().render(effort_user, effort_committed);
    const fi::RenderedChat effort_f = reasoning_effort_template().render(effort_user, effort_medium);
    failures += check(effort_f.text.starts_with(effort_c.text),
                      "reasoning-effort medium full does not start with committed");
    failures += check_splice_matches_cold(tokenizer, effort_c, effort_f, "P1-H6/P1-T2 effort medium");

    fi::ChatRenderOptions effort_xhigh;
    effort_xhigh.reasoning_effort            = ninfer::ReasoningEffort::XHigh;
    fi::ChatRenderOptions effort_xhigh_committed = effort_xhigh;
    effort_xhigh_committed.add_generation_prompt = false;
    const fi::RenderedChat xhigh_c =
        reasoning_effort_template().render(effort_user, effort_xhigh_committed);
    const fi::RenderedChat xhigh_f = reasoning_effort_template().render(effort_user, effort_xhigh);
    failures += check(xhigh_f.text.starts_with(xhigh_c.text),
                      "reasoning-effort xhigh full does not start with committed");
    failures += check_splice_matches_cold(tokenizer, xhigh_c, xhigh_f, "P1-T3 effort xhigh");

    const std::vector<fi::ChatMessage> late{
        chat_message(ninfer::ChatRole::System, "stable policy"),
        chat_message(ninfer::ChatRole::User, "hi"),
        chat_message(ninfer::ChatRole::System, "current diagnostics")};
    const std::vector<fi::ChatMessage> stable{
        chat_message(ninfer::ChatRole::System, "stable policy"),
        chat_message(ninfer::ChatRole::User, "hi")};
    const fi::RenderedChat stable_c = thinking_toggle_template().render(stable, no_generation);
    const fi::RenderedChat late_c   = thinking_toggle_template().render(late, no_generation);
    failures += check_splice_matches_cold(tokenizer, stable_c, late_c, "P1-H3 late system");

    const std::vector<fi::ChatMessage> cjk{chat_message(
        ninfer::ChatRole::User, "缓存复用 café and NFC")};
    const fi::RenderedChat cjk_c = thinking_toggle_template().render(cjk, no_generation);
    const fi::RenderedChat cjk_f = thinking_toggle_template().render(cjk, {});
    failures += check_splice_matches_cold(tokenizer, cjk_c, cjk_f, "P1-H4 CJK/NFC after im_end");

    failures += check(tokenizer.is_encode_loop_pos(committed.text, committed.text.size()),
                      "P1-H7 replay n==size is not a loop-pos");
    const auto replay = fi::try_splice_encoded_chat(
        tokenizer, tokenizer.encode(committed.text), committed, committed.text.size(), {},
        kNoTokenLimit);
    failures += check(replay && replay->input_ids == tokenizer.encode(committed.text) &&
                          !replay->rewrite_checkpoint,
                      "P1-H7 empty suffix replay failed");

    fi::RewriteCheckpointByteSpec at_committed{
        .kind   = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
        .offset = committed.text.size()};
    fi::RenderedChat full_at_committed = full;
    full_at_committed.rewrite_checkpoint = at_committed;
    const auto rel0 = fi::try_splice_encoded_chat(
        tokenizer, tokenizer.encode(committed.text), full_at_committed, committed.text.size(), {},
        kNoTokenLimit);
    failures += check(rel0 && rel0->rewrite_checkpoint &&
                          rel0->rewrite_checkpoint->frontier ==
                              tokenizer.encode(committed.text).size(),
                      "P1-H9 rel==0 frontier is not committed id count");

    const std::string hello = "hello";
    const fi::RenderedChat hello_rendered = text_rendered(hello);
    failures += check(!tokenizer.is_encode_loop_pos(hello, 2), "P1-R1 mid-word is a loop-pos");
    failures += check(!fi::try_splice_encoded_chat(tokenizer, tokenizer.encode("he"),
                                                   hello_rendered, 2, {}, kNoTokenLimit),
                      "P1-R1 mid-word splice was accepted");

    const std::string im_end = "<|im_end|>";
    failures += check(!tokenizer.is_encode_loop_pos(im_end, 3), "P1-R2 split im_end is a loop-pos");
    failures += check(!fi::try_splice_encoded_chat(
                          tokenizer, tokenizer.encode(im_end.substr(0, 3)), text_rendered(im_end),
                          3, {}, kNoTokenLimit),
                      "P1-R2 split im_end splice was accepted");

    fi::ChatRenderOptions no_thinking;
    no_thinking.enable_thinking = false;
    const std::vector<fi::ChatMessage> thought{
        chat_message(ninfer::ChatRole::User, "q1"),
        chat_message(ninfer::ChatRole::Assistant, "<think>\nold thought\n</think>\n\nold answer"),
        chat_message(ninfer::ChatRole::User, "q2")};
    const fi::RenderedChat with_think = thinking_toggle_template().render(
        {chat_message(ninfer::ChatRole::User, "q1"),
         chat_message(ninfer::ChatRole::Assistant, "<think>\nold thought\n</think>\n\nold answer")},
        no_generation);
    const fi::RenderedChat stripped = thinking_toggle_template().render(thought, no_thinking);
    failures += check(!stripped.text.starts_with(with_think.text),
                      "P1-R6 think-strip is still a byte prefix");

    fi::ChatMessage lookup = chat_message(ninfer::ChatRole::Assistant, "");
    lookup.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    const fi::RenderedChat one_tool = thinking_toggle_template().render(
        {chat_message(ninfer::ChatRole::User, "weather?"), lookup,
         chat_message(ninfer::ChatRole::Tool, "sunny")},
        no_generation);
    const fi::RenderedChat two_tools = thinking_toggle_template().render(
        {chat_message(ninfer::ChatRole::User, "weather?"), lookup,
         chat_message(ninfer::ChatRole::Tool, "sunny"),
         chat_message(ninfer::ChatRole::Tool, "20C")},
        no_generation);
    failures += check(!two_tools.text.starts_with(one_tool.text),
                      "P1-R7 tool-group growth is still a byte prefix");

    fi::ChatRenderOptions effort_off;
    effort_off.enable_thinking = false;
    const fi::RenderedChat effort_on_c =
        reasoning_effort_template().render(effort_user, effort_xhigh_committed);
    const fi::RenderedChat effort_off_full =
        reasoning_effort_template().render(effort_user, effort_off);
    failures += check(!effort_off_full.text.starts_with(effort_on_c.text),
                      "P1-R10 effort enable_thinking flip is still a prefix");

    auto toy_with_added = [](std::initializer_list<std::pair<int, const char*>> extra) {
        FrontendResources overlap_res = resources();
        nlohmann::json tok            = nlohmann::json::parse(overlap_res.tokenizer_json);
        nlohmann::json cfg            = nlohmann::json::parse(overlap_res.tokenizer_config_json);
        for (const auto& [id, content] : extra) {
            tok["added_tokens"].push_back(added(id, content));
            cfg["added_tokens_decoder"][std::to_string(id)] = decoder_added(content);
        }
        overlap_res.tokenizer_json        = tok.dump();
        overlap_res.tokenizer_config_json = cfg.dump();
        return fi::Tokenizer({.tokenizer_json         = overlap_res.tokenizer_json,
                              .tokenizer_config_json  = overlap_res.tokenizer_config_json,
                              .generation_config_json = overlap_res.generation_config_json});
    };
    const fi::Tokenizer local_bc = toy_with_added({{25, "AB"}, {26, "BC"}});
    failures += check(!local_bc.is_encode_loop_pos("ABC", 1),
                      "P1-R3 local BC start at n=1 is a loop-pos");
    const std::vector<int> ab = local_bc.encode("AB");
    failures += check(!fi::try_splice_encoded_chat(local_bc, ab, text_rendered("ABC"), 1, {},
                                                   kNoTokenLimit),
                      "P1-R3 local special probe splice was accepted");

    const fi::Tokenizer longer_first = toy_with_added({{25, "abc"}, {26, "ab"}});
    failures += check(!longer_first.is_encode_loop_pos("abcd", 2),
                      "P1-R4 interior of longer added token is a loop-pos");
    failures += check(longer_first.is_encode_loop_pos("abcd", 3),
                      "P1-R4 end of winning abc is not a loop-pos");
    failures += check(!fi::try_splice_encoded_chat(longer_first, std::vector<int>{26},
                                                   text_rendered("abcd"), 2, {}, kNoTokenLimit),
                      "P1-R4 splice at interior of abc was accepted");
    const fi::Tokenizer shorter_first = toy_with_added({{25, "ab"}, {26, "abc"}});
    failures += check(!shorter_first.is_encode_loop_pos("abcd", 3),
                      "P1-R4 leftover-run offset after shorter added token is a loop-pos");
    failures += check(!fi::try_splice_encoded_chat(shorter_first, std::vector<int>{25},
                                                   text_rendered("abcd"), 3, {}, kNoTokenLimit),
                      "P1-R4 splice in leftover run was accepted");

    fi::EncodeOptions no_added;
    no_added.parse_added_tokens = false;
    failures += check(tokenizer.is_encode_loop_pos(full.text, 0, no_added) &&
                          tokenizer.is_encode_loop_pos(full.text, full.text.size(), no_added) &&
                          !tokenizer.is_encode_loop_pos(full.text, committed.text.size(), no_added),
                      "P1-R8 parse_added_tokens=false treated an interior added-token cut as legal");

    const std::string cjk_char = "复";
    failures += check(cjk_char.size() > 1 && !tokenizer.is_encode_loop_pos(cjk_char, 1),
                      "P1-R9 mid UTF-8 codepoint is a loop-pos");
    const std::string combining = "é";
    failures += check(combining.size() > 1 && !tokenizer.is_encode_loop_pos(combining, 1),
                      "P1-R9 mid NFC combining sequence is a loop-pos");

    fi::RewriteCheckpointByteSpec before_committed{
        .kind   = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
        .offset = 1};
    fi::RenderedChat full_before = full;
    full_before.rewrite_checkpoint = before_committed;
    failures += check(committed.text.size() > 1 &&
                          !fi::try_splice_encoded_chat(
                              tokenizer, tokenizer.encode(committed.text), full_before,
                              committed.text.size(), {}, kNoTokenLimit),
                      "P1-R12 checkpoint offset < n was accepted");

    fi::RewriteCheckpointByteSpec inside_opener{
        .kind   = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
        .offset = committed.text.size() + 1};
    fi::RenderedChat full_inside = full;
    full_inside.rewrite_checkpoint = inside_opener;
    failures += check(!fi::try_splice_encoded_chat(tokenizer, tokenizer.encode(committed.text),
                                                   full_inside, committed.text.size(), {},
                                                   kNoTokenLimit),
                      "E-M11 non-exact checkpoint beyond n was accepted");
    return failures;
}

// The post-upstream feature: stored committed-region boundary results let a splice reproduce
// every frontier of the cold encode, not just the token ids.
int test_committed_boundary_reuse() {
    const fi::Tokenizer& tokenizer = fixture_tokenizer();
    int failures                   = 0;
    const std::vector<fi::ChatMessage> messages{
        chat_message(ninfer::ChatRole::User, "q1"),
        chat_message(ninfer::ChatRole::Assistant, "a1"),
        chat_message(ninfer::ChatRole::User, "q2")};
    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking = true;
    const fi::RenderedChat rendered = thinking_toggle_template().render(messages, preserve);
    const fi::EncodedChat cold      = fi::encode_rendered_chat(tokenizer, rendered);

    const auto splice_case = [&](std::size_t n, const char* label) {
        const std::string prefix = label;
        failures += check(tokenizer.is_encode_loop_pos(rendered.text, n),
                          (prefix + ": cut is not a loop-pos").c_str());
        const std::vector<int> committed_ids = tokenizer.encode(rendered.text.substr(0, n));
        const std::vector<fi::CommittedBoundary> stored =
            committed_boundaries_for(tokenizer, rendered, n);
        const auto spliced =
            fi::try_splice_encoded_chat(tokenizer, committed_ids, rendered, n, stored,
                                        kNoTokenLimit);
        failures += check(spliced && *spliced == cold,
                          (prefix + ": spliced EncodedChat differs from cold").c_str());
    };

    if (rendered.rewrite_checkpoint) {
        splice_case(rendered.rewrite_checkpoint->offset, "B1 checkpoint cut");
    } else {
        failures += check(false, "B1 render carries no rewrite checkpoint");
    }
    if (rendered.message_boundaries.size() > 1 && rendered.message_boundaries[1]) {
        splice_case(*rendered.message_boundaries[1], "B2 first message cut");
    } else {
        failures += check(false, "B2 render carries no first message boundary");
    }
    return failures;
}

int test_engine_shaped_cache() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    const Frontend effort_frontend =
        FrontendFactory::create_component(resources(reasoning_effort_template_source()));
    fi::EncodedHistoryCache cache;
    int failures = 0;

    ninfer::PromptOptions preserve;
    preserve.preserve_thinking = true;

    auto first = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "hello")}, preserve));
    failures += expect_match_cold(
        frontend, first,
        product_input({product_message(ninfer::ChatRole::User, "hello")}, preserve), false,
        "E-M8 first prepare");
    failures += check(cache.size() == 1, "first prepare did not insert committed");

    auto second = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "hello"),
                       product_message(ninfer::ChatRole::Assistant, "hi there"),
                       product_message(ninfer::ChatRole::User, "and more")},
                      preserve));
    failures += expect_match_cold(
        frontend, second,
        product_input({product_message(ninfer::ChatRole::User, "hello"),
                       product_message(ninfer::ChatRole::Assistant, "hi there"),
                       product_message(ninfer::ChatRole::User, "and more")},
                      preserve),
        true, "E-H1 append user");

    auto replay = cached_prepare(
        frontend, cache, product_input({product_message(ninfer::ChatRole::User, "hello")}, preserve));
    failures += expect_match_cold(
        frontend, replay,
        product_input({product_message(ninfer::ChatRole::User, "hello")}, preserve), true,
        "E-H2 same M twice");

    (void)EncodedHistoryPrepare::count_tokens(
        frontend, product_input({product_message(ninfer::ChatRole::User, "count me")}, preserve),
        cache);
    auto after_count = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "count me")}, preserve));
    failures += expect_match_cold(
        frontend, after_count,
        product_input({product_message(ninfer::ChatRole::User, "count me")}, preserve), true,
        "E-H3 count_tokens then prepare");

    const std::uint32_t counted_ext = EncodedHistoryPrepare::count_tokens(
        frontend,
        product_input({product_message(ninfer::ChatRole::User, "count me"),
                       product_message(ninfer::ChatRole::Assistant, "ok"),
                       product_message(ninfer::ChatRole::User, "next")},
                      preserve),
        cache);
    failures += check(
        fi::last_host_encode_observation.cache_hit &&
            counted_ext ==
                frontend.count_tokens(product_input(
                    {product_message(ninfer::ChatRole::User, "count me"),
                     product_message(ninfer::ChatRole::Assistant, "ok"),
                     product_message(ninfer::ChatRole::User, "next")},
                    preserve)),
        "E-H4 prepare then count_tokens of an extension missed or disagreed");

    auto late = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::System, "stable policy"),
                       product_message(ninfer::ChatRole::User, "hi")},
                      preserve));
    failures += expect_match_cold(
        frontend, late,
        product_input({product_message(ninfer::ChatRole::System, "stable policy"),
                       product_message(ninfer::ChatRole::User, "hi")},
                      preserve),
        false, "late system first");
    auto late2 = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::System, "stable policy"),
                       product_message(ninfer::ChatRole::User, "hi"),
                       product_message(ninfer::ChatRole::System, "current diagnostics")},
                      preserve));
    failures += expect_match_cold(
        frontend, late2,
        product_input({product_message(ninfer::ChatRole::System, "stable policy"),
                       product_message(ninfer::ChatRole::User, "hi"),
                       product_message(ninfer::ChatRole::System, "current diagnostics")},
                      preserve),
        true, "E-H5 late system append");

    ninfer::PromptOptions think_on  = preserve;
    ninfer::PromptOptions think_off = preserve;
    think_off.enable_thinking       = false;
    (void)cached_prepare(frontend, cache,
                         product_input({product_message(ninfer::ChatRole::User, "toggle")},
                                       think_on));
    auto flipped = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "toggle")}, think_off));
    failures += expect_match_cold(
        frontend, flipped,
        product_input({product_message(ninfer::ChatRole::User, "toggle")}, think_off), true,
        "E-H6 ThinkingToggle enable_thinking flip");
    failures += check(
        FrontendFactory::inspect(flipped.prompt).starts_in_reasoning == false,
        "E-H6 starts_in_reasoning did not follow this call");

    // Assistant continuation replays the final response: the render carries a ResponseReplay
    // checkpoint inside the committed region, so the call's own entry cannot be reused.
    ninfer::PromptOptions cfa;
    cfa.continuation     = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    cfa.enable_thinking  = false;
    fi::EncodedHistoryCache cfa_cache;
    auto cfa_first = cached_prepare(
        frontend, cfa_cache,
        product_input({product_message(ninfer::ChatRole::User, "replay"),
                       product_message(ninfer::ChatRole::Assistant, "ans")},
                      cfa));
    auto cfa_second = cached_prepare(
        frontend, cfa_cache,
        product_input({product_message(ninfer::ChatRole::User, "replay"),
                       product_message(ninfer::ChatRole::Assistant, "ans")},
                      cfa));
    failures += check(cfa_first.observation.inserted && !cfa_second.observation.cache_hit,
                      "CFA replay inserted but did not conservatively miss on its own checkpoint");
    failures += expect_match_cold(
        frontend, cfa_second,
        product_input({product_message(ninfer::ChatRole::User, "replay"),
                       product_message(ninfer::ChatRole::Assistant, "ans")},
                      cfa),
        false, "CFA replay returned non-cold ids");
    failures += check(
        FrontendFactory::inspect(cfa_second.prompt).identity.rewrite_checkpoint &&
            FrontendFactory::inspect(cfa_second.prompt).identity.rewrite_checkpoint->kind ==
                ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay,
        "CFA replay checkpoint is not ResponseReplay");
    (void)cfa_first;

    ninfer::PromptOptions turn_closure;
    turn_closure.preserve_thinking = false;
    auto tc = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "x")}, turn_closure));
    auto tc2 = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "x")}, turn_closure));
    failures += expect_match_cold(
        frontend, tc2,
        product_input({product_message(ninfer::ChatRole::User, "x")}, turn_closure), true,
        "E-H8 TurnClosure");
    failures += check(
        FrontendFactory::inspect(tc2.prompt).identity.rewrite_checkpoint &&
            FrontendFactory::inspect(tc2.prompt).identity.rewrite_checkpoint->kind ==
                ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
        "E-H8 kind is not TurnClosure from this render");

    auto rr = cached_prepare(
        frontend, cache, product_input({product_message(ninfer::ChatRole::User, "rr")}, preserve));
    auto rr2 = cached_prepare(
        frontend, cache, product_input({product_message(ninfer::ChatRole::User, "rr")}, preserve));
    failures += expect_match_cold(
        frontend, rr2,
        product_input({product_message(ninfer::ChatRole::User, "rr")}, preserve), true,
        "E-H9 ResponseReplay thinking");
    const std::string rr_committed =
        thinking_toggle_template()
            .render({chat_message(ninfer::ChatRole::User, "rr")},
                    [&] {
                        fi::ChatRenderOptions options;
                        options.add_generation_prompt = false;
                        return options;
                    }())
            .text;
    failures += check(
        FrontendFactory::inspect(rr2.prompt).identity.rewrite_checkpoint &&
            FrontendFactory::inspect(rr2.prompt).identity.rewrite_checkpoint->kind ==
                ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
            FrontendFactory::inspect(rr2.prompt).identity.rewrite_checkpoint->frontier ==
                fixture_tokenizer().encode(rr_committed).size(),
        "E-H9 ResponseReplay frontier is not the committed token count");
    (void)rr;
    (void)tc;

    auto nt = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "nt")}, think_off));
    auto nt2 = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "nt")}, think_off));
    failures += expect_match_cold(
        frontend, nt2,
        product_input({product_message(ninfer::ChatRole::User, "nt")}, think_off), true,
        "E-H10 ResponseReplay non-thinking");
    (void)nt;

    fi::EncodedHistoryCache branch_cache;
    (void)cached_prepare(frontend, branch_cache,
                         product_input({product_message(ninfer::ChatRole::User, "root")}, preserve));
    auto branch_a = cached_prepare(
        frontend, branch_cache,
        product_input({product_message(ninfer::ChatRole::User, "root"),
                       product_message(ninfer::ChatRole::Assistant, "a"),
                       product_message(ninfer::ChatRole::User, "left")},
                      preserve));
    failures += expect_match_cold(
        frontend, branch_a,
        product_input({product_message(ninfer::ChatRole::User, "root"),
                       product_message(ninfer::ChatRole::Assistant, "a"),
                       product_message(ninfer::ChatRole::User, "left")},
                      preserve),
        true, "E-H11 long prefix of root+left");
    auto branch_b = cached_prepare(
        frontend, branch_cache,
        product_input({product_message(ninfer::ChatRole::User, "root"),
                       product_message(ninfer::ChatRole::Assistant, "a"),
                       product_message(ninfer::ChatRole::User, "right")},
                      preserve));
    failures += expect_match_cold(
        frontend, branch_b,
        product_input({product_message(ninfer::ChatRole::User, "root"),
                       product_message(ninfer::ChatRole::Assistant, "a"),
                       product_message(ninfer::ChatRole::User, "right")},
                      preserve),
        true, "E-H12 different branch still prefixed by root");
    failures += check(branch_b.observation.cache_hit && branch_b.observation.prefix_bytes > 0,
                      "E-H12 did not hit the shared root prefix");

    ninfer::PromptOptions strip;
    strip.preserve_thinking = false;
    strip.enable_thinking   = false;
    fi::EncodedHistoryCache strip_cache;
    (void)cached_prepare(
        frontend, strip_cache,
        product_input({product_message(ninfer::ChatRole::User, "q1"),
                       product_message(ninfer::ChatRole::Assistant,
                                       "<think>\nold thought\n</think>\n\nold answer")},
                      preserve));
    auto stripped = cached_prepare(
        frontend, strip_cache,
        product_input({product_message(ninfer::ChatRole::User, "q1"),
                       product_message(ninfer::ChatRole::Assistant,
                                       "<think>\nold thought\n</think>\n\nold answer"),
                       product_message(ninfer::ChatRole::User, "q2")},
                      strip));
    failures += expect_match_cold(
        frontend, stripped,
        product_input({product_message(ninfer::ChatRole::User, "q1"),
                       product_message(ninfer::ChatRole::Assistant,
                                       "<think>\nold thought\n</think>\n\nold answer"),
                       product_message(ninfer::ChatRole::User, "q2")},
                      strip),
        false, "E-M1 think-strip miss");

    fi::EncodedHistoryCache tool_cache;
    ninfer::PromptOptions with_tools = preserve;
    with_tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"f","parameters":{"type":"object"}}})");
    (void)cached_prepare(frontend, tool_cache,
                         product_input({product_message(ninfer::ChatRole::User, "hi")}, preserve));
    auto tools_changed = cached_prepare(
        frontend, tool_cache,
        product_input({product_message(ninfer::ChatRole::User, "hi")}, with_tools));
    failures += expect_match_cold(
        frontend, tools_changed,
        product_input({product_message(ninfer::ChatRole::User, "hi")}, with_tools), false,
        "E-M2 tool_jsons change");

    ninfer::ChatMessage lookup;
    lookup.role = ninfer::ChatRole::Assistant;
    lookup.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "", .media = {}});
    lookup.tool_calls.push_back(
        ninfer::ToolCall{.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    fi::EncodedHistoryCache group_cache;
    (void)cached_prepare(
        frontend, group_cache,
        product_input({product_message(ninfer::ChatRole::User, "weather?"), lookup,
                       product_message(ninfer::ChatRole::Tool, "sunny")},
                      preserve));
    auto two_tools = cached_prepare(
        frontend, group_cache,
        product_input({product_message(ninfer::ChatRole::User, "weather?"), lookup,
                       product_message(ninfer::ChatRole::Tool, "sunny"),
                       product_message(ninfer::ChatRole::Tool, "20C")},
                      preserve));
    failures += expect_match_cold(
        frontend, two_tools,
        product_input({product_message(ninfer::ChatRole::User, "weather?"), lookup,
                       product_message(ninfer::ChatRole::Tool, "sunny"),
                       product_message(ninfer::ChatRole::Tool, "20C")},
                      preserve),
        false, "E-M4 tool-group growth");

    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage image_message;
    image_message.role = ninfer::ChatRole::User;
    image_message.parts.push_back(std::move(image));
    const std::size_t before_media = cache.size();
    auto media = cached_prepare(frontend, cache, product_input({std::move(image_message)}));
    failures += check(!media.observation.cache_hit && cache.size() == before_media,
                      "E-M5 media used or inserted into the text cache");
    failures += check(FrontendFactory::inspect(media.prompt).has_media(),
                      "E-M5 media prepare lost vision items");

    const std::size_t before_tokens                     = cache.size();
    const fi::HostEncodeObservation before_tokens_obs   = fi::last_host_encode_observation;
    const auto sample =
        frontend.prepare(product_input({product_message(ninfer::ChatRole::User, "tok")}));
    (void)frontend.prepare_tokens(FrontendFactory::inspect(sample).token_ids, false);
    failures += check(cache.size() == before_tokens &&
                          same_observation(before_tokens_obs, fi::last_host_encode_observation),
                      "E-M6 prepare_tokens touched the cache");

    const fi::HostEncodeObservation before_frontend = fi::last_host_encode_observation;
    (void)frontend.prepare(product_input({product_message(ninfer::ChatRole::User, "cold path")}));
    failures += check(same_observation(before_frontend, fi::last_host_encode_observation),
                      "E-M7 Frontend::prepare mutated host-encode observation");

    fi::EncodedHistoryCache a;
    fi::EncodedHistoryCache b;
    (void)cached_prepare(frontend, a, product_input({product_message(ninfer::ChatRole::User, "iso")}));
    auto b_first = cached_prepare(
        frontend, b,
        product_input({product_message(ninfer::ChatRole::User, "iso"),
                       product_message(ninfer::ChatRole::Assistant, "r"),
                       product_message(ninfer::ChatRole::User, "next")},
                      preserve));
    failures += check(!b_first.observation.cache_hit, "E-M13 cache B hit cache A's history");

    ninfer::PromptOptions effort_on;
    effort_on.reasoning_effort   = ninfer::ReasoningEffort::XHigh;
    effort_on.preserve_thinking  = true;
    ninfer::PromptOptions effort_off_opt;
    effort_off_opt.enable_thinking   = false;
    effort_off_opt.preserve_thinking = true;
    fi::EncodedHistoryCache effort_cache;
    (void)cached_prepare(effort_frontend, effort_cache,
                         product_input({product_message(ninfer::ChatRole::User, "effort")},
                                       effort_on));
    auto effort_flip = cached_prepare(
        effort_frontend, effort_cache,
        product_input({product_message(ninfer::ChatRole::User, "effort")}, effort_off_opt));
    failures += expect_match_cold(
        effort_frontend, effort_flip,
        product_input({product_message(ninfer::ChatRole::User, "effort")}, effort_off_opt), false,
        "E-M14 ReasoningEffort enable_thinking flip");

    ninfer::PromptOptions low;
    low.reasoning_effort   = ninfer::ReasoningEffort::Low;
    low.preserve_thinking  = true;
    auto effort_level = cached_prepare(
        effort_frontend, effort_cache,
        product_input({product_message(ninfer::ChatRole::User, "effort")}, low));
    failures += expect_match_cold(
        effort_frontend, effort_level,
        product_input({product_message(ninfer::ChatRole::User, "effort")}, low), false,
        "E-M3 reasoning_effort change");
    return failures;
}

int test_concurrency_and_copy_out() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    ninfer::PromptOptions preserve;
    preserve.preserve_thinking = true;
    int failures               = 0;

    std::atomic<int> thread_failures{0};
    std::vector<std::thread> threads;
    fi::EncodedHistoryCache shared;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            const std::string tag = "hist-" + std::to_string(t);
            auto first            = cached_prepare(
                frontend, shared,
                product_input({product_message(ninfer::ChatRole::User, tag)}, preserve));
            auto second = cached_prepare(
                frontend, shared,
                product_input({product_message(ninfer::ChatRole::User, tag),
                               product_message(ninfer::ChatRole::Assistant, "ok"),
                               product_message(ninfer::ChatRole::User, "next")},
                              preserve));
            const auto cold = frontend.prepare(product_input(
                {product_message(ninfer::ChatRole::User, tag),
                 product_message(ninfer::ChatRole::Assistant, "ok"),
                 product_message(ninfer::ChatRole::User, "next")},
                preserve));
            if (first.observation.cache_hit || !second.observation.cache_hit ||
                FrontendFactory::inspect(second.prompt).token_ids !=
                    FrontendFactory::inspect(cold).token_ids) {
                thread_failures.fetch_add(1);
            }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    failures += check(thread_failures.load() == 0, "C1 distinct-history threads failed");

    fi::EncodedHistoryCache identical;
    threads.clear();
    thread_failures.store(0);
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            auto first = cached_prepare(
                frontend, identical,
                product_input({product_message(ninfer::ChatRole::User, "same")}, preserve));
            auto second = cached_prepare(
                frontend, identical,
                product_input({product_message(ninfer::ChatRole::User, "same"),
                               product_message(ninfer::ChatRole::Assistant, "ok"),
                               product_message(ninfer::ChatRole::User, "next")},
                              preserve));
            const auto cold = frontend.prepare(product_input(
                {product_message(ninfer::ChatRole::User, "same"),
                 product_message(ninfer::ChatRole::Assistant, "ok"),
                 product_message(ninfer::ChatRole::User, "next")},
                preserve));
            if (FrontendFactory::inspect(first.prompt).token_ids !=
                    FrontendFactory::inspect(frontend.prepare(product_input(
                        {product_message(ninfer::ChatRole::User, "same")}, preserve)))
                        .token_ids ||
                !second.observation.cache_hit ||
                FrontendFactory::inspect(second.prompt).token_ids !=
                    FrontendFactory::inspect(cold).token_ids) {
                thread_failures.fetch_add(1);
            }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    failures += check(thread_failures.load() == 0, "C2 identical-history threads failed");
    failures += check(identical.size() <= fi::kHostEncodeCacheEntries, "C2 exceeded cache cap");

    fi::EncodedHistoryCache evict;
    const auto planted = cached_prepare(
        frontend, evict,
        product_input({product_message(ninfer::ChatRole::User, "keep")}, preserve));
    const fi::RenderedChat planted_full = thinking_toggle_template().render(
        {chat_message(ninfer::ChatRole::User, "keep"), chat_message(ninfer::ChatRole::Assistant, "a"),
         chat_message(ninfer::ChatRole::User, "next")},
        [&] {
            fi::ChatRenderOptions options;
            options.preserve_thinking = true;
            return options;
        }());
    auto copied = evict.copy_longest_prefix(
        planted_full.text, fixture_tokenizer(),
        planted_full.rewrite_checkpoint
            ? std::optional<std::size_t>{planted_full.rewrite_checkpoint->offset}
            : std::nullopt);
    failures += check(copied.has_value(), "C3 did not copy the planted prefix");
    for (int i = 0; i < 16; ++i) {
        const std::string text = "evict-" + std::to_string(i) + std::string(32, 'x');
        evict.insert_committed(text, fixture_tokenizer().encode(text), {});
    }
    if (copied) {
        const std::vector<fi::CommittedBoundary> no_boundaries{};
        const auto spliced =
            fi::try_splice_encoded_chat(fixture_tokenizer(), copied->ids, planted_full,
                                        copied->bytes.size(), no_boundaries, kNoTokenLimit);
        const fi::EncodedChat cold = fi::encode_rendered_chat(fixture_tokenizer(), planted_full);
        failures += check(spliced && spliced->input_ids == cold.input_ids,
                          "C3 copy-out splice disagreed with cold after eviction");
    }
    (void)planted;
    return failures;
}

int test_verify_poison() {
    const Frontend frontend = FrontendFactory::create_component(resources());
    ninfer::PromptOptions preserve;
    preserve.preserve_thinking = true;
    fi::EncodedHistoryCache cache;
    int failures = 0;

    set_test_env("NINFER_VERIFY_HOST_ENCODE", "1");
    (void)cached_prepare(frontend, cache,
                         product_input({product_message(ninfer::ChatRole::User, "poison")},
                                       preserve));
    fi::ChatRenderOptions committed_opts;
    committed_opts.add_generation_prompt = false;
    committed_opts.preserve_thinking     = true;
    const std::string committed =
        thinking_toggle_template()
            .render({chat_message(ninfer::ChatRole::User, "poison")}, committed_opts)
            .text;
    cache.poison_committed_ids(committed);
    auto poisoned = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "poison"),
                       product_message(ninfer::ChatRole::Assistant, "ans"),
                       product_message(ninfer::ChatRole::User, "next")},
                      preserve));
    failures += check(poisoned.observation.verified_mismatch,
                      "V3 poison did not trip verify mismatch");
    failures += check(!poisoned.observation.cache_hit, "V3 poison still reported a cache hit");
    failures += expect_match_cold(
        frontend, poisoned,
        product_input({product_message(ninfer::ChatRole::User, "poison"),
                       product_message(ninfer::ChatRole::Assistant, "ans"),
                       product_message(ninfer::ChatRole::User, "next")},
                      preserve),
        false, "V3 poison returned non-cold ids");

    auto after_drop = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "poison"),
                       product_message(ninfer::ChatRole::Assistant, "ans"),
                       product_message(ninfer::ChatRole::User, "next")},
                      preserve));
    failures += check(!after_drop.observation.verified_mismatch,
                      "V3 second call still saw a poisoned entry");

    auto legal = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "verify-ok")}, preserve));
    auto legal2 = cached_prepare(
        frontend, cache,
        product_input({product_message(ninfer::ChatRole::User, "verify-ok"),
                       product_message(ninfer::ChatRole::Assistant, "ans"),
                       product_message(ninfer::ChatRole::User, "next")},
                      preserve));
    failures += check(legal2.observation.cache_hit && !legal2.observation.verified_mismatch,
                      "V1 legal hit failed under verify");
    set_test_env("NINFER_VERIFY_HOST_ENCODE", "0");
    (void)legal;
    return failures;
}

int test_non_loop_pos_insert_refused() {
    const fi::Tokenizer& tokenizer = fixture_tokenizer();
    fi::EncodedHistoryCache cache;
    const std::string full = thinking_toggle_template()
                                 .render({chat_message(ninfer::ChatRole::User, "abcdef")}, {})
                                 .text;
    const std::size_t mid = full.find("bcd");
    int failures = check(mid != std::string::npos && !tokenizer.is_encode_loop_pos(full, mid + 1),
                         "mid-user cut was a loop-pos");
    cache.insert_committed(full.substr(0, mid + 1), tokenizer.encode(full.substr(0, mid + 1)), {});
    auto copied = cache.copy_longest_prefix(full, tokenizer, std::nullopt);
    failures += check(!copied, "non-loop-pos insert was used as a hit");
    return failures;
}

// Sweep every byte offset of the rendered templates: legal cuts splice exactly like a cold
// encode, illegal cuts are refused by both the splice and the cache, and one-byte scrambles
// never hit.
int byte_boundary_cases(const fi::Tokenizer& tokenizer, const char* label, bool official) {
    int failures                   = 0;
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking = true;
    fi::ChatRenderOptions no_think;
    no_think.enable_thinking = false;
    fi::ChatRenderOptions no_think_committed = no_think;
    no_think_committed.add_generation_prompt = false;

    fi::ChatMessage lookup = chat_message(ninfer::ChatRole::Assistant, "");
    lookup.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});

    const std::vector<fi::ChatMessage> user_hello{chat_message(ninfer::ChatRole::User, "hello")};
    const std::vector<fi::ChatMessage> user_cjk{
        chat_message(ninfer::ChatRole::User, "缓存复用 café and NFC 0123456789")};
    const std::vector<fi::ChatMessage> multi{
        chat_message(ninfer::ChatRole::User, "q1"),
        chat_message(ninfer::ChatRole::Assistant, "<think>\nold thought\n</think>\n\nold answer"),
        chat_message(ninfer::ChatRole::User, "q2 punctuation, digits, and CJK: 边界.")};
    const std::vector<fi::ChatMessage> tools_one{
        chat_message(ninfer::ChatRole::User, "weather?"), lookup,
        chat_message(ninfer::ChatRole::Tool, "sunny")};
    const std::vector<fi::ChatMessage> tools_two{
        chat_message(ninfer::ChatRole::User, "weather?"), lookup,
        chat_message(ninfer::ChatRole::Tool, "sunny"),
        chat_message(ninfer::ChatRole::Tool, "20C")};
    const std::vector<fi::ChatMessage> late{
        chat_message(ninfer::ChatRole::System, "stable policy"),
        chat_message(ninfer::ChatRole::User, "hi"),
        chat_message(ninfer::ChatRole::System, "current diagnostics")};
    const std::vector<fi::ChatMessage> effort_user{chat_message(ninfer::ChatRole::User, "hello")};

    struct Case {
        fi::RenderedChat rendered;
        const char* name;
    };
    std::vector<Case> cases;
    auto add = [&](fi::RenderedChat rendered, const char* name) {
        cases.push_back(Case{std::move(rendered), name});
    };
    add(thinking_toggle_template().render(user_hello, {}), "thinking-toggle hello full");
    add(thinking_toggle_template().render(user_hello, no_generation), "thinking-toggle hello committed");
    add(thinking_toggle_template().render(user_hello, no_think), "thinking-toggle hello no-think");
    add(thinking_toggle_template().render(user_cjk, {}), "thinking-toggle CJK/NFC full");
    add(thinking_toggle_template().render(multi, preserve), "thinking-toggle multi-turn think");
    add(thinking_toggle_template().render(multi, no_think), "thinking-toggle multi-turn strip");
    add(thinking_toggle_template().render(tools_one, no_generation), "thinking-toggle one tool");
    add(thinking_toggle_template().render(tools_two, no_generation), "thinking-toggle two tools");
    add(thinking_toggle_template().render(late, {}), "thinking-toggle late system");
    fi::ChatRenderOptions with_tools = preserve;
    with_tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"f","parameters":{"type":"object"}}})");
    add(thinking_toggle_template().render(user_hello, with_tools), "thinking-toggle tool_jsons");

    fi::ChatRenderOptions effort_medium;
    effort_medium.reasoning_effort = ninfer::ReasoningEffort::Medium;
    fi::ChatRenderOptions effort_medium_c = effort_medium;
    effort_medium_c.add_generation_prompt = false;
    fi::ChatRenderOptions effort_xhigh;
    effort_xhigh.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    add(reasoning_effort_template().render(effort_user, effort_medium), "effort medium full");
    add(reasoning_effort_template().render(effort_user, effort_medium_c), "effort medium committed");
    add(reasoning_effort_template().render(effort_user, effort_xhigh), "effort xhigh full");
    add(reasoning_effort_template().render(effort_user, no_think), "effort enable_thinking off");

    if (official) {
        // BPE merges make a mid-word split re-segment differently: the concat must disagree
        // with the cold encode where the toy byte tokenizer cannot express that.
        const std::vector<int> he    = tokenizer.encode("he");
        const std::vector<int> llo   = tokenizer.encode("llo");
        std::vector<int> concat      = he;
        concat.insert(concat.end(), llo.begin(), llo.end());
        failures += check(concat != tokenizer.encode("hello"),
                          "P1-R1 mid-word concat accidentally equals cold encode");
    }

    for (const Case& test_case : cases) {
        failures += check_every_byte_boundary(
            tokenizer, test_case.rendered.text, test_case.rendered.rewrite_checkpoint,
            (std::string(label) + " " + test_case.name).c_str());
        if (failures != 0) { return failures; }
    }

    failures += check_special_token_interiors(tokenizer, "<|im_start|>");
    failures += check_special_token_interiors(tokenizer, "<|im_end|>");
    failures += check_special_token_interiors(tokenizer, "<think>");
    failures += check_special_token_interiors(tokenizer, "</think>");
    failures += check_special_token_interiors(tokenizer, "<|vision_start|>");
    failures += check_special_token_interiors(tokenizer, "<|image_pad|>");
    if (failures != 0) { return failures; }

    const fi::RenderedChat alpha =
        thinking_toggle_template().render({chat_message(ninfer::ChatRole::User, "alpha")}, {});
    const fi::RenderedChat alphabet =
        thinking_toggle_template().render({chat_message(ninfer::ChatRole::User, "alphabet")}, {});
    std::size_t shared = 0;
    while (shared < alpha.text.size() && shared < alphabet.text.size() &&
           alpha.text[shared] == alphabet.text[shared]) {
        ++shared;
    }
    failures += check(shared > 0 && shared < alphabet.text.size(),
                      "alpha/alphabet rendered chats do not share a proper prefix");
    failures += check(!tokenizer.is_encode_loop_pos(alphabet.text, shared),
                      "shared alpha/alphabet cut is a tokenizer boundary of alphabet");
    if (shared > 0) {
        fi::EncodedHistoryCache shared_cache;
        shared_cache.insert_committed(std::string(alphabet.text.substr(0, shared)), {1}, {});
        failures += check(
            !shared_cache.copy_longest_prefix(alphabet.text, tokenizer, std::nullopt),
            "shared non-boundary bytes were used as a cache hit on alphabet");
        fi::EncodedHistoryCache alpha_cache;
        const std::vector<int> alpha_ids = tokenizer.encode(alpha.text);
        alpha_cache.insert_committed(alpha.text, alpha_ids, {});
        failures += check(!alpha_cache.copy_longest_prefix(alphabet.text, tokenizer, std::nullopt),
                          "alpha committed bytes were used as a hit on alphabet");
    }

    const fi::RenderedChat word =
        thinking_toggle_template().render({chat_message(ninfer::ChatRole::User, "boundaryword")}, {});
    std::size_t legal_short  = 0;
    std::size_t illegal_long = 0;
    for (std::size_t n = 1; n < word.text.size(); ++n) {
        if (tokenizer.is_encode_loop_pos(word.text, n)) {
            if (legal_short == 0) { legal_short = n; }
        } else if (legal_short != 0) {
            illegal_long = n;
            break;
        }
    }
    failures += check(legal_short != 0 && illegal_long > legal_short,
                      "could not find a legal cut followed by an illegal longer prefix");
    if (legal_short != 0 && illegal_long > legal_short) {
        const std::vector<int> short_ids = tokenizer.encode(word.text.substr(0, legal_short));
        fi::EncodedHistoryCache mixed;
        mixed.insert_committed(word.text.substr(0, illegal_long), {1, 2, 3}, {});
        mixed.insert_committed(word.text.substr(0, legal_short), short_ids, {});
        const auto hit = mixed.copy_longest_prefix(word.text, tokenizer, std::nullopt);
        failures += check(hit && hit->bytes.size() == legal_short && hit->ids == short_ids,
                          "illegal longer byte prefix beat the shorter legal tokenizer cut");
    }

    fi::EncodedHistoryCache longer_than_full;
    longer_than_full.insert_committed(word.text + "x", tokenizer.encode(word.text + "x"), {});
    failures += check(!longer_than_full.copy_longest_prefix(word.text, tokenizer, std::nullopt),
                      "stored bytes longer than full were used as a hit");

    failures += check_every_byte_boundary(tokenizer, "AB", std::nullopt,
                                          (std::string(label) + " AB").c_str());
    failures += check_every_byte_boundary(tokenizer, "<|im_start|>", std::nullopt,
                                          (std::string(label) + " im_start").c_str());
    failures += check_every_byte_boundary(tokenizer, "<|im_end|>", std::nullopt,
                                          (std::string(label) + " im_end").c_str());
    return failures;
}

int test_byte_match_and_boundaries(bool official) {
    int failures = byte_boundary_cases(fixture_tokenizer(), "toy", false);
    if (failures != 0) { return failures; }
    if (official) {
        failures += byte_boundary_cases(official_tokenizer(), "official", true);
    }
    return failures;
}

int test_coverage_gaps() {
    try {
        const Frontend frontend = FrontendFactory::create_component(resources());
        ninfer::PromptOptions preserve;
        preserve.preserve_thinking = true;
        int failures               = 0;

        ninfer::ChatMessage lookup;
        lookup.role = ninfer::ChatRole::Assistant;
        lookup.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "", .media = {}});
        lookup.tool_calls.push_back(
            ninfer::ToolCall{.id = "", .name = "lookup",
                             .arguments_json = R"({"city":"Paris"})"});
        fi::EncodedHistoryCache tool_hit_cache;
        (void)cached_prepare(
            frontend, tool_hit_cache,
            product_input({product_message(ninfer::ChatRole::User, "weather?"), lookup,
                           product_message(ninfer::ChatRole::Tool, "sunny")},
                          preserve));
        auto tool_closed = cached_prepare(
            frontend, tool_hit_cache,
            product_input({product_message(ninfer::ChatRole::User, "weather?"), lookup,
                           product_message(ninfer::ChatRole::Tool, "sunny"),
                           product_message(ninfer::ChatRole::User, "thanks")},
                          preserve));
        failures += expect_match_cold(
            frontend, tool_closed,
            product_input({product_message(ninfer::ChatRole::User, "weather?"), lookup,
                           product_message(ninfer::ChatRole::Tool, "sunny"),
                           product_message(ninfer::ChatRole::User, "thanks")},
                          preserve),
            true, "closed tool history append should hit");

        fi::EncodedHistoryCache turn3;
        (void)cached_prepare(
            frontend, turn3,
            product_input({product_message(ninfer::ChatRole::User, "t3")}, preserve));
        auto turn3_second = cached_prepare(
            frontend, turn3,
            product_input({product_message(ninfer::ChatRole::User, "t3"),
                           product_message(ninfer::ChatRole::Assistant, "a1"),
                           product_message(ninfer::ChatRole::User, "u2")},
                          preserve));
        failures += expect_match_cold(
            frontend, turn3_second,
            product_input({product_message(ninfer::ChatRole::User, "t3"),
                           product_message(ninfer::ChatRole::Assistant, "a1"),
                           product_message(ninfer::ChatRole::User, "u2")},
                          preserve),
            true, "turn-2 append should hit");
        auto turn3_third = cached_prepare(
            frontend, turn3,
            product_input({product_message(ninfer::ChatRole::User, "t3"),
                           product_message(ninfer::ChatRole::Assistant, "a1"),
                           product_message(ninfer::ChatRole::User, "u2"),
                           product_message(ninfer::ChatRole::Assistant, "a2"),
                           product_message(ninfer::ChatRole::User, "u3")},
                          preserve));
        failures += expect_match_cold(
            frontend, turn3_third,
            product_input({product_message(ninfer::ChatRole::User, "t3"),
                           product_message(ninfer::ChatRole::Assistant, "a1"),
                           product_message(ninfer::ChatRole::User, "u2"),
                           product_message(ninfer::ChatRole::Assistant, "a2"),
                           product_message(ninfer::ChatRole::User, "u3")},
                          preserve),
            true, "turn-3 append should hit the longer committed");
        failures += check(
            turn3_third.observation.prefix_bytes > turn3_second.observation.prefix_bytes,
            "turn-3 spliced the first-turn prefix instead of the inserted longer committed");

        ninfer::MessagePart video;
        video.kind              = ninfer::MessagePartKind::Media;
        video.media.kind        = ninfer::MediaKind::Video;
        video.media.bytes       = gradient_ppm();
        video.media.media_type  = "image/x-portable-pixmap";
        video.media.source_name = "single-frame.ppm";
        ninfer::ChatMessage video_message;
        video_message.role = ninfer::ChatRole::User;
        video_message.parts.push_back(std::move(video));
        fi::EncodedHistoryCache media_cache;
        (void)cached_prepare(
            frontend, media_cache,
            product_input({product_message(ninfer::ChatRole::User, "AB")}));
        const std::size_t after_text = media_cache.size();
        auto video_call = cached_prepare(frontend, media_cache,
                                         product_input({video_message}));
        failures += check(!video_call.observation.cache_hit && media_cache.size() == after_text &&
                              FrontendFactory::inspect(video_call.prompt).has_media(),
                          "video prepare used or inserted into the text cache");
        const std::size_t before_video_count = media_cache.size();
        const std::uint32_t video_count      = EncodedHistoryPrepare::count_tokens(
            frontend, product_input({std::move(video_message)}), media_cache);
        failures += check(video_count > 0 && media_cache.size() == before_video_count &&
                              !fi::last_host_encode_observation.cache_hit &&
                              !fi::last_host_encode_observation.inserted,
                          "video count_tokens used or inserted into the text cache");

        fi::EncodedHistoryCache cap_cache;
        cap_cache.insert_committed(std::string(fi::kHostEncodeCacheMaxBytes + 1, 'a'), {1, 2, 3},
                                   {});
        failures += check(cap_cache.size() == 0, "E-M12 oversized UTF-8 was inserted");
        cap_cache.insert_committed("ok", std::vector<int>(fi::kHostEncodeCacheMaxIds + 1, 1), {});
        failures += check(cap_cache.size() == 0, "E-M12 oversized id vector was inserted");
        cap_cache.insert_committed("", {1}, {});
        failures += check(cap_cache.size() == 0, "empty committed was inserted");

        fi::EncodedHistoryCache lru;
        for (int i = 0; i < 16; ++i) {
            (void)cached_prepare(
                frontend, lru,
                product_input({product_message(ninfer::ChatRole::User,
                                               "lru-" + std::to_string(i))},
                              preserve));
        }
        failures += check(lru.size() == 16, "C6 did not fill 16 entries");
        (void)cached_prepare(
            frontend, lru,
            product_input({product_message(ninfer::ChatRole::User, "lru-16")}, preserve));
        auto evicted = cached_prepare(
            frontend, lru,
            product_input({product_message(ninfer::ChatRole::User, "lru-0"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, evicted,
            product_input({product_message(ninfer::ChatRole::User, "lru-0"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            false, "C6 LRU victim still hit after eviction");
        auto recent = cached_prepare(
            frontend, lru,
            product_input({product_message(ninfer::ChatRole::User, "lru-16"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, recent,
            product_input({product_message(ninfer::ChatRole::User, "lru-16"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            true, "C6 most recent insert missed");

        fi::EncodedHistoryCache lru_hit;
        for (int i = 0; i < 16; ++i) {
            (void)cached_prepare(
                frontend, lru_hit,
                product_input({product_message(ninfer::ChatRole::User,
                                               "touch-" + std::to_string(i))},
                              preserve));
        }
        auto touched = cached_prepare(
            frontend, lru_hit,
            product_input({product_message(ninfer::ChatRole::User, "touch-0")}, preserve));
        failures += expect_match_cold(
            frontend, touched,
            product_input({product_message(ninfer::ChatRole::User, "touch-0")}, preserve), true,
            "C6 replay of oldest entry missed");
        (void)cached_prepare(
            frontend, lru_hit,
            product_input({product_message(ninfer::ChatRole::User, "touch-16")}, preserve));
        auto refreshed = cached_prepare(
            frontend, lru_hit,
            product_input({product_message(ninfer::ChatRole::User, "touch-0"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, refreshed,
            product_input({product_message(ninfer::ChatRole::User, "touch-0"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            true, "C6 LRU hit did not refresh the victim");
        auto stale = cached_prepare(
            frontend, lru_hit,
            product_input({product_message(ninfer::ChatRole::User, "touch-1"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, stale,
            product_input({product_message(ninfer::ChatRole::User, "touch-1"),
                           product_message(ninfer::ChatRole::Assistant, "ok"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            false, "C6 untouched neighbor survived after a refreshed hit");

        fi::EncodedHistoryCache mixed;
        (void)cached_prepare(
            frontend, mixed,
            product_input({product_message(ninfer::ChatRole::User, "mix")}, preserve));
        std::atomic<int> mix_failures{0};
        std::vector<std::thread> mix_threads;
        for (int t = 0; t < 8; ++t) {
            mix_threads.emplace_back([&, t] {
                if ((t % 2) == 0) {
                    const std::uint32_t count = EncodedHistoryPrepare::count_tokens(
                        frontend,
                        product_input({product_message(ninfer::ChatRole::User, "mix"),
                                       product_message(ninfer::ChatRole::Assistant, "ok"),
                                       product_message(ninfer::ChatRole::User, "next")},
                                      preserve),
                        mixed);
                    const std::uint32_t cold = frontend.count_tokens(product_input(
                        {product_message(ninfer::ChatRole::User, "mix"),
                         product_message(ninfer::ChatRole::Assistant, "ok"),
                         product_message(ninfer::ChatRole::User, "next")},
                        preserve));
                    if (!fi::last_host_encode_observation.cache_hit || count != cold) {
                        mix_failures.fetch_add(1);
                    }
                } else {
                    auto call = cached_prepare(
                        frontend, mixed,
                        product_input({product_message(ninfer::ChatRole::User, "mix"),
                                       product_message(ninfer::ChatRole::Assistant, "ok"),
                                       product_message(ninfer::ChatRole::User, "next")},
                                      preserve));
                    const auto cold = frontend.prepare(product_input(
                        {product_message(ninfer::ChatRole::User, "mix"),
                         product_message(ninfer::ChatRole::Assistant, "ok"),
                         product_message(ninfer::ChatRole::User, "next")},
                        preserve));
                    if (!call.observation.cache_hit ||
                        FrontendFactory::inspect(call.prompt).token_ids !=
                            FrontendFactory::inspect(cold).token_ids) {
                        mix_failures.fetch_add(1);
                    }
                }
            });
        }
        for (std::thread& thread : mix_threads) { thread.join(); }
        failures += check(mix_failures.load() == 0, "C8 concurrent count_tokens/prepare failed");

        fi::EncodedHistoryCache byte_poison;
        (void)cached_prepare(
            frontend, byte_poison,
            product_input({product_message(ninfer::ChatRole::User, "keep-bytes")}, preserve));
        (void)cached_prepare(
            frontend, byte_poison,
            product_input({product_message(ninfer::ChatRole::User, "scramble-me")}, preserve));
        fi::ChatRenderOptions committed_opts;
        committed_opts.add_generation_prompt = false;
        committed_opts.preserve_thinking     = true;
        const std::string scrambled =
            thinking_toggle_template()
                .render({chat_message(ninfer::ChatRole::User, "scramble-me")}, committed_opts)
                .text;
        byte_poison.scramble_committed_bytes(scrambled);
        auto scrambled_call = cached_prepare(
            frontend, byte_poison,
            product_input({product_message(ninfer::ChatRole::User, "scramble-me"),
                           product_message(ninfer::ChatRole::Assistant, "ans"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, scrambled_call,
            product_input({product_message(ninfer::ChatRole::User, "scramble-me"),
                           product_message(ninfer::ChatRole::Assistant, "ans"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            false, "V4 scrambled bytes still hit");
        failures += check(!scrambled_call.observation.verified_mismatch,
                          "V4 scrambled bytes went through verify instead of memcmp miss");
        auto other_still = cached_prepare(
            frontend, byte_poison,
            product_input({product_message(ninfer::ChatRole::User, "keep-bytes"),
                           product_message(ninfer::ChatRole::Assistant, "ans"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, other_still,
            product_input({product_message(ninfer::ChatRole::User, "keep-bytes"),
                           product_message(ninfer::ChatRole::Assistant, "ans"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            true, "V4 scramble dropped an unrelated entry");

        set_test_env("NINFER_VERIFY_HOST_ENCODE", "1");
        fi::EncodedHistoryCache verify_rr;
        (void)cached_prepare(frontend, verify_rr,
                             product_input({product_message(ninfer::ChatRole::User, "v-rr")},
                                           preserve));
        auto rr = cached_prepare(
            frontend, verify_rr,
            product_input({product_message(ninfer::ChatRole::User, "v-rr"),
                           product_message(ninfer::ChatRole::Assistant, "ans"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve));
        failures += expect_match_cold(
            frontend, rr,
            product_input({product_message(ninfer::ChatRole::User, "v-rr"),
                           product_message(ninfer::ChatRole::Assistant, "ans"),
                           product_message(ninfer::ChatRole::User, "next")},
                          preserve),
            true, "V7 verify ResponseReplay append");
        failures += check(
            !rr.observation.verified_mismatch &&
                FrontendFactory::inspect(rr.prompt).identity.rewrite_checkpoint &&
                FrontendFactory::inspect(rr.prompt).identity.rewrite_checkpoint->kind ==
                    ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay,
            "V7 verify frontier/kind disagreed");
        set_test_env("NINFER_VERIFY_HOST_ENCODE", "0");
        return failures;
    } catch (const std::exception& error) {
        std::cerr << "test_coverage_gaps exception: " << error.what() << '\n';
        return 1;
    }
}

int run_prepare_bench() {
    if (!official_tokenizer_available()) {
        std::cerr << "run_prepare_bench: NINFER_OFFICIAL_TOKENIZER_DIR not set; skipping\n";
        return 0;
    }
    set_test_env("NINFER_VERIFY_HOST_ENCODE", "0");
    const Frontend frontend =
        FrontendFactory::create_component(
            official_frontend_resources(thinking_toggle_template_source()));
    fi::EncodedHistoryCache cache;
    const std::string paragraph =
        "Write a concise systems explanation of paged KV cache reuse, speculative decoding, "
        "and why host-side tokenization can hide under GPU prefill at 32k tokens. Include "
        "ASCII punctuation, numbers 0123456789, and a few Chinese characters: 缓存复用。\n";
    const std::vector<int> unit_ids = official_tokenizer().encode(paragraph);
    const int copies = (150000 + static_cast<int>(unit_ids.size()) - 1) /
                       static_cast<int>(unit_ids.size());
    std::string body;
    body.reserve(paragraph.size() * static_cast<std::size_t>(copies));
    for (int i = 0; i < copies; ++i) { body += paragraph; }

    ninfer::PromptOptions preserve;
    preserve.preserve_thinking = true;
    const auto t0              = std::chrono::steady_clock::now();
    const auto first_prompt    = EncodedHistoryPrepare::prepare(
        frontend, product_input({product_message(ninfer::ChatRole::User, body)}, preserve), cache);
    const double first_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const auto t1 = std::chrono::steady_clock::now();
    const auto second_prompt = EncodedHistoryPrepare::prepare(
        frontend,
        product_input({product_message(ninfer::ChatRole::User, body),
                       product_message(ninfer::ChatRole::Assistant, "ok"),
                       product_message(ninfer::ChatRole::User, "short follow-up")},
                      preserve),
        cache);
    const double second_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    const fi::HostEncodeObservation second_obs = fi::last_host_encode_observation;
    const auto cold = frontend.prepare(product_input(
        {product_message(ninfer::ChatRole::User, body),
         product_message(ninfer::ChatRole::Assistant, "ok"),
         product_message(ninfer::ChatRole::User, "short follow-up")},
        preserve));
    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "prepare_bench first_ms=" << first_ms << " second_ms=" << second_ms
              << " hit=" << second_obs.cache_hit
              << " tokens=" << FrontendFactory::inspect(second_prompt).token_ids.size() << '\n';
    if (!second_obs.cache_hit) {
        std::cerr << "second prepare was not a cache hit\n";
        return 1;
    }
    if (FrontendFactory::inspect(second_prompt).token_ids !=
        FrontendFactory::inspect(cold).token_ids) {
        std::cerr << "second prepare ids differ from cold\n";
        return 1;
    }
    (void)first_prompt;
    if (second_ms > first_ms * 0.5 && second_ms > 8.0) {
        std::cerr << "second prepare was not a suffix encode\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    set_test_env("NINFER_VERIFY_HOST_ENCODE", nullptr);
    if (std::getenv("NINFER_BENCH_PREPARE") != nullptr) { return run_prepare_bench(); }
    const bool official = official_tokenizer_available();
    if (!official) {
        std::cerr << "note: NINFER_OFFICIAL_TOKENIZER_DIR not set; the official-tokenizer "
                     "sections (BPE concat inequality, official per-byte sweep) are skipped\n";
    }
    int failures = 0;
    failures += test_loop_pos_and_splice();
    failures += test_committed_boundary_reuse();
    failures += test_engine_shaped_cache();
    failures += test_concurrency_and_copy_out();
    failures += test_verify_poison();
    failures += test_non_loop_pos_insert_refused();
    failures += test_coverage_gaps();
    failures += test_byte_match_and_boundaries(official);
    return failures == 0 ? 0 : 1;
}
