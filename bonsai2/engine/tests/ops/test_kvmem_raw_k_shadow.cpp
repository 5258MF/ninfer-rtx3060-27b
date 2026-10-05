// Op test for the KVMem raw-K shadow (src/ops/kvmem/raw_k_shadow.{h,cpp}, launch in
// src/ops/kvmem/raw_k_shadow_launch.cu).
//
// WHAT IS UNDER TEST
//   ops::raw_k_shadow_copy() is a pure BF16 device->device bit copy of the pre-RoPE key `kn`
//   (the window between text_context_impl.h:844 and :850). Because nothing is transformed, the
//   verdict is BIT EXACTNESS, not a tolerance: any difference at all is a bug.
//
// HOW THE TEST IS BUILT (and why it is built this way)
//   1. Host-only first. The negative controls and the switch semantics need no device, so they run
//      BEFORE the CUDA-availability skip. A GPU-less run still exercises the checker itself.
//   2. A NEGATIVE CONTROL IS MANDATORY. "The op matched" is worth nothing until a WRONG buffer is
//      shown to be rejected by the same code path. Four mutants are generated (transposed layout,
//      dropped last token row, one-slot shift, a single flipped mantissa bit) and the test FAILS if
//      any of them is accepted.
//   3. tokens > 1 is load-bearing. At tokens == 1 the declared [kv_size, tokens] placement and the
//      transposed [tokens, kv_size] placement are the SAME memory order, so a tokens==1 suite
//      cannot see a layout error at all. This file asserts that blind spot explicitly (so nobody
//      "simplifies" the case list down to tokens=1) and covers tokens = 1, 7, 64, 2048.
//   4. The FP64 oracle is the source lifted to double. That is what a copy's oracle IS, and it is
//      written by iterating the geometry so the index expression appears in the oracle as well.
//      It proves "no numeric transformation happened" (any scaling/rounding shows up as rel_l2 > 0)
//      but NOT the layout (an identity oracle is layout-blind) and NOT the sign of zero (+0 and -0
//      both lift to 0.0). Those two gaps are why the bit-exact compare is the primary verdict and
//      why the layout claims are carried by the mutants and the declared-shape checks.
//   5. Capture. The op is meant to run inside a captured CUDA graph, so one case captures it, then
//      replays the graph three times with DIFFERENT staging content and checks the shadow follows
//      the new content each time. The same capture also calls the host drain: it must refuse
//      (device->host copies and stream synchronises are illegal while capturing) and the capture
//      must still complete and instantiate -- a naive drain would break the capture here.
//
// THE SWITCH
//   NINFER_TERNARY_KVMEM is read once per process (kvmem_shadow.h), so this test cannot flip it.
//   It asserts the documented polarity and then asserts the behaviour of WHICHEVER arm it is
//   running in: ON -> the drain completes and delivers the staging bytes; OFF (the default) -> the
//   drain delivers nothing and the pinned buffer is left untouched (the OFF arm's "zero host
//   writes" claim). Run it both ways to cover both.

#include "ops/kvmem/kvmem_shadow.h"
#include "ops/kvmem/raw_k_shadow.h"

#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// --- geometry of the real call site (targets/qwen3_6_27b/impl/config.h:31-33) ------------------
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kKVSize  = kKVHeads * kHeadDim; // TextConfig::kv_size == 1024

// A BF16 pattern no test input produces, used both as a staging-buffer pre-fill (so "was this byte
// written at all" is observable) and as the body of the "dropped row" mutant.
constexpr std::uint16_t kSentinelBits = 0x5a5au;
constexpr std::uint8_t kSentinelByte  = 0x5au;

// The tightest criterion available: the op is a pure bit copy, so the error must be exactly zero.
constexpr ReductionCriterion kBitExact{0.0, 0.0, 0.0};

// The declared placement: element (dim, head, token) of raw K lands at
//     dim + head_dim * (head + kv_heads * token)
// i.e. one token's kv_size values are contiguous, and kv_size is the fastest axis. This expression
// is the whole layout contract of the op; the oracle, the mutants and the probes all use it.
[[nodiscard]] constexpr std::size_t shadow_index(int dim, int head, int token) {
    return static_cast<std::size_t>(dim) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

// The classic wrong placement, made explicit: what a tensor declared [tokens, kv_size] addresses,
// i.e. token as the fastest axis instead of the slowest. It is a permutation of the same bytes for
// every token count, and it is identical to the declared placement exactly when tokens == 1.
[[nodiscard]] constexpr std::size_t transposed_index(int dim, int head, int token, int tokens) {
    return static_cast<std::size_t>(token) +
           static_cast<std::size_t>(tokens) *
               (static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(head) +
                static_cast<std::size_t>(dim));
}

// Size contract, checked at compile time (no device needed): this is the capacity statement the
// runtime's workspace plan will use, so it must not drift.
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, 1) == 2048);
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, 7) == 14336);
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, 2048) == 4194304);
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, 1) ==
              sizeof(std::uint16_t) * static_cast<std::size_t>(kKVSize));
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(0, 8) == 0);
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(-4, 8) == 0);
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, 0) == 0);
static_assert(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, -1) == 0);

// The two index expressions must agree at tokens == 1 -- that IS the blind spot, stated as
// arithmetic: for a single token there is only one placement.
static_assert(shadow_index(255, 3, 0) == transposed_index(255, 3, 0, 1));
static_assert(shadow_index(255, 3, 0) == kKVSize - 1);
// ... and they must differ for tokens > 1, otherwise no case could ever see a transpose.
static_assert(shadow_index(0, 0, 1) != transposed_index(0, 0, 1, 7));
static_assert(transposed_index(0, 0, 1, 7) == 1);
static_assert(shadow_index(0, 0, 1) == kKVSize);

// --- deterministic input -----------------------------------------------------------------------
// Finite BF16 by construction: the exponent field is masked to 0x7f so the FP64 oracle stays
// finite. inf (0x7f80 / 0xff80) and the NaN payloads are excluded on purpose -- a bit-exact copy
// would carry them, but they would make the reduction criterion meaningless (non-finite values).
// A handful of exact bit patterns are planted afterwards, including the awkward ones: both zeros,
// the smallest subnormal, the largest finite magnitude of either sign, and +-1.
std::vector<std::uint16_t> make_raw_key_bits(int tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kKVSize) * tokens, 0);
    std::uint32_t state = seed * 2654435761u + 1u;
    for (std::size_t index = 0; index < bits.size(); ++index) {
        state     = state * 1664525u + 1013904223u;
        const auto raw = static_cast<std::uint16_t>(state >> 16);
        // Mask the sign bit and the top exponent bit, then put the sign back: exponent <= 0x7f.
        bits[index] = static_cast<std::uint16_t>((raw & 0x7f7fu) | (raw & 0x8000u));
    }
    const std::size_t planted = std::min<std::size_t>(bits.size(), 8);
    const std::uint16_t specials[8] = {0x0000u, 0x8000u, 0x0001u, 0x7f7fu,
                                       0xff7fu, 0x3f80u, 0xbf80u, 0x0080u};
    for (std::size_t i = 0; i < planted; ++i) {
        bits[i] = specials[i];
    }
    return bits;
}

// --- FP64 oracle -------------------------------------------------------------------------------
// The oracle of a pure copy is the source lifted into double, but it is written by walking the
// geometry so that the placement expression is exercised here too. Any numeric transformation
// inside the op (a rescale, a re-round, a f16 detour) shows up as a non-zero rel_l2 below.
std::vector<double> shadow_oracle(const std::vector<std::uint16_t>& source_bits, int tokens) {
    std::vector<double> expected(static_cast<std::size_t>(kKVSize) * tokens, 0.0);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < kKVHeads; ++head) {
            for (int dim = 0; dim < kHeadDim; ++dim) {
                const std::size_t index = shadow_index(dim, head, token);
                expected[index]        = static_cast<double>(bf16_to_f32(source_bits[index]));
            }
        }
    }
    return expected;
}

std::vector<double> lifted(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size(), 0.0);
    for (std::size_t index = 0; index < bits.size(); ++index) {
        values[index] = static_cast<double>(bf16_to_f32(bits[index]));
    }
    return values;
}

// --- the verdict -------------------------------------------------------------------------------
// Returns 0 when the buffer IS the raw K, 1 otherwise. This single function is used for the real
// device result AND for every mutant, which is what makes the negative controls meaningful: they
// run through exactly the same verdict as the happy path.
int verify_shadow_bits(const std::string& label, const std::vector<std::uint16_t>& got_bits,
                       const std::vector<std::uint16_t>& source_bits, int tokens) {
    const std::size_t count = static_cast<std::size_t>(kKVSize) * tokens;
    if (got_bits.size() != count || source_bits.size() != count) {
        std::cerr << label << ": size mismatch got=" << got_bits.size()
                  << " source=" << source_bits.size() << " expected=" << count << '\n';
        return 1;
    }

    int failures = 0;

    // (1) PRIMARY: bit-for-bit, at the declared placement. A BF16 shadow is a raw copy, so anything
    // other than equality is a defect; this is also the only check that can see the sign of zero.
    std::size_t mismatches = 0;
    std::size_t first_bad  = count;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < kKVHeads; ++head) {
            for (int dim = 0; dim < kHeadDim; ++dim) {
                const std::size_t index = shadow_index(dim, head, token);
                if (got_bits[index] != source_bits[index]) {
                    ++mismatches;
                    if (first_bad == count) { first_bad = index; }
                }
            }
        }
    }
    if (mismatches != 0) {
        std::cerr << label << ": " << mismatches << " of " << count
                  << " elements differ from the raw K; first at " << first_bad << " (got "
                  << got_bits[first_bad] << ", raw K " << source_bits[first_bad] << ")\n";
        ++failures;
    }

    // (2) FP64 reduction against the oracle, with a zero-tolerance criterion.
    failures += verify_reduction(label + " fp64", lifted(got_bits), shadow_oracle(source_bits, tokens),
                                 kBitExact);

    // (3) Layout sensitivity probe: for tokens > 1 the received bytes must NOT also match the
    // transposed placement, otherwise this case could not detect a transpose at all. (At tokens == 1
    // matching both is arithmetically unavoidable and is asserted separately, not treated as a bug.)
    if (tokens > 1) {
        bool matches_transposed = true;
        for (int token = 0; token < tokens && matches_transposed; ++token) {
            for (int head = 0; head < kKVHeads && matches_transposed; ++head) {
                for (int dim = 0; dim < kHeadDim; ++dim) {
                    if (got_bits[transposed_index(dim, head, token, tokens)] !=
                        source_bits[shadow_index(dim, head, token)]) {
                        matches_transposed = false;
                        break;
                    }
                }
            }
        }
        if (matches_transposed) {
            std::cerr << label
                      << ": buffer matches the transposed [tokens, kv_size] placement too, so this "
                         "case cannot see a layout error (needs tokens > 1 and non-degenerate "
                         "data)\n";
            ++failures;
        }
    }
    return failures;
}

// --- mutants (the negative controls) -----------------------------------------------------------
using Mutant = std::vector<std::uint16_t> (*)(const std::vector<std::uint16_t>&, int);

std::vector<std::uint16_t> mutant_transposed(const std::vector<std::uint16_t>& source, int tokens) {
    std::vector<std::uint16_t> out(source.size(), kSentinelBits);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < kKVHeads; ++head) {
            for (int dim = 0; dim < kHeadDim; ++dim) {
                out[transposed_index(dim, head, token, tokens)] = source[shadow_index(dim, head, token)];
            }
        }
    }
    return out;
}

std::vector<std::uint16_t> mutant_missing_last_token(const std::vector<std::uint16_t>& source,
                                                     int tokens) {
    std::vector<std::uint16_t> out = source;
    if (tokens <= 1) { return out; }
    std::fill(out.begin() + static_cast<std::ptrdiff_t>(kKVSize) * (tokens - 1), out.end(),
              kSentinelBits);
    return out;
}

std::vector<std::uint16_t> mutant_shifted(const std::vector<std::uint16_t>& source, int tokens) {
    (void)tokens;
    std::vector<std::uint16_t> out = source;
    std::rotate(out.begin(), out.begin() + 1, out.end()); // every value one slot too far
    return out;
}

std::vector<std::uint16_t> mutant_one_bit_flipped(const std::vector<std::uint16_t>& source,
                                                  int tokens) {
    std::vector<std::uint16_t> out = source;
    // The smallest perturbation that is still a perturbation: one mantissa LSB of one element in
    // the middle of the buffer. XOR guarantees the bits differ even if the element happens to be a
    // planted special; a "clear the LSB" mutation could be a no-op on 0x0000.
    const std::size_t index = static_cast<std::size_t>(kKVSize) * (tokens / 2) + 313;
    out[index]              = static_cast<std::uint16_t>(out[index] ^ 0x0001u);
    return out;
}

// --- host-only checks --------------------------------------------------------------------------
// The polarity documented in kvmem_shadow.h is re-derived from the environment here rather than
// assumed, so a deliberate flip of the default is caught, while running the test WITH the variable
// set is not reported as a failure.
int verify_switch_semantics() {
    const char* value    = std::getenv("NINFER_TERNARY_KVMEM");
    const bool expected  = value != nullptr && std::string(value) == "1";
    const bool actual    = ops::detail::kvmem_shadow_enabled();
    if (actual == expected) { return 0; }
    std::cerr << "kvmem_raw_k_shadow: NINFER_TERNARY_KVMEM semantics drifted: env="
              << (value != nullptr ? value : "(unset)") << " expected=" << expected
              << " actual=" << actual << '\n';
    return 1;
}

int verify_host_negative_controls() {
    int failures = 0;
    const int tokens       = 7; // > 1 on purpose
    const auto source_bits = make_raw_key_bits(tokens, 0x91u);

    struct Case {
        const char* name;
        Mutant make; // nullptr == the unmutated buffer (positive control)
    };
    const Case cases[] = {
        {"positive control (raw K itself)", nullptr},
        {"transposed [tokens, kv_size] layout", mutant_transposed},
        {"last token row dropped", mutant_missing_last_token},
        {"one-slot shift", mutant_shifted},
        {"a single flipped mantissa bit", mutant_one_bit_flipped},
    };

    int rejected = 0;
    for (const Case& item : cases) {
        const std::vector<std::uint16_t> got = item.make == nullptr ? source_bits
                                                                   : item.make(source_bits, tokens);
        const int verdict = verify_shadow_bits(std::string("control: ") + item.name, got, source_bits,
                                               tokens);
        if (item.make == nullptr) {
            if (verdict != 0) {
                std::cerr << "kvmem_raw_k_shadow: the POSITIVE control was rejected -- the verdict "
                             "function is broken, so nothing it accepts means anything\n";
                ++failures;
            }
            continue;
        }
        if (verdict == 0) {
            std::cerr << "kvmem_raw_k_shadow: NEGATIVE CONTROL PASSED (it must be rejected): "
                      << item.name << '\n';
            ++failures;
            continue;
        }
        ++rejected;
        std::cout << "kvmem_raw_k_shadow: negative control rejected as required: " << item.name
                  << '\n';
    }

    // The T=1 blind spot, asserted instead of tolerated: at a single token the transposed mutant IS
    // the correct buffer, so a tokens==1-only suite proves nothing about the layout, and it must
    // NOT be "caught" here (if it were, the two index expressions would contradict themselves).
    const int one_token = 1;
    const auto source_one = make_raw_key_bits(one_token, 0x92u);
    const auto transposed_one = mutant_transposed(source_one, one_token);
    if (transposed_one != source_one) {
        std::cerr << "kvmem_raw_k_shadow: at tokens=1 the transposed placement is not the declared "
                     "one -- the index expressions are inconsistent\n";
        ++failures;
    }
    if (verify_shadow_bits("blind spot tokens=1", transposed_one, source_one, one_token) != 0) {
        std::cerr << "kvmem_raw_k_shadow: tokens=1 unexpectedly detected a transpose\n";
        ++failures;
    }
    std::cout << "kvmem_raw_k_shadow: tokens=1 is layout-blind by construction (asserted); the "
              << rejected << " rejected controls above ran at tokens=" << tokens << '\n';
    return failures;
}

// --- device cases ------------------------------------------------------------------------------
template <typename Callable>
int expect_invalid_argument(const std::string& label, Callable&& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << ": threw the wrong exception type: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": no exception was thrown, so the contract was not enforced\n";
    return 1;
}

// One full device case: allocate, copy, synchronise, read back, judge -- plus the workspace-arena
// path (the byte function used AS an allocation), the read-only property of the source, and the
// guard canaries on both buffers.
int verify_device_case(int tokens, std::uint32_t seed) {
    const std::string label               = "T=" + std::to_string(tokens);
    const std::vector<std::uint16_t> bits = make_raw_key_bits(tokens, seed);
    const std::size_t count               = bits.size();
    const std::size_t shadow_bytes =
        static_cast<std::size_t>(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, tokens));

    int failures = 0;

    GuardedDeviceBuffer source(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer shadow(shadow_bytes);
    source.copy_from_host(bits.data(), count * sizeof(std::uint16_t));
    shadow.fill(kSentinelByte);

    Tensor raw_key(source.data(), DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor shadow_tensor(shadow.data(), DType::BF16, {kKVSize, tokens});
    ops::raw_k_shadow_copy(raw_key, shadow_tensor, nullptr);
    cuda_synchronize();

    failures += verify_shadow_bits(label, from_device<std::uint16_t>(shadow.data(), count), bits,
                                   tokens);
    // The op must be strictly read-only on its input: the raw K it is handed must come back
    // unchanged, or the harvest would silently corrupt the model.
    const std::string readonly_label = label + " raw K is read-only";
    failures += verify_exact(readonly_label.c_str(),
                             from_device<std::uint16_t>(source.data(), count), bits);
    failures += source.verify_guards(label + " source");
    failures += shadow.verify_guards(label + " shadow");

    // The call site's own view algebra: the same bytes declared {kv_size, tokens} must address the
    // same elements in the same order as {head_dim, kv_heads, tokens}. This is the step that would
    // silently transpose the harvest if the strides were token-major.
    const Tensor raw_flat = raw_key.view({kKVSize, tokens});
    const std::string view_label = label + " {kv_size, tokens} view of the raw K";
    failures += verify_exact(view_label.c_str(), from_device<std::uint16_t>(raw_flat.data, count),
                             from_device<std::uint16_t>(shadow.data(), count));

    // The byte function is the capacity statement segment 2 will use, so exercise it AS an
    // allocation: an undersized capacity would show up here as a torn tail (the arena's own
    // alignment is inside alloc_bytes, which is why the byte count is exact rather than padded).
    GuardedDeviceBuffer scratch(shadow_bytes);
    {
        WorkspaceArena arena(DeviceSpan{scratch.data(), scratch.bytes()});
        Tensor arena_shadow = arena.alloc(DType::BF16, {kKVSize, tokens});
        ops::raw_k_shadow_copy(raw_key, arena_shadow, nullptr);
        cuda_synchronize();
        failures += verify_shadow_bits(label + " via workspace arena",
                                       from_device<std::uint16_t>(arena_shadow.data, count), bits,
                                       tokens);
        // The contract is "the declared bytes are enough"; the arena's internal 256-byte alignment
        // is allowed to make its own offset differ, so only over-consumption is a failure.
        std::cout << label << ": arena consumed " << arena.used() << " of the declared "
                  << shadow_bytes << " bytes\n";
        if (arena.peak_used() > shadow_bytes) {
            std::cerr << label << ": arena needed " << arena.peak_used()
                      << " bytes for a " << shadow_bytes << "-byte shadow\n";
            ++failures;
        }
    }
    failures += scratch.verify_guards(label + " arena scratch");
    return failures;
}

// Shape/dtype enforcement. These run on the device only because they need buffers, not because they
// need a launch: a rejected call must throw BEFORE anything is issued.
int verify_validation(int tokens) {
    int failures = 0;
    const std::vector<std::uint16_t> bits =
        make_raw_key_bits(tokens, 0xb1u); // tokens > 1: needed for the non-contiguous case
    const std::size_t count        = bits.size();
    const std::size_t shadow_bytes = count * sizeof(std::uint16_t);
    GuardedDeviceBuffer source(shadow_bytes);
    GuardedDeviceBuffer shadow(shadow_bytes);
    source.copy_from_host(bits.data(), shadow_bytes);

    Tensor raw_key(source.data(), DType::BF16, {kHeadDim, kKVHeads, tokens});

    failures += expect_invalid_argument("shadow with the wrong token count", [&] {
        GuardedDeviceBuffer wide(shadow_bytes * 2);
        Tensor bad(wide.data(), DType::BF16, {kKVSize, tokens + 1});
        ops::raw_k_shadow_copy(raw_key, bad, nullptr);
    });
    failures += expect_invalid_argument("shadow declared {tokens, kv_size} (the transpose)", [&] {
        Tensor bad(shadow.data(), DType::BF16, {tokens, kKVSize});
        ops::raw_k_shadow_copy(raw_key, bad, nullptr);
    });
    failures += expect_invalid_argument("shadow with the wrong dtype", [&] {
        Tensor bad(shadow.data(), DType::FP32, {kKVSize, tokens});
        ops::raw_k_shadow_copy(raw_key, bad, nullptr);
    });
    failures += expect_invalid_argument("raw key with a mismatched token axis", [&] {
        Tensor bad_key(source.data(), DType::BF16, {kHeadDim, kKVHeads, tokens + 1});
        Tensor good(shadow.data(), DType::BF16, {kKVSize, tokens});
        ops::raw_k_shadow_copy(bad_key, good, nullptr);
    });
    failures += expect_invalid_argument("non-contiguous shadow", [&] {
        Tensor good(shadow.data(), DType::BF16, {kKVSize, tokens});
        Tensor permuted = good.permute({1, 0, 3, 2}); // {tokens, kv_size}: not contiguous for T>1
        ops::raw_k_shadow_copy(raw_key, permuted, nullptr);
    });
    failures += expect_invalid_argument("an empty host store", [&] {
        ops::RawKShadowHostStore empty(0);
        (void)empty;
    });
    // The undersized-store check lives behind the drain, and the drain returns FALSE before it ever
    // looks at the store when the harvest is switched off (its host-side switch gate). So this
    // control is only meaningful in the ON arm -- and saying so is better than pretending.
    if (ops::detail::kvmem_shadow_enabled()) {
        failures += expect_invalid_argument("a host store that is too small", [&] {
            ops::RawKShadowHostStore small(shadow_bytes / 2);
            Tensor good(shadow.data(), DType::BF16, {kKVSize, tokens});
            (void)ops::raw_k_shadow_drain(good, small, nullptr);
        });
    } else {
        std::cout << "kvmem_raw_k_shadow: note -- the undersized-store control needs "
                     "NINFER_TERNARY_KVMEM=1 (the drain refuses before the size check when the "
                     "harvest is off)\n";
    }
    return failures;
}

// Capture: the op must be legal INSIDE a graph capture, the drain must refuse there, and the
// captured graph must deliver the CURRENT staging contents on every replay (pointer indirection,
// not a frozen copy).
int verify_capture_case(int tokens, std::uint32_t seed) {
    const std::string label = "capture T=" + std::to_string(tokens);
    const std::size_t count = static_cast<std::size_t>(kKVSize) * tokens;
    const std::size_t shadow_bytes =
        static_cast<std::size_t>(ops::detail::kvmem_raw_k_shadow_bytes(kKVSize, tokens));
    const bool enabled = ops::detail::kvmem_shadow_enabled();

    int failures = 0;

    GuardedDeviceBuffer source(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer shadow(shadow_bytes);
    // Built OUTSIDE the capture: the store's constructor allocates pinned host memory, which is
    // illegal while a capture is open (the same reason the drain refuses inside one).
    ops::RawKShadowHostStore host_store(shadow_bytes);

    std::vector<std::uint16_t> bits = make_raw_key_bits(tokens, seed);
    source.copy_from_host(bits.data(), count * sizeof(std::uint16_t));
    shadow.fill(kSentinelByte);
    Tensor raw_key(source.data(), DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor shadow_tensor(shadow.data(), DType::BF16, {kKVSize, tokens});

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "raw-K shadow graph stream");
    cuda_synchronize();

    DecodeGraphDefinition definition;
    bool drained_during_capture = true; // forced to false by the drain's capture check
    definition.capture(stream, [&] {
        ops::raw_k_shadow_copy(raw_key, shadow_tensor, stream);
        drained_during_capture = ops::raw_k_shadow_drain(shadow_tensor, host_store, stream);
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);

    if (drained_during_capture) {
        std::cerr << label << ": the drain copied device->host while the stream was capturing\n";
        ++failures;
    }
    if (!enabled) {
        std::cout << label << ": note -- NINFER_TERNARY_KVMEM is off in this process, so the drain "
                             "refused for the switch reason and the capture check was not exercised; "
                             "re-run with NINFER_TERNARY_KVMEM=1 to cover that path\n";
    }

    for (int replay = 0; replay < 3; ++replay) {
        bits = make_raw_key_bits(tokens, seed + 0x11u * static_cast<std::uint32_t>(replay + 1));
        cuda_synchronize(stream);
        source.copy_from_host(bits.data(), count * sizeof(std::uint16_t));
        shadow.fill(kSentinelByte);
        executable.launch(stream);
        cuda_synchronize(stream);
        failures += verify_shadow_bits(label + " replay " + std::to_string(replay),
                                       from_device<std::uint16_t>(shadow.data(), count), bits, tokens);
        const std::string readonly_label = label + " replay " + std::to_string(replay) +
                                          " raw K is read-only";
        failures += verify_exact(readonly_label.c_str(),
                                 from_device<std::uint16_t>(source.data(), count), bits);
    }
    failures += shadow.verify_guards(label + " shadow");
    failures += source.verify_guards(label + " source");

    // A legal drain, outside the capture. The pinned buffer is pre-filled so that "did the drain
    // write anything" is observable in both arms.
    std::memset(host_store.data(), kSentinelByte, shadow_bytes);
    const bool drained = ops::raw_k_shadow_drain(shadow_tensor, host_store, stream);
    if (drained != enabled) {
        std::cerr << label << ": the drain returned " << drained << " with the switch "
                  << (enabled ? "on" : "off") << '\n';
        ++failures;
    }
    const auto* host_bytes = static_cast<const std::uint8_t*>(host_store.data());
    bool untouched = true;
    for (std::size_t i = 0; i < shadow_bytes; ++i) {
        if (host_bytes[i] != kSentinelByte) {
            untouched = false;
            break;
        }
    }
    if (enabled) {
        std::vector<std::uint16_t> host_bits(count, 0);
        std::memcpy(host_bits.data(), host_bytes, count * sizeof(std::uint16_t));
        const std::string drain_label = label + " drained host bytes";
        failures += verify_exact(drain_label.c_str(), host_bits, bits);
    } else if (!untouched) {
        std::cerr << label << ": the OFF arm wrote to the pinned host buffer\n";
        ++failures;
    }

    cuda_check(cudaStreamDestroy(stream), "destroy raw-K shadow graph stream");
    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;

        // Host-only checks first: a machine without a usable device must still run the negative
        // controls, otherwise "SKIP" would hide whether the checker can see anything at all.
        failures += verify_switch_semantics();
        failures += verify_host_negative_controls();
        std::cout << "kvmem_raw_k_shadow: geometry kv_heads=" << kKVHeads
                  << " head_dim=" << kHeadDim << " kv_size=" << kKVSize
                  << "; NINFER_TERNARY_KVMEM="
                  << (ops::detail::kvmem_shadow_enabled() ? "1 (harvest ON)" : "off (default)")
                  << '\n';
        if (failures != 0) {
            std::cerr << "kvmem_raw_k_shadow: host self-check FAILED failures=" << failures << '\n';
            return 1;
        }
        std::cout << "kvmem_raw_k_shadow: host self-check PASS\n";

        if (cuda_unavailable()) {
            std::cout << "kvmem_raw_k_shadow: SKIP (CUDA unavailable)\n";
            return 77;
        }

        // tokens = 1 (the degenerate shape the real decode path uses), 7 (prime, so no power of two
        // can hide an axis error), 64, and 2048 (the prefill ceiling the planner allows).
        for (const int tokens : {1, 7, 64, 2048}) {
            failures += verify_device_case(tokens, 0xa0u + static_cast<std::uint32_t>(tokens));
        }
        failures += verify_validation(7);
        failures += verify_capture_case(7, 0xc7u);
        failures += verify_capture_case(64, 0xc8u);

        if (failures != 0) {
            std::cerr << "kvmem_raw_k_shadow failures=" << failures << '\n';
            return 1;
        }
        std::cout << "kvmem_raw_k_shadow: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kvmem_raw_k_shadow: " << error.what() << '\n';
        return 1;
    }
}
