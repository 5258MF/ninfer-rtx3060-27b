#include "runtime/kvmem_allocation.h"
#include <iostream>
#include <limits>
#include <stdexcept>

#if __has_include("ops/kvmem/kvmem_request_allocation.h")
#include "ops/kvmem/kvmem_request_allocation.h"
#endif

namespace {
void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
}

int main() {
    using ninfer::runtime::plan_kvmem_allocation;
    constexpr unsigned K = 1024;
    const auto large = plan_kvmem_allocation(112*K, 256*K, K, 64);
    check(large.sink_tokens == K && large.retrieval_tokens == 36*K &&
          large.output_tokens == 75*K, "256K allocation or output ceiling");
    const auto half = plan_kvmem_allocation(40*K, 128*K, 2*K, 128);
    check(half.retrieval_min == 16*K && half.retrieval_tokens == 18*K &&
          half.output_tokens == 20*K, "128K retrieval range");
    const auto tight = plan_kvmem_allocation(20*K, 256*K, 1000, 64);
    check(tight.sink_tokens == K && tight.retrieval_tokens == 11*K &&
          tight.output_tokens == 8*K, "pressure must shrink retrieval first");
    const auto big_prefix = plan_kvmem_allocation(64*K, 256*K, 20*K, 64);
    check(big_prefix.sink_tokens == 20*K && big_prefix.output_tokens == 8*K,
          "system prefix must never be truncated");
    for (unsigned unit : {64U,128U}) {
        for (unsigned prefix : {0U,1U,63U,64U,65U,127U,128U,129U,8193U}) {
            const auto a = plan_kvmem_allocation(80*K, 256*K, prefix, unit);
            check(a.sink_tokens >= prefix && a.sink_tokens-prefix <= unit,
                  "prefix alignment");
            check(a.history_tokens()+a.output_tokens == 80*K &&
                  a.output_tokens >= 8*K, "resident capacity conservation");
        }
        const auto boundary = plan_kvmem_allocation(8192+3*unit,256*K,unit,unit);
        check(boundary.retrieval_tokens == 2*unit && boundary.output_tokens == 8192,
              "minimum retrieval boundary");
        for (unsigned prefix : {13*K,std::numeric_limits<unsigned>::max()}) {
            bool rejected = false;
            try { (void)plan_kvmem_allocation(20*K,256*K,prefix,unit); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "oversized system prefix must be rejected");
        }
    }
#if __has_include("ops/kvmem/kvmem_request_allocation.h")
    using namespace ninfer::ops::detail;
    check(!kvmem_request_value("NINFER_TERNARY_KVMEM_SCORE_SINK"), "scope initially empty");
    {
        ScopedKvmemAllocation outer(37*K,K,36*K,75*K);
        try {
            ScopedKvmemAllocation inner(12*K,K,11*K,8*K);
            check(*kvmem_request_value("NINFER_TERNARY_KVMEM_SCORE_BUDGET") == 12*K,
                  "inner request budget");
            throw std::logic_error("simulate rejected execution");
        } catch (const std::logic_error&) {}
        check(*kvmem_request_value("NINFER_TERNARY_KVMEM_SCORE_BUDGET") == 37*K,
              "exception must restore outer request budget");
    }
    check(!kvmem_request_value("NINFER_TERNARY_KVMEM_SCORE_BUDGET"), "scope must not leak");
#endif
    std::cout << "KVMem allocation tests passed\n";
}
