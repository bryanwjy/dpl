// Copyright 2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=3000000
// @dpl[msvc].compile-flags: /constexpr:steps3000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.algorithm.scanning;

// Tests for dpp::scan on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
// (1) dpp::scan(val, func)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        if consteval {
            // Reduce number of types to scan compile time steps
            using types =
                dpl::type_pack<dpl::int8, dpl::int16, dpl::int32, float>;
            return dpl::test::scan<abi_t>::run_all(types{}, engine);
        } else {
            using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16,
                dpl::uint16, dpl::int32, dpl::uint32, dpl::int64, dpl::uint64,
                float, double, dpl::ext::float16, dpl::ext::bfloat16>;
            return dpl::test::scan<abi_t>::run_all(types{}, engine);
        }
    };

    static_assert(run());
    assert(run());
    return 0;
}
