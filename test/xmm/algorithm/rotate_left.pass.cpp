// Copyright 2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=12000000
// @dpl[msvc].compile-flags: /constexpr:steps12000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.algorithm.rotate;

// Tests for dpp::rotate_left on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
// (1) dpp::rotate_left(lhs, rhs)                  -- unmasked, always available
// (2) dpp::rotate_left(src, mask, lhs, rhs)       -- merge-masked
// (3) dpp::rotate_left(dpp::zero, mask, lhs, rhs) -- zero-masked explicit
// (4) dpp::rotate_left(mask, lhs, rhs)            -- zero-masked alias (== form
// 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        if consteval {
            // Reduce number of types to reduce compile time steps
            using types =
                dpl::type_pack<dpl::int8, dpl::int16, dpl::int32, dpl::int64>;
            return dpl::test::rotate_left<abi_t>::run_all(types{}, engine);
        } else {
            using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16,
                dpl::uint16, dpl::int32, dpl::uint32, dpl::int64, dpl::uint64,
                float, double, dpl::ext::float16, dpl::ext::bfloat16>;
            return dpl::test::rotate_left<abi_t>::run_all(types{}, engine);
        }
    };

    static_assert(run());
    assert(run());
    return 0;
}
