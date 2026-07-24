// Copyright 2025-2026 Bryan Wong
// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=2000000
// @dpl[msvc].compile-flags: /constexpr:steps2000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.bitwise_mask;

// Tests for dpp::bw* on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::bw*(lhs, rhs)                  -- unmasked, always available

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types = dpl::type_pack<dpl::int8, dpl::int16, dpl::int32, dpl::int64>;
    using ops = dpl::constant_type_pack<dpp::bwand, dpp::bwandnot, dpp::bwor,
        dpp::bwornot, dpp::bwxor, dpp::bwnot, dpp::bwshift_left,
        dpp::bwshift_right>;

    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::pack::all_of(
            [&](auto cbwop) {
                constexpr auto bwop = cbwop();
                return dpl::test::bitwise_mask<abi_t>::run_all<bwop>(
                    types{}, engine);
            },
            ops{});
    };

    static_assert(run());
    assert(run());
    return 0;
}
