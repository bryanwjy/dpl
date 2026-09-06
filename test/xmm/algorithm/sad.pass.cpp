// Copyright 2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=8000000
// @dpl[msvc].compile-flags: /constexpr:steps8000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.algorithm.sad;

// Tests for dpp::sad on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
// Only defined for integrals
//
// Forms tested:
//   (1) dpp::sad(src, lhs, rhs)
//   (2) dpp::sad(src, mask, lhs, rhs)
//   (3) dpp::sad(dpp::zero, mask, src, lhs, rhs)
//   (4) dpp::sad(mask, src, lhs, rhs)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        if consteval {
            // Reduce number of types to reduce compile time steps
            using types = dpl::type_pack<dpl::uint32, dpl::uint64>;
            return dpl::test::sad<abi_t>::run_all(types{}, engine);
        } else {
            using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16,
                dpl::uint16, dpl::int32, dpl::uint32, dpl::int64, dpl::uint64>;
            return dpl::test::sad<abi_t>::run_all(types{}, engine);
        }
    };

    static_assert(run());
    assert(run());
    return 0;
}
