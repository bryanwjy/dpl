// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=8000000
// @dpl[msvc].compile-flags: /constexpr:steps8000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test;

// Tests for dpp::bwshift_left on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::bwshift_left(lhs, rhs)                  -- unmasked, always
//   available (2) dpp::bwshift_left(src, mask, lhs, rhs)       -- merge-masked
//   (3) dpp::bwshift_left(dpp::zero, mask, lhs, rhs) -- zero-masked explicit
//   (4) dpp::bwshift_left(mask, lhs, rhs)            -- zero-masked alias (==
//   form 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16, dpl::uint16,
        dpl::int32, dpl::uint32, dpl::int64, dpl::uint64, float, double,
        dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::bitwise<abi_t>::run_all<dpp::bwshift_left>(
            types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
