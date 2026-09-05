// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=8000000
// @dpl[msvc].compile-flags: /constexpr:steps8000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.arithmetic;

// Tests for dpp::subadd on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::subadd(lhs, rhs)                  -- unmasked, always available
//   (2) dpp::subadd(src, mask, lhs, rhs)       -- merge-masked
//   (3) dpp::subadd(dpp::zero, mask, lhs, rhs) -- zero-masked explicit
//   (4) dpp::subadd(mask, lhs, rhs)            -- zero-masked alias (== form 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16, dpl::uint16,
        dpl::int32, dpl::uint32, dpl::int64, dpl::uint64, float, double,
        dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::subaddition<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
