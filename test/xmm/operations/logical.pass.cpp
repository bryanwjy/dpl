// Copyright 2025-2026 Bryan Wong
// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=2000000
// @dpl[msvc].compile-flags: /constexpr:steps2000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.logical;

// Tests for logical reduction (all_of, any_of, none_of, some_of) on xmm ABI.
// Requires SSE4.2 (implied by dpl.xmm).

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types = dpl::type_pack<dpl::int8, dpl::int16, dpl::int32, dpl::int64>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::logical<abi_t>::run_all(types{}, engine);
    };

    // static_assert(run());
    assert(run());
    return 0;
}
