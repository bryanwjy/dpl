// Copyright 2025-2026 Bryan Wong

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.isunordered;

// Tests for dpp::cmpneq on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::cmpunord(lhs, rhs)        -- unmasked, always available
//   (2) dpp::cmpunord(mask, lhs, rhs)  -- merge-masked

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::isunordered<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
