// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.combo_multiply;

// Tests for dpp::nmuladd on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::nmuladd(a, b, c)                  -- unmasked, always available
//   (2) dpp::nmuladd(src, mask, a, b, c)       -- merge-masked
//   (3) dpp::nmuladd(dpp::zero, mask, a, b, c) -- zero-masked explicit
//   (4) dpp::nmuladd(mask, a, b, c)            -- zero-masked alias (== form 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::nmuladd<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
