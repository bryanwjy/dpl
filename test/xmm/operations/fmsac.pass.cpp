// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.fused_multiply;

// Tests for dpp::fmsac on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::fmsac(a, b, c)                  -- unmasked, always available
//   (2) dpp::fmsac(src, mask, a, b, c)       -- merge-masked
//   (3) dpp::fmsac(dpp::zero, mask, a, b, c) -- zero-masked explicit
//   (4) dpp::fmsac(mask, a, b, c)            -- zero-masked alias (== form 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    static_assert([]() {
        dpl::test::mt19937 engine;
        return dpl::test::fmsac<abi_t>::run_all(types{}, engine);
    }());

    dpl::test::mt19937 engine;
    assert(dpl::test::fmsac<abi_t>::run_all(types{}, engine));
    return 0;
}
