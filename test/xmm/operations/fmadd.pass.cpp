// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test;

// Tests for dpp::fmadd on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::fmadd(a, b, c)                  -- unmasked, always available
//   (2) dpp::fmadd(src, mask, a, b, c)       -- merge-masked
//   (3) dpp::fmadd(dpp::zero, mask, a, b, c) -- zero-masked explicit
//   (4) dpp::fmadd(mask, a, b, c)            -- zero-masked alias (== form 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    static constexpr auto expected = [](auto l, auto m, auto r) {
        return l * m + r;
    };
    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    static_assert([]() {
        dpl::test::mt19937 engine;
        return dpl::test::fused_multiply<abi_t>::run_all<dpp::fmadd>(
            types{}, expected, engine);
    }());

    dpl::test::mt19937 engine;
    assert(dpl::test::fused_multiply<abi_t>::run_all<dpp::fmadd>(
        types{}, expected, engine));
    return 0;
}
