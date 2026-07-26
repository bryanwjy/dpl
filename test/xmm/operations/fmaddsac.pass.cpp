// Copyright 2025-2026 Bryan Wong
// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=2500000
// @dpl[msvc].compile-flags: /constexpr:steps2500000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.fused_multiply;

// Tests for dpp::fmaddsac on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::fmaddsac(a, b, c)                  -- unmasked, always available
//   (2) dpp::fmaddsac(src, mask, a, b, c)       -- merge-masked
//   (3) dpp::fmaddsac(dpp::zero, mask, a, b, c) -- zero-masked explicit
//   (4) dpp::fmaddsac(mask, a, b, c)            -- zero-masked alias (== form
//   3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    static_assert([]() {
        dpl::test::mt19937 engine;
        return dpl::test::fmaddsac<abi_t>::run_all(types{}, engine);
    }());

    dpl::test::mt19937 engine;
    assert(dpl::test::fmaddsac<abi_t>::run_all(types{}, engine));
    return 0;
}
