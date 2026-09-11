// Copyright 2025-2026 Bryan Wong
// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=4000000
// @dpl[msvc].compile-flags: /constexpr:steps4000000

#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test.harness.operations.combo_multiply;

// Tests for dpp::muladdsac on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::muladdsac(a, b, c)                  -- unmasked, always available
//   (2) dpp::muladdsac(src, mask, a, b, c)       -- merge-masked
//   (3) dpp::muladdsac(dpp::zero, mask, a, b, c) -- zero-masked explicit
//   (4) dpp::muladdsac(mask, a, b, c)            -- zero-masked alias (== form
//   3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::muladdsac<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
