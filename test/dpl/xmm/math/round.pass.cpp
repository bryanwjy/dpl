// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=6000000
// @dpl[msvc].compile-flags+: /constexpr:steps6000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.rounding;

#include "round/bf16.h"
#include "round/f16.h"
#include "round/f32.h"
#include "round/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types = dpl::type_pack<float>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::round<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
