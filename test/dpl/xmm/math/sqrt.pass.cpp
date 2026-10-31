// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=30000000
// @dpl[msvc].compile-flags+: /constexpr:steps30000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.unary;

#include "sqrt/bf16.h"
#include "sqrt/f16.h"
#include "sqrt/f32.h"
#include "sqrt/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::sqrt<abi_t, 0.5>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
