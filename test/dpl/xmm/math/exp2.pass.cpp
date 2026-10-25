// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=12000000
// @dpl[msvc].compile-flags+: /constexpr:steps12000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.unary;

#include "exp2/bf16.h"
#include "exp2/f16.h"
#include "exp2/f32.h"
#include "exp2/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::exp2<abi_t, 0.75>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
