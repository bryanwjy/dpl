// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=38000000
// @dpl[gcc].compile-flags+: -fconstexpr-ops-limit=38000000
// @dpl[msvc].compile-flags+: /constexpr:steps38000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.pow;

#include "pow/bf16.h"
#include "pow/f16.h"
#include "pow/f32.h"
#include "pow/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::pow<abi_t, 0.75>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
