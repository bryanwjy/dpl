// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=2000000
// @dpl[msvc].compile-flags+: /constexpr:steps2000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.rounding;

#include "trunc/bf16.h"
#include "trunc/f16.h"
#include "trunc/f32.h"
#include "trunc/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::trunc<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
