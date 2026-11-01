// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=18000000
// @dpl[msvc].compile-flags+: /constexpr:steps18000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.rsqrt;

#include "rsqrt/bf16.h"
#include "rsqrt/f16.h"
#include "rsqrt/f32.h"
#include "rsqrt/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::rsqrt_approx<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
