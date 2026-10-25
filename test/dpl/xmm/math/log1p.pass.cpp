// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=16000000
// @dpl[msvc].compile-flags+: /constexpr:steps16000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.unary;

#include "log1p/bf16.h"
#include "log1p/f16.h"
#include "log1p/f32.h"
#include "log1p/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::log1p<abi_t, 0.75>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
