// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=15000000
// @dpl[msvc].compile-flags+: /constexpr:steps15000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.unary;

#include "exp/bf16.h"
#include "exp/f16.h"
#include "exp/f32.h"
#include "exp/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::exp<abi_t, 0.50>::run<double>(engine) &&
            dpl::test::exp<abi_t, 0.75>::run<float>(engine) &&
            dpl::test::exp<abi_t, 0.75>::run<dpl::ext::float16>(engine) &&
            dpl::test::exp<abi_t, 0.5>::run<dpl::ext::bfloat16>(engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
