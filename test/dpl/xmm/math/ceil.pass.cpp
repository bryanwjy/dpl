// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=6000000
// @dpl[msvc].compile-flags+: /constexpr:steps6000000
// @dpl.mpfr+: ceil f32 -n 12 -r urand -0x1p23:-1 -r urand 1:0x1p23 \
// -r urand -1:1 -r urand 0x1p22:infinity -r urand -infinity:-0x1p22

// @dpl.mpfr+: ceil f64 -n 12 -r urand -0x1p52:-1 -r urand 1:0x1p52 \
// -r urand -1:1 -r urand 0x1p51:infinity -r urand -infinity:-0x1p51

// @dpl.mpfr+: ceil f16 -n 12 -r urand -0x1p10:-1 -r urand 1:0x1p10 \
// -r urand -1:1 -r urand 0x1p9:infinity -r urand -infinity:-0x1p9

// @dpl.mpfr+: ceil bf16 -n 12 -r urand -0x1p7:-1 -r urand 1:0x1p7 \
// -r urand -1:1 -r urand 0x1p6:infinity -r urand -infinity:-0x1p6

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.rounding;

#include "ceil/bf16.h"
#include "ceil/f16.h"
#include "ceil/f32.h"
#include "ceil/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::ceil<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
