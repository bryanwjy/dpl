// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=2000000
// @dpl[msvc].compile-flags+: /constexpr:steps2000000
// @dpl.mpfr+: trunc f32 -n 12 -r urand -0x1p23:-1 -r urand 1:0x1p23 \
// -r urand -1:1 -r urand 0x1p22:infinity -r urand -infinity:-0x1p22 \
// -o trunc-f32.generated.h

// @dpl.mpfr+: trunc f64 -n 12 -r urand -0x1p52:-1 -r urand 1:0x1p52 \
// -r urand -1:1 -r urand 0x1p51:infinity -r urand -infinity:-0x1p51 \
// -o trunc-f64.generated.h

// @dpl.mpfr+: trunc f16 -n 12 -r urand -0x1p10:-1 -r urand 1:0x1p10 \
// -r urand -1:1 -r urand 0x1p9:infinity -r urand -infinity:-0x1p9 \
// -o trunc-f16.generated.h

// @dpl.mpfr+: trunc bf16 -n 12 -r urand -0x1p7:-1 -r urand 1:0x1p7 \
// -r urand -1:1 -r urand 0x1p6:infinity -r urand -infinity:-0x1p6 \
// -o trunc-bf16.generated.h
#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.rounding;

#include "trunc-bf16.generated.h"
#include "trunc-f16.generated.h"
#include "trunc-f32.generated.h"
#include "trunc-f64.generated.h"

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
