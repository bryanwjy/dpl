// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=38000000
// @dpl[gcc].compile-flags+: -fconstexpr-ops-limit=38000000
// @dpl[msvc].compile-flags+: /constexpr:steps38000000

// @dpl.mpfr+: pow f32 -n 12 \
// -r linear 0.5:2 -4:4 -r linear 0.999:1.001 -10000:10000 \
// -r urand 0x1p-8:0x1p8 -8:8 -r linear -4:-0.25 -4:4 -r urand 0x1p-126:1 0:1 \
// -r linear 1.99:2 120:128 -r linear 2:2.01 128:256 \
// -r linear 2:2.01 -149:-126 -r linear 2:2.01 -300:-150 -r urand : : -i nan 0 \
// -i infinity -0 -i 1 nan -i 1 infinity -i 1 -infinity -i -1 infinity \
// -i -1 -infinity -i 0 -1 -i -0 -1 -i -0 -2 -i 0 -0.5 -i -0 -infinity -i 0 3 \
// -i -0 3 -i -0 2 -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 -i -2 0.5 -i -1.5 31 \
// -i -1 0x1p14 -i 0.5 infinity -i 0.5 -infinity -i 2 infinity -i 2 -infinity \
// -i -infinity 3 -i -infinity 2 -i -infinity -3 -i -infinity -2 \
// -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 -i 4 0.5 -i 9 0.5 \
// -i 10 2 -i 3 -2

// @dpl.mpfr+: pow f64 -n 12 \
// -r linear 0.5:2 -4:4 -r linear 0.999:1.001 -100000:100000 \
// -r urand 0x1p-8:0x1p8 -8:8 -r linear -4:-0.25 -4:4 -r urand 0x1p-1022:1 0:1 \
// -r linear 1.99:2 1016:1024 -r linear 2:2.01 1024:2048 \
// -r linear 2:2.01 -1074:-1022 -r linear 2:2.01 -2150:-1075 -r urand : : \
// -i nan 0 -i infinity -0 -i 1 nan -i 1 infinity -i 1 -infinity \
// -i -1 infinity -i -1 -infinity -i 0 -1 -i -0 -1 -i -0 -2 -i 0 -0.5 \
// -i -0 -infinity -i 0 3 -i -0 3 -i -0 2 -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 \
// -i -2 0.5 -i -1.5 31 -i -1 0x1p14 -i 0.5 infinity -i 0.5 -infinity \
// -i 2 infinity -i 2 -infinity -i -infinity 3 -i -infinity 2 -i -infinity -3 \
// -i -infinity -2 -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 \
// -i 4 0.5 -i 9 0.5 -i 10 2 -i 3 -2

// @dpl.mpfr+: pow f16 -n 12 \
// -r linear 0.5:2 -4:4 -r linear 0.99:1.01 -500:500 \
// -r urand 0x1p-4:0x1p4 -3:3 -r linear -4:-0.25 -4:4 -r urand 0x1p-14:1 0:1 \
// -r linear 1.99:2 8:16 -r linear 2:2.01 16:32 -r linear 2:2.01 -24:-14 \
// -r linear 2:2.01 -50:-25 -r urand : : -i nan 0 -i infinity -0 -i 1 nan \
// -i 1 infinity -i 1 -infinity -i -1 infinity -i -1 -infinity -i 0 -1 \
// -i -0 -1 -i -0 -2 -i 0 -0.5 -i -0 -infinity -i 0 3 -i -0 3 -i -0 2 \
// -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 -i -2 0.5 -i -1.5 31 -i -1 0x1p14 \
// -i 0.5 infinity -i 0.5 -infinity -i 2 infinity -i 2 -infinity \
// -i -infinity 3 -i -infinity 2 -i -infinity -3 -i -infinity -2 \
// -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 -i 4 0.5 -i 9 0.5 \
// -i 10 2 -i 3 -2

// @dpl.mpfr+: pow bf16 -n 12 \
// -r linear 0.5:2 -4:4 -r linear 0.97:1.03 -200:200 \
// -r urand 0x1p-8:0x1p8 -8:8 -r linear -4:-0.25 -4:4 -r urand 0x1p-126:1 0:1 \
// -r linear 1.99:2 120:128 -r linear 2:2.01 128:256 \
// -r linear 2:2.01 -133:-126 -r linear 2:2.01 -268:-134 -r urand : : -i nan 0 \
// -i infinity -0 -i 1 nan -i 1 infinity -i 1 -infinity -i -1 infinity \
// -i -1 -infinity -i 0 -1 -i -0 -1 -i -0 -2 -i 0 -0.5 -i -0 -infinity -i 0 3 \
// -i -0 3 -i -0 2 -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 -i -2 0.5 -i -1.5 31 \
// -i -1 0x1p14 -i 0.5 infinity -i 0.5 -infinity -i 2 infinity -i 2 -infinity \
// -i -infinity 3 -i -infinity 2 -i -infinity -3 -i -infinity -2 \
// -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 -i 4 0.5 -i 9 0.5 \
// -i 10 2 -i 3 -2

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
