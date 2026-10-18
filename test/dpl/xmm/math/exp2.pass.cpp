// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=10000000
// @dpl[msvc].compile-flags+: /constexpr:steps10000000
// @dpl.mpfr+: exp2 f32 -n 12 -r linear -1:1 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:32 -r linear -32:-1 -r linear -126:128 -r linear 127:128 \
// -r linear 128:256 -r linear -150:-126 -r linear -300:-150 -r urand :

// @dpl.mpfr+: exp2 f64 -n 12 -r linear -1:1 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:32 -r linear -32:-1 -r linear -1022:1024 -r linear 1023:1024 \
// -r linear 1024:2048 -r linear -1075:-1022 -r linear -2150:-1075 -r urand :

// @dpl.mpfr+: exp2 f16 -n 12 -r linear -1:1 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:15 -r linear -15:-1 -r linear -14:16 -r linear 15:16 \
// -r linear 16:32 -r linear -25:-14 -r linear -50:-25 -r urand :

// @dpl.mpfr+: exp2 bf16 -n 12 -r linear -1:1 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:32 -r linear -32:-1 -r linear -126:128 -r linear 127:128 \
// -r linear 128:256 -r linear -134:-126 -r linear -268:-134 -r urand :

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
        return dpl::test::exp2<abi_t, 1>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
