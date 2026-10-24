// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=16000000
// @dpl[msvc].compile-flags+: /constexpr:steps16000000

// @dpl.mpfr+: log1p f32 -n 12 \
// -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
// -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
// -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
// -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-126 -i 0x1p-149 \
// -i -0x1p-149 -i 0x1.fffffcp-127 -i 0x1.fffffep+127 -i -0x1.fffffep-1

// @dpl.mpfr+: log1p f64 -n 12 \
// -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
// -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
// -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
// -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-1022 -i 0x1p-1074 \
// -i -0x1p-1074 -i 0x1.ffffffffffffep-1023 -i 0x1.fffffffffffffp+1023 \
// -i -0x1.fffffffffffffp-1

// @dpl.mpfr+: log1p f16 -n 12 \
// -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
// -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
// -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
// -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-14 -i 0x1p-24 \
// -i -0x1p-24 -i 0x1.ff8p-15 -i 65504 -i -0x1.ffcp-1

// @dpl.mpfr+: log1p bf16 -n 12 \
// -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
// -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
// -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
// -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-126 -i 0x1p-133 \
// -i -0x1p-133 -i 0x1.fcp-127 -i 0x1.fep+127 -i -0x1.fep-1

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
        // Is this too strict?
        return dpl::test::log1p<abi_t, 0.75>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
