// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=12000000
// @dpl[msvc].compile-flags+: /constexpr:steps12000000

// @dpl.mpfr+: log2 f32 -n 12 \
// -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.999:1.001 \
// -r urand 0.9375:1.0625 -r linear 1:65536 -r linear 0:1 -r urand 0x1p-126: \
// -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
// -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p127 -i 0x1.fffffep+127 \
// -i 0x1p-126 -i 0x1p-149 -i 0x1.fffffcp-127 -i 0x1.000002p+0 \
// -i 0x1.fffffep-1

// @dpl.mpfr+: log2 f64 -n 12 \
// -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.999:1.001 \
// -r urand 0.9375:1.0625 -r linear 1:65536 -r linear 0:1 -r urand 0x1p-1022: \
// -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
// -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p1023 \
// -i 0x1.fffffffffffffp+1023 -i 0x1p-1022 -i 0x1p-1074 \
// -i 0x1.ffffffffffffep-1023 -i 0x1.0000000000001p+0 -i 0x1.fffffffffffffp-1

// @dpl.mpfr+: log2 f16 -n 12 \
// -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.99:1.01 \
// -r urand 0.9375:1.0625 -r linear 1:1024 -r linear 0:1 -r urand 0x1p-14: \
// -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
// -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p15 -i 65504 -i 0x1p-14 \
// -i 0x1p-24 -i 0x1.ff8p-15 -i 0x1.004p+0 -i 0x1.ffcp-1

// @dpl.mpfr+: log2 bf16 -n 12 \
// -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.97:1.03 \
// -r urand 0.9375:1.0625 -r linear 1:65536 -r linear 0:1 -r urand 0x1p-126: \
// -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
// -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
// -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p127 -i 0x1.fep+127 \
// -i 0x1p-126 -i 0x1p-133 -i 0x1.fcp-127 -i 0x1.02p+0 -i 0x1.fep-1

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.math.unary;

#include "log2/bf16.h"
#include "log2/f16.h"
#include "log2/f32.h"
#include "log2/f64.h"

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::log2<abi_t, 0.75>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
