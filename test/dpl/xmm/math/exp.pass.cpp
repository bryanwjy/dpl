// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags+: -fconstexpr-steps=15000000
// @dpl[msvc].compile-flags+: /constexpr:steps15000000
// @dpl.mpfr+: exp f32 -n 12 -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:32 -r linear -32:-1 -r linear -87.3:88.72 -r linear 87.7:88.72 \
// -r linear 88.73:178 -r linear -103.97:-87.34 -r linear -208:-103.98 \
// -r urand :

// 0x1.62e42fefa39efp+9  -0x1.74910d52d3051p+9
// @dpl.mpfr+: exp f64 -n 12 -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:32 -r linear -32:-1 -r linear -708.3:709.78 \
// -r linear 708.7:709.78 -r linear 709.79:1420 -r linear -745.13:-708.4 \
// -r linear -1490:-745.14 -r urand :

// @dpl.mpfr+: exp f16 -n 12 -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:8 -r linear -8:-1 -r linear -9.7:11.08 -r linear 10:11.08 \
// -r linear 11.1:22 -r linear -17.32:-9.71 -r linear -35:-17.33 \
// -r urand :

// @dpl.mpfr+: exp bf16 -n 12 -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 \
// -r linear 1:32 -r linear -32:-1 -r linear -87.3:88.5 -r linear 87.5:88.5 \
// -r linear 89:178 -r linear -92.88:-87.34 -r linear -186:-92.89 -r urand :

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
