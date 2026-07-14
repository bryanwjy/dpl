// Copyright 2025-2026 Bryan Wong

// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=2000000
// @dpl[msvc].compile-flags: /constexpr:steps2000000

#include "../common.h"

import dpl.xmm;
import dpl.test;

int main() {
    using abi_t = dpl::xmm::abi_tag;
    static_assert(dpl::test::fp_int_cast<abi_t>::run_all());
    assert(dpl::test::fp_int_cast<abi_t>::run_all());
    return 0;
}
