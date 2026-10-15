// Copyright 2025-2026 Bryan Wong
#include "../common.h"

import dpl.xmm;
import dpl.test.harness.operations.element_cast;

int main() {
    using abi_t = dpl::xmm::abi_tag;
    static_assert(dpl::test::fp_fp_cast<abi_t>::run_all());
    assert(dpl::test::fp_fp_cast<abi_t>::run_all());

    return 0;
}
