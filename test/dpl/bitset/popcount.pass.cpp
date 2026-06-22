// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {
template <dpl::size_t W>
constexpr void test_popcount() {
    // all zeros -> 0
    dpl::bitset<W> z;
    assert(dpl::popcount(z) == 0);

    // all ones -> W (validates padding bits are not counted)
    dpl::bitset<W> all;
    all.invert();
    assert(dpl::popcount(all) == static_cast<int>(W));

    // single bit k set -> 1, for every position
    for (auto k = 0zu; k < W; ++k) {
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::popcount(b) == 1);
    }

    // incremental build: each set() increases popcount by exactly 1
    dpl::bitset<W> b;
    for (auto r = 0zu; r < W; ++r) {
        assert(dpl::popcount(b) == static_cast<int>(r));
        b.set(r);
    }
    assert(dpl::popcount(b) == static_cast<int>(W));
}

constexpr bool run_all() {
    // integral-backed
    test_popcount<1>();
    test_popcount<5>();
    test_popcount<7>();
    test_popcount<8>();
    test_popcount<12>();
    test_popcount<16>();
    test_popcount<20>();
    test_popcount<32>();
    test_popcount<50>();
    test_popcount<64>();

    // array-backed
    test_popcount<129>();
    test_popcount<150>();
    test_popcount<192>();
    test_popcount<200>();
    test_popcount<256>();
    test_popcount<300>();
    test_popcount<320>();
    test_popcount<384>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
