// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {
template <dpl::size_t W>
constexpr void test_set_idx() {
    // start from all-zeros, set one bit at a time and confirm only that bit
    // is set
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(b[i] == true);
        assert(dpl::popcount(b) == 1zu);
    }
}

template <dpl::size_t W>
constexpr void test_set_idx_val() {
    // set(idx, true) on an all-zero bitset
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i, true);
        assert(b[i] == true);
        assert(dpl::popcount(b) == 1zu);
    }

    // set(idx, false) on an all-ones bitset
    dpl::bitset<W> all;
    all.invert(); // all bits -> 1
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b = all;
        b.set(i, false);
        assert(b[i] == false);
        assert(dpl::popcount(b) == W - 1zu);
    }

    // set(idx, false) on an already-clear bit is a no-op
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i, false);
        assert(b[i] == false);
        assert(dpl::popcount(b) == 0zu);
    }

    // set(idx, true) on an already-set bit is a no-op
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        b.set(i, true);
        assert(b[i] == true);
        assert(dpl::popcount(b) == 1zu);
    }
}

template <dpl::size_t W>
constexpr void test_clear_idx() {
    // start from all-ones, clear one bit at a time and confirm only that bit
    // is cleared
    dpl::bitset<W> all;
    all.invert();
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b = all;
        b.clear(i);
        assert(b[i] == false);
        assert(dpl::popcount(b) == W - 1zu);
    }

    // clear(idx) on an already-clear bit is a no-op
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.clear(i);
        assert(b[i] == false);
        assert(dpl::popcount(b) == 0zu);
    }
}

template <dpl::size_t W>
constexpr void test_clear_all() {
    // clear() on an all-ones bitset produces a zero bitset
    dpl::bitset<W> b;
    b.invert();
    assert(dpl::popcount(b) == W);
    b.clear();
    assert(dpl::popcount(b) == 0zu);
    for (auto i = 0zu; i < W; ++i) {
        assert(b[i] == false);
    }

    // clear() on an already-zero bitset is a no-op
    dpl::bitset<W> z;
    z.clear();
    assert(dpl::popcount(z) == 0zu);
}

template <dpl::size_t W>
constexpr void test_invert() {
    // invert() flips every bit in a freshly-constructed zero bitset
    dpl::bitset<W> b;
    b.invert();
    assert(dpl::popcount(b) == W);
    for (auto i = 0zu; i < W; ++i) {
        assert(b[i] == true);
    }

    // double-invert round-trips to zero
    b.invert();
    assert(dpl::popcount(b) == 0zu);

    // invert() on a non-uniform pattern flips exactly the right bits
    dpl::bitset<W> p;
    for (auto i = 0zu; i < W; i += 2zu) {
        p.set(i);
    }
    auto const before_count = dpl::popcount(p);
    p.invert();
    for (auto i = 0zu; i < W; ++i) {
        assert(p[i] == (i % 2zu != 0));
    }
    assert(dpl::popcount(p) == W - before_count);

    // invert() returns *this -- chain two inverts and verify identity
    dpl::bitset<W> c;
    c.set(0);
    auto& ref = c.invert().invert();
    assert(&ref == &c);
    assert(c[0] == true);
    assert(dpl::popcount(c) == 1zu);
}

template <dpl::size_t W>
constexpr void test_all() {
    test_set_idx<W>();
    test_set_idx_val<W>();
    test_clear_idx<W>();
    test_clear_all<W>();
    test_invert<W>();
}

constexpr bool run_all() {
    // integral-backed
    test_all<1>();
    test_all<5>();
    test_all<7>();
    test_all<8>();
    test_all<12>();
    test_all<16>();
    test_all<20>();
    test_all<32>();
    test_all<50>();
    test_all<64>();

    // array-backed
    test_all<129>();
    test_all<150>();
    test_all<192>();
    test_all<200>();
    test_all<256>();
    test_all<300>();
    test_all<320>();
    test_all<384>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
