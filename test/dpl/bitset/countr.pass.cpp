// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

template <dpl::size_t W>
constexpr void test_countr_zero() {
    // all zeros -> W
    dpl::bitset<W> z;
    assert(dpl::countr_zero(z) == static_cast<int>(W));

    // all ones -> 0 (bit 0 is set)
    dpl::bitset<W> all;
    all.invert();
    assert(dpl::countr_zero(all) == 0);

    // single bit k set: k trailing zeros below it
    for (auto k = 0zu; k < W; ++k) {
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::countr_zero(b) == static_cast<int>(k));
    }
}

template <dpl::size_t W>
constexpr void test_countr_one() {
    // all zeros -> 0
    dpl::bitset<W> z;
    assert(dpl::countr_one(z) == 0);

    // all ones -> W
    dpl::bitset<W> all;
    all.invert();
    assert(dpl::countr_one(all) == static_cast<int>(W));

    // single bit k set: trailing ones only when k == 0
    for (auto k = 0zu; k < W; ++k) {
        dpl::bitset<W> b;
        b.set(k);
        int const expected = (k == 0zu) ? 1 : 0;
        assert(dpl::countr_one(b) == expected);
    }

    // run of r ones from bit 0 upward: bits 0..r-1 all set -> r trailing ones
    for (auto r = 1zu; r <= W; ++r) {
        dpl::bitset<W> b;
        for (auto i = 0zu; i < r; ++i) {
            b.set(i);
        }
        assert(dpl::countr_one(b) == static_cast<int>(r));
    }
}

// For array-backed widths, probe explicitly at word-boundary bit positions
// since the LSW-first walk is most likely to have an off-by-one there.
template <dpl::size_t W>
constexpr void test_countr_chunk_boundaries() {
    static_assert(W > 128);

    constexpr dpl::size_t boundaries[] = {0, 63, 64, 127, 128};

    // countr_zero: single bit at boundary -> that many trailing zeros
    for (auto k : boundaries) {
        if (k >= W)
            continue;
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::countr_zero(b) == static_cast<int>(k));
    }

    // countr_one: run of ones from bit 0 stopping exactly at each boundary
    for (auto k : boundaries) {
        if (k == 0zu || k >= W)
            continue;
        // set bits 0..k-1 (inclusive) -> run length = k
        dpl::bitset<W> b;
        for (auto i = 0zu; i < k; ++i) {
            b.set(i);
        }
        assert(dpl::countr_one(b) == static_cast<int>(k));
    }
}

constexpr bool run_all() {
    // integral-backed
    test_countr_zero<1>();
    test_countr_zero<5>();
    test_countr_zero<7>();
    test_countr_zero<8>();
    test_countr_zero<12>();
    test_countr_zero<16>();
    test_countr_zero<20>();
    test_countr_zero<32>();
    test_countr_zero<50>();
    test_countr_zero<64>();

    test_countr_one<1>();
    test_countr_one<5>();
    test_countr_one<7>();
    test_countr_one<8>();
    test_countr_one<12>();
    test_countr_one<16>();
    test_countr_one<20>();
    test_countr_one<32>();
    test_countr_one<50>();
    test_countr_one<64>();

    // array-backed
    test_countr_zero<129>();
    test_countr_zero<150>();
    test_countr_zero<192>();
    test_countr_zero<200>();
    test_countr_zero<256>();

    test_countr_one<129>();
    test_countr_one<150>();
    test_countr_one<192>();
    test_countr_one<200>();
    test_countr_one<256>();

    test_countr_chunk_boundaries<129>();
    test_countr_chunk_boundaries<192>();
    test_countr_chunk_boundaries<256>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
