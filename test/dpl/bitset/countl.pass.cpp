// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {
template <dpl::size_t W>
constexpr void test_countl_zero() {
    // all zeros -> W
    dpl::bitset<W> z;
    assert(dpl::countl_zero(z) == static_cast<int>(W));

    // all ones -> 0
    dpl::bitset<W> all;
    all.invert();
    assert(dpl::countl_zero(all) == 0);

    // single bit k set: W-1-k leading zeros
    for (auto k = 0zu; k < W; ++k) {
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::countl_zero(b) == static_cast<int>(W - 1zu - k));
    }
}

template <dpl::size_t W>
constexpr void test_countl_one() {
    // all zeros -> 0
    dpl::bitset<W> z;
    assert(dpl::countl_one(z) == 0);

    // all ones -> W
    dpl::bitset<W> all;
    all.invert();
    assert(dpl::countl_one(all) == static_cast<int>(W));

    // single bit k set: only if k == W-1 (the MSB) is there a leading one
    for (auto k = 0zu; k < W; ++k) {
        dpl::bitset<W> b;
        b.set(k);
        int const expected = (k == W - 1zu) ? 1 : 0;
        assert(dpl::countl_one(b) == expected);
    }

    // run of r ones from the MSB downward: bits W-1, W-2, ... W-r all set
    for (auto r = 1zu; r <= W; ++r) {
        dpl::bitset<W> b;
        for (auto i = 0zu; i < r; ++i) {
            b.set(W - 1zu - i);
        }
        assert(dpl::countl_one(b) == static_cast<int>(r));
    }
}

// For array-backed widths, also probe explicitly at word-boundary bit
// positions since the MSW-first walk is most likely to have an off-by-one
// there.
template <dpl::size_t W>
constexpr void test_countl_chunk_boundaries() {
    static_assert(W > 128);

    // countl_zero: set exactly one bit at a boundary position and check
    constexpr dpl::size_t boundaries[] = {0, 63, 64, 127, 128};
    for (auto k : boundaries) {
        if (k >= W)
            continue;
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::countl_zero(b) == static_cast<int>(W - 1zu - k));
    }

    // countl_one: run of ones stopping exactly at each boundary bit
    for (auto k : boundaries) {
        if (k >= W)
            continue;
        // set bits W-1 down to k (inclusive) -> run length = W - k
        dpl::bitset<W> b;
        for (auto i = k; i < W; ++i) {
            b.set(i);
        }
        assert(dpl::countl_one(b) == static_cast<int>(W - k));
    }
}

constexpr bool run_all() {
    // integral-backed
    test_countl_zero<1>();
    test_countl_zero<5>();
    test_countl_zero<7>();
    test_countl_zero<8>();
    test_countl_zero<12>();
    test_countl_zero<16>();
    test_countl_zero<20>();
    test_countl_zero<32>();
    test_countl_zero<50>();
    test_countl_zero<64>();

    test_countl_one<1>();
    test_countl_one<5>();
    test_countl_one<7>();
    test_countl_one<8>();
    test_countl_one<12>();
    test_countl_one<16>();
    test_countl_one<20>();
    test_countl_one<32>();
    test_countl_one<50>();
    test_countl_one<64>();

    // array-backed
    test_countl_zero<129>();
    test_countl_zero<150>();
    test_countl_zero<192>();
    test_countl_zero<200>();
    test_countl_zero<256>();

    test_countl_one<129>();
    test_countl_one<150>();
    test_countl_one<192>();
    test_countl_one<200>();
    test_countl_one<256>();

    test_countl_chunk_boundaries<129>();
    test_countl_chunk_boundaries<192>();
    test_countl_chunk_boundaries<256>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
