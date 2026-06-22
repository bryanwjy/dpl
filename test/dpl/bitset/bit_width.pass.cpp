// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {
static_assert(dpl::bit_width(dpl::bitset<1>()) == 0);
template <dpl::size_t W>
constexpr void test_bit_width() {
    // all zeros -> 0
    dpl::bitset<W> z;
    assert(dpl::bit_width(z) == 0);

    // each single bit k set -> k + 1
    for (auto k = 0zu; k < W; ++k) {
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::bit_width(b) == static_cast<int>(k + 1zu));
    }

    // all bits set -> W
    dpl::bitset<W> all;
    all.invert();
    assert(dpl::bit_width(all) == static_cast<int>(W));

    // highest bit + lower bits set: highest bit dominates, result still W
    if constexpr (W >= 2) {
        dpl::bitset<W> top;
        top.set(W - 1zu);
        top.set(0);
        assert(dpl::bit_width(top) == static_cast<int>(W));
    }

    // second-highest bit set, highest clear: result is W - 1
    if constexpr (W >= 2) {
        dpl::bitset<W> b;
        b.set(W - 2zu);
        assert(dpl::bit_width(b) == static_cast<int>(W - 1zu));
    }
}

// For array-backed widths, explicitly probe chunk-boundary bit positions.
template <dpl::size_t W>
constexpr void test_bit_width_chunk_boundaries() {
    static_assert(W > 128);
    constexpr dpl::size_t boundaries[] = {63, 64, 127, 128};
    for (auto k : boundaries) {
        if (k >= W)
            continue;
        dpl::bitset<W> b;
        b.set(k);
        assert(dpl::bit_width(b) == static_cast<int>(k + 1zu));
    }
}

constexpr bool run_all() {
    // integral-backed
    test_bit_width<1>();
    test_bit_width<5>();
    test_bit_width<7>();
    test_bit_width<8>();
    test_bit_width<12>();
    test_bit_width<16>();
    test_bit_width<20>();
    test_bit_width<32>();
    test_bit_width<50>();
    test_bit_width<64>();

    // array-backed
    test_bit_width<129>();
    test_bit_width<150>();
    test_bit_width<192>();
    test_bit_width<200>();
    test_bit_width<256>();
    test_bit_width<300>();
    test_bit_width<320>();
    test_bit_width<384>();

    // chunk-boundary probing (array-backed only)
    test_bit_width_chunk_boundaries<129>();
    test_bit_width_chunk_boundaries<192>();
    test_bit_width_chunk_boundaries<256>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
