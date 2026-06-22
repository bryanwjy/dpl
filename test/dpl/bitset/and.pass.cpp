// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

template <dpl::size_t W>
constexpr void test_and() {
    dpl::bitset<W> z;
    dpl::bitset<W> all;
    all.invert();

    // zero annihilator: b & 0 == 0
    assert((all & z) == z);
    assert((z & all) == z);

    // all-ones identity: b & ~0 == b
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b & all) == b);
        assert((all & b) == b);
    }

    // idempotent: b & b == b
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b & b) == b);
    }

    // commutativity and disjoint-bit check: full O(W^2) at runtime,
    // fixed probe points at compile time to stay within constexpr step limits
    auto check_pair = [&](dpl::size_t i, dpl::size_t j) {
        dpl::bitset<W> a, b;
        a.set(i);
        b.set(j);
        assert((a & b) == (b & a));
        auto r = a & b;
        if (i == j) {
            assert(r == a);
            assert(dpl::popcount(r) == 1);
        } else {
            assert(r == z);
            assert(dpl::popcount(r) == 0);
        }
    };

    if consteval {
        // Too many compile time steps
        constexpr dpl::size_t probes[] = {0, 1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto i : probes) {
            if (i >= W)
                continue;
            for (auto j : probes) {
                if (j >= W)
                    continue;
                check_pair(i, j);
            }
        }
    } else {
        for (auto i = 0zu; i < W; ++i)
            for (auto j = 0zu; j < W; ++j)
                check_pair(i, j);
    }
}

template <dpl::size_t W>
constexpr void test_and_assign() {
    dpl::bitset<W> z;
    dpl::bitset<W> all;
    all.invert();

    // &= with all-ones is identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        dpl::bitset<W> copy = b;
        b &= all;
        assert(b == copy);
    }

    // &= with zero clears
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        b &= z;
        assert(b == z);
    }

    // &= returns *this (fluent chaining)
    {
        dpl::bitset<W> a, b;
        a.set(0);
        b.set(0);
        auto& ref = (a &= b);
        assert(&ref == &a);
    }
}

// cross-width & and &= (large_bitset only)
constexpr void test_and_cross_width() {
    // large &= large: overlapping words ANDed, excess zeroed
    {
        dpl::bitset<192> big;
        big.invert();
        dpl::bitset<128> small;
        small.invert();

        auto r = big & small;
        static_assert(dpl::is_same_v<decltype(r), dpl::bitset<192>>);
        for (auto i = 0zu; i < 128zu; ++i)
            assert(r[i] == true);
        for (auto i = 128zu; i < 192zu; ++i)
            assert(r[i] == false);

        dpl::bitset<192> b;
        b.invert();
        b &= small;
        for (auto i = 0zu; i < 128zu; ++i)
            assert(b[i] == true);
        for (auto i = 128zu; i < 192zu; ++i)
            assert(b[i] == false);
    }

    // large &= integral-backed: storage_[0] ANDed, storage_[1..N-1] must be
    // zeroed -- encodes correct behavior, will fail until the missing zeroing
    // loop in the OW<=chunk_size branch is fixed
    {
        dpl::bitset<192> b;
        b.invert();
        dpl::bitset<64> small;
        small.invert();

        b &= small;
        for (auto i = 0zu; i < 64zu; ++i)
            assert(b[i] == true);
        for (auto i = 64zu; i < 192zu; ++i)
            assert(b[i] == false);
    }
}

constexpr bool run_all() {
    test_and<1>();
    test_and<5>();
    test_and<8>();
    test_and<12>();
    test_and<16>();
    test_and<32>();
    test_and<50>();
    test_and<64>();

    test_and_assign<1>();
    test_and_assign<5>();
    test_and_assign<8>();
    test_and_assign<12>();
    test_and_assign<16>();
    test_and_assign<32>();
    test_and_assign<50>();
    test_and_assign<64>();

    test_and<129>();
    test_and<192>();
    test_and<256>();

    test_and_assign<129>();
    test_and_assign<192>();
    test_and_assign<256>();

    test_and_cross_width();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
