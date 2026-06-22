// Copyright 2025-2026 Bryan Wong

#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

template <dpl::size_t W>
constexpr void test_or() {
    dpl::bitset<W> z;
    dpl::bitset<W> all;
    all.invert();

    // zero identity: b | 0 == b
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b | z) == b);
        assert((z | b) == b);
    }

    // all-ones absorber: b | ~0 == ~0
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b | all) == all);
        assert((all | b) == all);
    }

    // idempotent: b | b == b
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b | b) == b);
    }

    // commutativity and union-bit check: full O(W^2) at runtime,
    // fixed probe points at compile time
    auto check_pair = [&](dpl::size_t i, dpl::size_t j) {
        dpl::bitset<W> a, b;
        a.set(i);
        b.set(j);
        assert((a | b) == (b | a));
        auto r = a | b;
        assert(r[i] == true);
        assert(r[j] == true);
        assert(dpl::popcount(r) == (i == j ? 1 : 2));
    };

    if consteval {
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
constexpr void test_or_assign() {
    dpl::bitset<W> z;
    dpl::bitset<W> all;
    all.invert();

    // |= with zero is identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        dpl::bitset<W> copy = b;
        b |= z;
        assert(b == copy);
    }

    // |= with all-ones saturates
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        b |= all;
        assert(b == all);
    }

    // |= returns *this (fluent chaining)
    {
        dpl::bitset<W> a, b;
        a.set(0);
        b.set(W - 1zu);
        auto& ref = (a |= b);
        assert(&ref == &a);
        assert(a[0] == true);
        assert(a[W - 1zu] == true);
    }
}

// cross-width | and |= (large_bitset only)
constexpr void test_or_cross_width() {
    // big all-ones | small all-ones == big (small is a subset)
    {
        dpl::bitset<192> big;
        big.invert();
        dpl::bitset<128> small;
        small.invert();

        auto r = big | small;
        static_assert(dpl::is_same_v<decltype(r), dpl::bitset<192>>);
        assert(r == big);
    }

    // big all-zero | small all-ones: lower 128 bits set, upper 64 clear
    {
        dpl::bitset<192> zero_big;
        dpl::bitset<128> small;
        small.invert();

        auto r = zero_big | small;
        static_assert(dpl::is_same_v<decltype(r), dpl::bitset<192>>);
        for (auto i = 0zu; i < 128zu; ++i)
            assert(r[i] == true);
        for (auto i = 128zu; i < 192zu; ++i)
            assert(r[i] == false);
    }

    // |= leaves upper words of *this untouched
    {
        dpl::bitset<192> b;
        b.set(191);
        dpl::bitset<128> small;
        small.invert();

        b |= small;
        for (auto i = 0zu; i < 128zu; ++i)
            assert(b[i] == true);
        assert(b[191] == true);
        assert(dpl::popcount(b) == 129);
    }
}

constexpr bool run_all() {
    test_or<1>();
    test_or<5>();
    test_or<8>();
    test_or<12>();
    test_or<16>();
    test_or<32>();
    test_or<50>();
    test_or<64>();

    test_or_assign<1>();
    test_or_assign<5>();
    test_or_assign<8>();
    test_or_assign<12>();
    test_or_assign<16>();
    test_or_assign<32>();
    test_or_assign<50>();
    test_or_assign<64>();

    test_or<129>();
    test_or<192>();
    test_or<256>();

    test_or_assign<129>();
    test_or_assign<192>();
    test_or_assign<256>();

    test_or_cross_width();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}