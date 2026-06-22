// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

template <dpl::size_t W>
constexpr void test_shl() {
    dpl::bitset<W> z;

    // shift 0: identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b << 0zu) == b);
    }

    // all-zeros stays all-zeros under any shift
    for (auto s = 0zu; s < W; ++s)
        assert((z << s) == z);

    // no bits shift in from below: bits 0..s-1 are always zero after shift
    for (auto s = 1zu; s < W; ++s) {
        dpl::bitset<W> all;
        all.invert();
        auto r = all << s;
        for (auto i = 0zu; i < s; ++i)
            assert(r[i] == false);
    }

    // bit-position mapping: full O(W^2) at runtime, probe set at compile time
    auto check_shift = [&](dpl::size_t s, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        auto r = b << s;
        if (i + s < W) {
            assert(r[i + s] == true);
            assert(dpl::popcount(r) == 1);
        } else {
            assert(r == z);
        }
    };

    if consteval {
        constexpr dpl::size_t probes[] = {0, 1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto s : probes) {
            if (s >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_shift(s, i);
            }
        }
    } else {
        for (auto s = 1zu; s < W; ++s)
            for (auto i = 0zu; i < W; ++i)
                check_shift(s, i);
    }
}

template <dpl::size_t W>
constexpr void test_shl_assign() {
    // <<= 0: identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        dpl::bitset<W> copy = b;
        b <<= 0zu;
        assert(b == copy);
    }

    // <<= returns *this
    {
        dpl::bitset<W> b;
        b.set(0);
        auto& ref = (b <<= 1zu);
        assert(&ref == &b);
    }

    // <<= agrees with non-mutating <<
    auto check_assign = [](dpl::size_t s, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        dpl::bitset<W> copy = b;
        b <<= s;
        assert(b == (copy << s));
    };

    if consteval {
        constexpr dpl::size_t probes[] = {1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto s : probes) {
            if (s >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_assign(s, i);
            }
        }
    } else {
        for (auto s = 1zu; s < W; ++s)
            for (auto i = 0zu; i < W; ++i)
                check_assign(s, i);
    }
}

// large_bitset only: shift >= W clears; word-boundary and cross-word shifts
constexpr void test_shl_large() {
    dpl::bitset<192> all;
    all.invert();

    assert((all << 192zu) == dpl::bitset<192>{});
    assert((all << 300zu) == dpl::bitset<192>{});

    // exact word-boundary shift: bit 0 << 64 -> bit 64
    {
        dpl::bitset<192> b;
        b.set(0);
        auto r = b << 64zu;
        assert(r[64] == true);
        assert(dpl::popcount(r) == 1);
    }

    // exact word-boundary shift: bit 0 << 128 -> bit 128
    {
        dpl::bitset<192> b;
        b.set(0);
        auto r = b << 128zu;
        assert(r[128] == true);
        assert(dpl::popcount(r) == 1);
    }

    // cross-word (bit_shift != 0, word_shift == 0): bit 1 << 63 -> bit 64
    {
        dpl::bitset<192> b;
        b.set(1);
        auto r = b << 63zu;
        assert(r[64] == true);
        assert(dpl::popcount(r) == 1);
    }

    // cross-word (bit_shift != 0, word_shift != 0): bit 1 << 127 -> bit 128
    {
        dpl::bitset<192> b;
        b.set(1);
        auto r = b << 127zu;
        assert(r[128] == true);
        assert(dpl::popcount(r) == 1);
    }
}

constexpr bool run_all() {
    test_shl<1>();
    test_shl<5>();
    test_shl<8>();
    test_shl<12>();
    test_shl<16>();
    test_shl<32>();
    test_shl<50>();
    test_shl<64>();

    test_shl_assign<1>();
    test_shl_assign<5>();
    test_shl_assign<8>();
    test_shl_assign<12>();
    test_shl_assign<16>();
    test_shl_assign<32>();
    test_shl_assign<50>();
    test_shl_assign<64>();

    test_shl<129>();
    test_shl<192>();
    test_shl<256>();

    test_shl_assign<129>();
    test_shl_assign<192>();
    test_shl_assign<256>();

    test_shl_large();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
