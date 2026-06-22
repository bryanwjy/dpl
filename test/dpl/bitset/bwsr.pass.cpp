// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

template <dpl::size_t W>
constexpr void test_shr() {
    dpl::bitset<W> z;

    // shift 0: identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert((b >> 0zu) == b);
    }

    // all-zeros stays all-zeros under any shift
    for (auto s = 0zu; s < W; ++s)
        assert((z >> s) == z);

    // no bits shift in from above: bits W-s..W-1 always zero after shift
    for (auto s = 1zu; s < W; ++s) {
        dpl::bitset<W> all;
        all.invert();
        auto r = all >> s;
        for (auto i = W - s; i < W; ++i)
            assert(r[i] == false);
    }

    // bit-position mapping: full O(W^2) at runtime, probe set at compile time
    auto check_shift = [&](dpl::size_t s, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        auto r = b >> s;
        if (i >= s) {
            assert(r[i - s] == true);
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
constexpr void test_shr_assign() {
    // >>= 0: identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        dpl::bitset<W> copy = b;
        b >>= 0zu;
        assert(b == copy);
    }

    // >>= returns *this
    {
        dpl::bitset<W> b;
        b.set(W - 1zu);
        auto& ref = (b >>= 1zu);
        assert(&ref == &b);
    }

    // >>= agrees with non-mutating >>
    auto check_assign = [](dpl::size_t s, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        dpl::bitset<W> copy = b;
        b >>= s;
        assert(b == (copy >> s));
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
constexpr void test_shr_large() {
    dpl::bitset<192> all;
    all.invert();

    assert((all >> 192zu) == dpl::bitset<192>{});
    assert((all >> 300zu) == dpl::bitset<192>{});

    // exact word-boundary shift: bit 64 >> 64 -> bit 0
    {
        dpl::bitset<192> b;
        b.set(64);
        auto r = b >> 64zu;
        assert(r[0] == true);
        assert(dpl::popcount(r) == 1);
    }

    // exact word-boundary shift: bit 128 >> 128 -> bit 0
    {
        dpl::bitset<192> b;
        b.set(128);
        auto r = b >> 128zu;
        assert(r[0] == true);
        assert(dpl::popcount(r) == 1);
    }

    // cross-word (bit_shift != 0, word_shift == 0): bit 64 >> 63 -> bit 1
    {
        dpl::bitset<192> b;
        b.set(64);
        auto r = b >> 63zu;
        assert(r[1] == true);
        assert(dpl::popcount(r) == 1);
    }

    // cross-word (bit_shift != 0, word_shift != 0): bit 128 >> 127 -> bit 1
    {
        dpl::bitset<192> b;
        b.set(128);
        auto r = b >> 127zu;
        assert(r[1] == true);
        assert(dpl::popcount(r) == 1);
    }
}

constexpr bool run_all() {
    test_shr<1>();
    test_shr<5>();
    test_shr<8>();
    test_shr<12>();
    test_shr<16>();
    test_shr<32>();
    test_shr<50>();
    test_shr<64>();

    test_shr_assign<1>();
    test_shr_assign<5>();
    test_shr_assign<8>();
    test_shr_assign<12>();
    test_shr_assign<16>();
    test_shr_assign<32>();
    test_shr_assign<50>();
    test_shr_assign<64>();

    test_shr<129>();
    test_shr<192>();
    test_shr<256>();

    test_shr_assign<129>();
    test_shr_assign<192>();
    test_shr_assign<256>();

    test_shr_large();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
