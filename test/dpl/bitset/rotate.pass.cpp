// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

template <dpl::size_t W>
constexpr void test_rotl() {
    dpl::bitset<W> z;

    // count == 0: identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotl(b, 0) == b);
    }

    // count == W: full rotation is identity (via count %= W)
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotl(b, static_cast<int>(W)) == b);
    }

    // count > W: wraps correctly
    {
        dpl::bitset<W> b;
        b.set(0);
        assert(dpl::rotl(b, static_cast<int>(W) + 1) == dpl::rotl(b, 1));
    }

    // all-zeros: any rotation stays all-zeros
    for (auto k = 0; k <= static_cast<int>(W); ++k)
        assert(dpl::rotl(z, k) == z);

    // all-ones: any rotation stays all-ones
    {
        dpl::bitset<W> all;
        all.invert();
        for (auto k = 0; k <= static_cast<int>(W); ++k)
            assert(dpl::rotl(all, k) == all);
    }

    // bit-position mapping: bit i rotl by k lands at (i+k)%W
    // full O(W^2) at runtime, probe set at compile time
    auto check_rotl = [&](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        auto r = dpl::rotl(b, static_cast<int>(k));
        auto expected = (i + k) % W;
        assert(r[expected] == true);
        assert(dpl::popcount(r) == 1);
    };

    if consteval {
        constexpr dpl::size_t probes[] = {0, 1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_rotl(k, i);
            }
        }
    } else {
        for (auto k = 1zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check_rotl(k, i);
    }

    // negative count: rotl(b, -k) == rotr(b, k)
    auto check_rotl_neg = [](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotl(b, -static_cast<int>(k)) ==
            dpl::rotr(b, static_cast<int>(k)));
    };

    if consteval {
        constexpr dpl::size_t probes[] = {1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_rotl_neg(k, i);
            }
        }
    } else {
        for (auto k = 1zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check_rotl_neg(k, i);
    }

    // round-trip: rotr(rotl(b, k), k) == b
    auto check_roundtrip = [](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotr(dpl::rotl(b, static_cast<int>(k)),
                   static_cast<int>(k)) == b);
    };

    if consteval {
        constexpr dpl::size_t probes[] = {1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_roundtrip(k, i);
            }
        }
    } else {
        for (auto k = 1zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check_roundtrip(k, i);
    }
}

template <dpl::size_t W>
constexpr void test_rotr() {
    dpl::bitset<W> z;

    // count == 0: identity
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotr(b, 0) == b);
    }

    // count == W: full rotation is identity (via count %= W)
    for (auto i = 0zu; i < W; ++i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotr(b, static_cast<int>(W)) == b);
    }

    // count > W: wraps correctly
    {
        dpl::bitset<W> b;
        b.set(0);
        assert(dpl::rotr(b, static_cast<int>(W) + 1) == dpl::rotr(b, 1));
    }

    // all-zeros: any rotation stays all-zeros
    for (auto k = 0; k <= static_cast<int>(W); ++k)
        assert(dpl::rotr(z, k) == z);

    // all-ones: any rotation stays all-ones
    {
        dpl::bitset<W> all;
        all.invert();
        for (auto k = 0; k <= static_cast<int>(W); ++k)
            assert(dpl::rotr(all, k) == all);
    }

    // bit-position mapping: bit i rotr by k lands at (i+W-k)%W
    // full O(W^2) at runtime, probe set at compile time
    auto check_rotr = [&](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        auto r = dpl::rotr(b, static_cast<int>(k));
        auto expected = (i + W - k) % W;
        assert(r[expected] == true);
        assert(dpl::popcount(r) == 1);
    };

    if consteval {
        constexpr dpl::size_t probes[] = {0, 1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_rotr(k, i);
            }
        }
    } else {
        for (auto k = 1zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check_rotr(k, i);
    }

    // negative count: rotr(b, -k) == rotl(b, k)
    auto check_rotr_neg = [](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotr(b, -static_cast<int>(k)) ==
            dpl::rotl(b, static_cast<int>(k)));
    };

    if consteval {
        constexpr dpl::size_t probes[] = {1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_rotr_neg(k, i);
            }
        }
    } else {
        for (auto k = 1zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check_rotr_neg(k, i);
    }

    // round-trip: rotl(rotr(b, k), k) == b
    auto check_roundtrip = [](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotl(dpl::rotr(b, static_cast<int>(k)),
                   static_cast<int>(k)) == b);
    };

    if consteval {
        constexpr dpl::size_t probes[] = {1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check_roundtrip(k, i);
            }
        }
    } else {
        for (auto k = 1zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check_roundtrip(k, i);
    }
}

// rotl by k == rotr by W-k (the two are duals)
template <dpl::size_t W>
constexpr void test_duality() {
    auto check = [](dpl::size_t k, dpl::size_t i) {
        dpl::bitset<W> b;
        b.set(i);
        assert(dpl::rotl(b, static_cast<int>(k)) ==
            dpl::rotr(b, static_cast<int>(W - k)));
    };

    if consteval {
        constexpr dpl::size_t probes[] = {0, 1, W / 4, W / 2, W - 2zu, W - 1zu};
        for (auto k : probes) {
            if (k >= W)
                continue;
            for (auto i : probes) {
                if (i >= W)
                    continue;
                check(k, i);
            }
        }
    } else {
        for (auto k = 0zu; k < W; ++k)
            for (auto i = 0zu; i < W; ++i)
                check(k, i);
    }
}

constexpr bool run_all() {
    // integral-backed
    test_rotl<1>();
    test_rotl<5>();
    test_rotl<8>();
    test_rotl<12>();
    test_rotl<16>();
    test_rotl<32>();
    test_rotl<50>();
    test_rotl<64>();

    test_rotr<1>();
    test_rotr<5>();
    test_rotr<8>();
    test_rotr<12>();
    test_rotr<16>();
    test_rotr<32>();
    test_rotr<50>();
    test_rotr<64>();

    test_duality<1>();
    test_duality<5>();
    test_duality<8>();
    test_duality<16>();
    test_duality<32>();
    test_duality<64>();

    // array-backed
    test_rotl<129>();
    test_rotl<150>();
    test_rotl<192>();

    test_rotr<129>();
    test_rotr<150>();
    test_rotr<192>();

    test_duality<129>();
    test_duality<192>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
