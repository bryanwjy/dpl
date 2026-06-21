// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

namespace {

using dpl::bitset;
template <dpl::size_t W>
constexpr void test_default_construction() {
    bitset<W> b;
    static_assert(bitset<W>::width == W);
    assert(b.size() == W);
    assert(dpl::popcount(b) == 0zu);
    for (auto i = 0zu; i < W; ++i) {
        assert(!b[i]);
    }
}

template <dpl::size_t>
using always_bool = bool;

template <dpl::size_t W, dpl::size_t... Is>
constexpr bitset<W> make_alternating(dpl::index_sequence<Is...>) {
    // copy-list-init via the return statement: only well-formed if the
    // full-width bool-pack constructor is non-explicit, as documented.
    return {(Is % 2zu == 0)...};
}

template <dpl::size_t W>
constexpr void test_bool_pack_full() {
    auto b = make_alternating<W>(dpl::make_index_sequence<W>{});
    auto expected_popcount = 0zu;
    for (auto i = 0zu; i < W; ++i) {
        bool const expected = (i % 2zu == 0);
        assert(b[i] == expected);
        expected_popcount += expected ? 1zu : 0zu;
    }
    assert(dpl::popcount(b) == expected_popcount);
}

template <dpl::size_t W>
constexpr void test_bool_pack_partial() {
    if constexpr (W > 2) {
        bitset<W> b(true, false); // explicit direct-init: 2 < W
        assert(b[0] == true);
        assert(b[1] == false);
        for (auto i = 2zu; i < W; ++i) {
            assert(b[i] == false);
        }
        assert(dpl::popcount(b) == 1zu);
    }
}

template <dpl::size_t W1, dpl::size_t W2>
constexpr void test_concat2() {
    bitset<W1> a(true); // single bit set
    bitset<W2> z;       // all zero
    bitset<W1 + W2> c(a, z);
    static_assert(bitset<W1 + W2>::width == W1 + W2);
    assert(dpl::popcount(c) == 1zu);
    assert(c != bitset<W1 + W2>{});
}

template <dpl::size_t W1, dpl::size_t W2, dpl::size_t W3>
constexpr void test_concat3() {
    // exercises two levels of recursion in the concatenating constructor
    bitset<W1> a(true);
    bitset<W2> b(true);
    bitset<W3> c; // all zero
    bitset<W1 + W2 + W3> combined(a, b, c);
    static_assert(bitset<W1 + W2 + W3>::width == W1 + W2 + W3);
    assert(dpl::popcount(combined) == 2zu);
    assert(combined != bitset<W1 + W2 + W3>{});
}

constexpr bool test_all() {
    test_default_construction<129>();
    test_default_construction<150>();
    test_default_construction<192>();
    test_default_construction<200>();
    test_default_construction<256>();
    test_default_construction<300>();
    test_default_construction<320>();
    test_default_construction<384>();

    test_bool_pack_full<129>();
    test_bool_pack_full<150>();
    test_bool_pack_full<192>();
    test_bool_pack_full<200>();
    test_bool_pack_full<256>();
    test_bool_pack_full<300>();
    test_bool_pack_full<320>();
    test_bool_pack_full<384>();

    test_bool_pack_partial<129>();
    test_bool_pack_partial<150>();
    test_bool_pack_partial<192>();
    test_bool_pack_partial<200>();
    test_bool_pack_partial<256>();
    test_bool_pack_partial<300>();
    test_bool_pack_partial<320>();
    test_bool_pack_partial<384>();

    // small (integral-backed) + large (array-backed) piece
    test_concat2<64, 150>();
    // both pieces already array-backed
    test_concat2<192, 192>();

    // three SMALL integral-backed pieces summing to an array-backed result
    test_concat3<64, 64, 64>();
    // mixed small + large + large
    test_concat3<50, 150, 200>();

    return true;
}

} // namespace

int main() {
    static_assert(test_all());
    assert(test_all());
    return 0;
}
