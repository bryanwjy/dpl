// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

using dpl::bitset;

template <dpl::size_t W>
constexpr void test_default_construction() {
    bitset<W> b;
    static_assert(bitset<W>::width == W);
    assert(b.size() == W);
    assert(dpl::to_underlying(b) == 0);
}

template <dpl::size_t W>
constexpr void test_integral_construction() {
    using U = typename bitset<W>::underlying_type;
    constexpr auto bits_of_U = sizeof(U) * dpl::char_bit_v;

    // implicit: T == underlying_type (unsigned, same size)
    constexpr U all_ones = static_cast<U>(~U{0});
    bitset<W> from_underlying = all_ones;
    constexpr U expected_mask = (W == bits_of_U)
        ? all_ones
        : static_cast<U>((static_cast<U>(1) << W) - static_cast<U>(1));
    assert(dpl::to_underlying(from_underlying) == expected_mask);

    using S = dpl::make_signed_t<U>;
    if constexpr (W == bits_of_U) {
        // exact width: sizeof(S)*char_bit_v == W, so the "no masking" branch
        // is taken -- val's bit pattern is reinterpreted as-is via static_cast
        bitset<W> from_signed(static_cast<S>(-1)); // explicit: S is signed
        assert(dpl::to_underlying(from_signed) == all_ones);
    } else {
        // non-exact width: no integral type has exactly W bits, so the
        // masking branch always applies regardless of which T is passed
        bitset<W> from_signed(static_cast<S>(1)); // explicit: S is signed
        assert(dpl::to_underlying(from_signed) == static_cast<U>(1));
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
    for (auto i = 0zu; i < W; ++i) {
        assert(b[i] == (i % 2zu == 0));
    }
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
    }
}

template <dpl::size_t W1, dpl::size_t W2>
constexpr void test_concat2() {
    bitset<W1> a(true); // single bit set
    bitset<W2> z;       // all zero
    bitset<W1 + W2> c(a, z);
    static_assert(bitset<W1 + W2>::width == W1 + W2);
    assert(dpl::popcount(c) == 1);
    assert(c != bitset<W1 + W2>{});
}

template <dpl::size_t W1, dpl::size_t W2, dpl::size_t W3>
constexpr void test_concat3() {
    // attempt two levels of recursion in the concatenating constructor
    bitset<W1> a(true);
    bitset<W2> b(true);
    bitset<W3> c; // all zero
    bitset<W1 + W2 + W3> combined(a, b, c);
    static_assert(bitset<W1 + W2 + W3>::width == W1 + W2 + W3);
    assert(dpl::popcount(combined) == 2);
    assert(combined != bitset<W1 + W2 + W3>{});
}

constexpr bool test_all() {
    test_default_construction<1>();
    test_default_construction<5>();
    test_default_construction<7>();
    test_default_construction<8>();
    test_default_construction<12>();
    test_default_construction<16>();
    test_default_construction<20>();
    test_default_construction<32>();
    test_default_construction<50>();
    test_default_construction<64>();

    test_integral_construction<1>();
    test_integral_construction<5>();
    test_integral_construction<7>();
    test_integral_construction<8>();
    test_integral_construction<12>();
    test_integral_construction<16>();
    test_integral_construction<20>();
    test_integral_construction<32>();
    test_integral_construction<50>();
    test_integral_construction<64>();

    test_bool_pack_full<1>();
    test_bool_pack_full<5>();
    test_bool_pack_full<7>();
    test_bool_pack_full<8>();
    test_bool_pack_full<12>();
    test_bool_pack_full<16>();
    test_bool_pack_full<20>();
    test_bool_pack_full<32>();
    test_bool_pack_full<50>();
    test_bool_pack_full<64>();

    test_bool_pack_partial<1>();
    test_bool_pack_partial<5>();
    test_bool_pack_partial<7>();
    test_bool_pack_partial<8>();
    test_bool_pack_partial<12>();
    test_bool_pack_partial<16>();
    test_bool_pack_partial<20>();
    test_bool_pack_partial<32>();
    test_bool_pack_partial<50>();
    test_bool_pack_partial<64>();

    test_concat2<3, 5>();
    test_concat2<8, 8>();
    test_concat2<12, 20>();
    test_concat2<50, 14>();

    test_concat3<2, 3, 3>();
    test_concat3<10, 10, 12>();

    return true;
}

} // namespace

int main() {
    static_assert(test_all());
    assert(test_all());
    return 0;
}
