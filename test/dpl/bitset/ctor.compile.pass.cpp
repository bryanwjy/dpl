// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

using dpl::bitset;

// Models "can {args...} copy-initialize a T", i.e. whether a *non-explicit*
// constructor exists for exactly this argument list. is_convertible_v only
// covers single-argument conversions, so multi-argument constructors
// (the bool pack, the concatenating constructor) need this instead.
template <typename T>
void implicit_sink(T) {}

template <dpl::size_t W>
void check_integral_explicitness() {
    using U = typename bitset<W>::underlying_type;
    using S = dpl::make_signed_t<U>;
    using DiffSizeU = dpl::conditional_t<dpl::is_same_v<U, dpl::uint8>,
        dpl::uint16, dpl::uint8>;

    // T == underlying_type (unsigned, same size): implicit
    static_assert(dpl::is_convertible_v<U, bitset<W>>);
    // signed, even if same size as underlying_type: explicit
    static_assert(!dpl::is_convertible_v<S, bitset<W>>);
    static_assert(dpl::is_constructible_v<bitset<W>, S>);
    // unsigned but a different size from underlying_type: explicit
    static_assert(!dpl::is_convertible_v<DiffSizeU, bitset<W>>);
    static_assert(dpl::is_constructible_v<bitset<W>, DiffSizeU>);
}

template <dpl::size_t>
using always_bool = bool;

template <dpl::size_t W, dpl::size_t... Is>
constexpr bool full_pack_is_implicit(dpl::index_sequence<Is...>) {
    return !dpl::is_explicitly_constructible_v<bitset<W>, always_bool<Is>...>;
}

template <dpl::size_t W, dpl::size_t... Is>
constexpr bool too_many_bools_rejected(dpl::index_sequence<Is...>) {
    return !dpl::is_constructible_v<bitset<W>, always_bool<Is>...>;
}

template <dpl::size_t W>
void check_bool_pack_explicitness() {
    // full width (count == W): implicit
    static_assert(full_pack_is_implicit<W>(dpl::make_index_sequence<W>{}));

    if constexpr (W > 2) {
        // partial pack (count < W): explicit
        static_assert(
            dpl::is_explicitly_constructible_v<bitset<W>, bool, bool>);
        static_assert(dpl::is_constructible_v<bitset<W>, bool, bool>);
    }

    // count == W + 1 exceeds the constructor's `sizeof...(Bs) <= W`
    // constraint and must not participate in overload resolution at all
    static_assert(
        too_many_bools_rejected<W>(dpl::make_index_sequence<W + 1>{}));
}

template <dpl::size_t W1, dpl::size_t W2>
void check_concat_implicit() {
    // the concatenating constructor is implicit
    static_assert(!dpl::is_explicitly_constructible_v<bitset<W1 + W2>,
        bitset<W1>, bitset<W2>>);
}

void check_simple_ctad() {
    // template <integral T> bitset(T val) -> bitset<sizeof(T) * char_bit_v>;
    dpl::bitset b(0u);
    static_assert(
        dpl::is_same_v<decltype(b), dpl::bitset<sizeof(0u) * dpl::char_bit_v>>);
}

void instantiate_all() {
    check_integral_explicitness<1>();
    check_integral_explicitness<5>();
    check_integral_explicitness<7>();
    check_integral_explicitness<8>();
    check_integral_explicitness<12>();
    check_integral_explicitness<16>();
    check_integral_explicitness<20>();
    check_integral_explicitness<32>();
    check_integral_explicitness<50>();
    check_integral_explicitness<64>();

    check_bool_pack_explicitness<1>();
    check_bool_pack_explicitness<5>();
    check_bool_pack_explicitness<7>();
    check_bool_pack_explicitness<8>();
    check_bool_pack_explicitness<12>();
    check_bool_pack_explicitness<16>();
    check_bool_pack_explicitness<20>();
    check_bool_pack_explicitness<32>();
    check_bool_pack_explicitness<50>();
    check_bool_pack_explicitness<64>();

    check_concat_implicit<3, 5>();
    check_concat_implicit<8, 8>();
    check_concat_implicit<12, 20>();
    check_concat_implicit<50, 14>();

    check_simple_ctad();
}

} // namespace
