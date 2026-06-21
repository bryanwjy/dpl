// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

using dpl::bitset;

// Models "can {args...} copy-initialize a T", i.e. whether a *non-explicit*
// constructor exists for exactly this argument list.
template <class T>
void implicit_sink(T) {}

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

    // partial pack (count < W): explicit
    static_assert(dpl::is_explicitly_constructible_v<bitset<W>, bool, bool>);
    static_assert(dpl::is_constructible_v<bitset<W>, bool, bool>);

    // count == W + 1 exceeds the constructor's `sizeof...(Bs) <= W`
    // constraint and must not participate in overload resolution at all
    static_assert(
        too_many_bools_rejected<W>(dpl::make_index_sequence<W + 1>{}));
}

template <dpl::size_t W1, dpl::size_t W2>
void check_concat_implicit() {
    // the concatenating constructor has no explicit(...) qualifier at all
    static_assert(!dpl::is_explicitly_constructible_v<bitset<W1 + W2>,
        bitset<W1>, bitset<W2>>);
}

void instantiate_all() {
    check_bool_pack_explicitness<129>();
    check_bool_pack_explicitness<150>();
    check_bool_pack_explicitness<192>();
    check_bool_pack_explicitness<200>();
    check_bool_pack_explicitness<256>();
    check_bool_pack_explicitness<300>();
    check_bool_pack_explicitness<320>();
    check_bool_pack_explicitness<384>();

    check_concat_implicit<64, 150>();
    check_concat_implicit<192, 192>();
}

} // namespace
