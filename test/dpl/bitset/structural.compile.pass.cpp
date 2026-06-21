// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

using dpl::bitset;

template <dpl::bitset<8> V>
struct constant8 {
    static constexpr auto value = V;
};

template <dpl::bitset<129> V>
struct constant129 {
    static constexpr auto value = V;
};

template <dpl::size_t W, dpl::bitset<W> V>
struct constant_generic {
    static constexpr auto value = V;
};

constexpr constant8<dpl::bitset<8>(true, false, true)> explicit_inst8;
static_assert(explicit_inst8.value[0] == true);
static_assert(explicit_inst8.value[1] == false);
static_assert(explicit_inst8.value[2] == true);

constexpr constant129<dpl::bitset<129>(true)> explicit_inst129;
static_assert(explicit_inst129.value[0] == true);
static_assert(explicit_inst129.value[1] == false);

constexpr constant_generic<8, dpl::bitset<8>(true)> explicit_inst_generic8;
constexpr constant_generic<129, dpl::bitset<129>(true)>
    explicit_inst_generic129;
static_assert(explicit_inst_generic8.value[0] == true);
static_assert(explicit_inst_generic129.value[0] == true);

// explicit NTTP on a function template
template <dpl::bitset<8> V>
constexpr dpl::bitset<8> explicit_identity() {
    return V;
}
static_assert(explicit_identity<dpl::bitset<8>(true, false)>()[0] == true);
static_assert(explicit_identity<dpl::bitset<8>(true, false)>()[1] == false);

// equivalent NTTP values (same bits, different construction path) must
// collapse to the same instantiation
static_assert(dpl::is_same_v<constant8<dpl::bitset<8>(true, false, true)>,
    constant8<dpl::bitset<8>(static_cast<dpl::uint8>(0b101))>>);

// distinct NTTP values must produce distinct instantiations
static_assert(!dpl::is_same_v<constant8<dpl::bitset<8>(true, false, true)>,
    constant8<dpl::bitset<8>(false, true, false)>>);

// ---- auto NTTP form --------------------------------------------------------

template <auto V>
struct auto_constant {
    using value_type = decltype(V);
    static constexpr auto value = V;
};

constexpr auto_constant<dpl::bitset<8>(true, false, true)> auto_inst8;
static_assert(dpl::is_same_v<decltype(auto_inst8)::value_type, dpl::bitset<8>>);
static_assert(auto_inst8.value[0] == true);
static_assert(auto_inst8.value[1] == false);
static_assert(auto_inst8.value[2] == true);

constexpr auto_constant<dpl::bitset<129>(true)> auto_inst129;
static_assert(
    dpl::is_same_v<decltype(auto_inst129)::value_type, dpl::bitset<129>>);
static_assert(auto_inst129.value[0] == true);
static_assert(auto_inst129.value[1] == false);

// auto NTTP on a function template
template <auto V>
constexpr auto auto_identity() {
    return V;
}
static_assert(auto_identity<dpl::bitset<8>(true, false)>()[0] == true);
static_assert(auto_identity<dpl::bitset<8>(true, false)>()[1] == false);

// equivalent NTTP values collapse to the same instantiation, including
// across different construction paths (bool-pack vs. integral vs.
// concatenation), for both storage kinds
static_assert(dpl::is_same_v<auto_constant<dpl::bitset<8>(true, false, true)>,
    auto_constant<dpl::bitset<8>(static_cast<dpl::uint8>(0b101))>>);

static_assert(dpl::is_same_v<auto_constant<dpl::bitset<129>(true)>,
    auto_constant<dpl::bitset<129>(
        dpl::bitset<1>(true), dpl::bitset<64>{}, dpl::bitset<64>{})>>);

// distinct NTTP values must produce distinct instantiations
static_assert(!dpl::is_same_v<auto_constant<dpl::bitset<8>(true, false, true)>,
    auto_constant<dpl::bitset<8>(false, true, false)>>);

static_assert(!dpl::is_same_v<auto_constant<dpl::bitset<129>(true)>,
    auto_constant<dpl::bitset<129>{}>>);

} // namespace
