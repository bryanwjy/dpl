// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {
using namespace dpl::bit_literals;

// ---- binary ----------------------------------------------------------------

static_assert(dpl::to_underlying("0b0"_bits) == 0);
static_assert(dpl::to_underlying("0b1"_bits) == 1);
static_assert(dpl::to_underlying("0b1010"_bits) == 0b1010); // 10
static_assert(dpl::to_underlying("0b11111111"_bits) == 0xFF);
static_assert(dpl::to_underlying("0B1010"_bits) == 0b1010); // uppercase prefix

// ---- hex ---------------------------------------------------------------

static_assert(dpl::to_underlying("0x0"_bits) == 0x0);
static_assert(dpl::to_underlying("0x1"_bits) == 0x1);
static_assert(dpl::to_underlying("0x1A"_bits) == 0x1A); // 26, mixed-case digit
static_assert(dpl::to_underlying("0xff"_bits) == 0xFF); // lowercase digits
static_assert(dpl::to_underlying("0xFF"_bits) == 0xFF); // uppercase digits
static_assert(dpl::to_underlying("0X1A"_bits) == 0x1A); // uppercase prefix

// ---- octal -------------------------------------------------------------

static_assert(dpl::to_underlying("00"_bits) == 0);
static_assert(dpl::to_underlying("01"_bits) == 1);
static_assert(dpl::to_underlying("017"_bits) == 017);   // 15
static_assert(dpl::to_underlying("0755"_bits) == 0755); // 493
static_assert(dpl::to_underlying("0777"_bits) == 0777); // 511

// result is a genuine bitset, usable wherever a constexpr bitset is

constexpr auto literal_value = "0b1010"_bits;
static_assert(dpl::bitset_type<decltype(literal_value)>);
static_assert(literal_value[0] == false);
static_assert(literal_value[1] == true);
static_assert(literal_value[2] == false);
static_assert(literal_value[3] == true);

// usable directly as an NTTP (auto-deduced), since operator""_bits is
// consteval and the result is a structural type
template <auto V>
struct constant {
    static constexpr auto value = V;
};

constexpr constant<"0b1010"_bits> inst;
static_assert(inst.value[1] == true);
static_assert(inst.value[3] == true);

} // namespace
