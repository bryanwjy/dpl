// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/char_bit.h"
#include "dpl/std/bit/countl.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace details::bit {
template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, CONST)
constexpr T bit_log2(T x) noexcept {
    return char_bit_v * sizeof(T) - 1 - __DPL countl_zero(x);
}
} // namespace details::bit

template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONT, NODISCARD)
constexpr int bit_width(T val) noexcept {
    return val == 0 ? 0 : details::bit::bit_log2(val) + 1;
}

DPL_DEFAULT_NAMESPACE_END
