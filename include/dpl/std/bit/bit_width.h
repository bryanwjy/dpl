// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/char_bit.h"
#include "dpl/std/bit/countl.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONT, NODISCARD)
constexpr int bit_width(T val) noexcept {
    constexpr auto shift = __DPL char_bit_v * sizeof(T) - 1;
    return val == 0 ? 0 : (shift - __DPL countl_zero(val)) + 1;
}

DPL_DEFAULT_NAMESPACE_END
