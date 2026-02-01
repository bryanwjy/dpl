// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/bit_width.h"
#include "dpl/std/bit/char_bit.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace details::bit {
template <typename T>
requires requires(T val) {
    { +val } -> same_as<T>;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T bit_ceil(T val) noexcept {
    return static_cast<T>(1) << __DPL bit_width(static_cast<T>(val - 1));
}

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T bit_ceil(T val) noexcept {
    auto const lhs =
        static_cast<T>(1u << (__DPL bit_width(static_cast<T>(val - 1)) +
                           unsigned_width_diff_v<T>));
    return lhs >> unsigned_width_diff_v<T>;
}

} // namespace details::bit

template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T bit_ceil(T val) noexcept {
    return val <= 1u ? static_cast<T>(1) : details::bit::bit_ceil(val);
}

DPL_DEFAULT_NAMESPACE_END
