// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/bit_width.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/details/bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T bit_ceil(T val) noexcept {
    return val <= 1u ? static_cast<T>(1) : [](T val) noexcept {
        if constexpr (requires(T val) {
                          { +val } -> same_as<T>;
                      }) {
            return static_cast<T>(1)
                << __DPL bit_width(static_cast<T>(val - 1));
        } else {
            auto const lhs = static_cast<T>(
                1u << (__DPL bit_width(static_cast<T>(val - 1)) +
                    details::bit::unsigned_width_diff_v<T>));
            return lhs >> details::bit::unsigned_width_diff_v<T>;
        }
    }(val);
}

DPL_DEFAULT_NAMESPACE_END
