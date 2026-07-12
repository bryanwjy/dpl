// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/exponent_bits.h"
#include "dpl/core/immediate/constants/mantissa_bits.h"
#include "dpl/core/immediate/constants/min_value.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/integral.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct max_value_t : broadcastable_base<max_value_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr max_value_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this max_value_t self) noexcept {
        return ~static_cast<T>(min_value);
    }

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this max_value_t) noexcept {
        constexpr auto digits = floating_point_traits<T>::digits;
        constexpr auto exp = floating_point_traits<T>::exponent_mask;
        constexpr auto mantissa = floating_point_traits<T>::mantissa_mask;
        constexpr auto width = floating_point_traits<T>::width;
        constexpr auto exp_width = __DPL popcount(exp);
        constexpr auto expv = __DPL to_underlying(
            __DPL truncate<exp_width>(exp >> (digits - 1)));
        constexpr auto result_exp = bitset<width>(bitset<exp_width>(expv - 1u));
        if constexpr (floating_point_traits<T>::has_hidden_bit) {
            return __DPL bit_cast<T>(result_exp | mantissa);
        } else {
            return __DPL bit_cast<T>(result_exp | mantissa | (mantissa << 1));
        }
    }
};

inline constexpr max_value_t max_value{};

template <typename T>
requires explicitly_convertible_to<max_value_t, T>
inline constexpr auto max_value_v = static_cast<T>(max_value);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
