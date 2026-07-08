// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct value_bits_t : broadcastable_base<value_bits_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr value_bits_t() noexcept = default;

    template <signed_integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this value_bits_t) noexcept {
        using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
        return __DPL bit_cast<T>(
            static_cast<bit_type>(static_cast<bit_type>(-1) >> 1));
    }

    template <unsigned_integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this value_bits_t) noexcept {
        return static_cast<T>(-1);
    }

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this value_bits_t) noexcept {
        if constexpr (floating_point_traits<T>::has_hidden_bit) {
            return __DPL bit_cast<T>(floating_point_traits<T>::exponent_mask |
                floating_point_traits<T>::mantissa_mask);
        } else {
            return __DPL bit_cast<T>(floating_point_traits<T>::exponent_mask |
                (floating_point_traits<T>::mantissa_mask << 1) |
                floating_point_traits<T>::mantissa_mask);
        }
    }
};

DPL_EXPORT inline constexpr value_bits_t value_bits{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<value_bits_t, T>
inline constexpr auto value_bits_v = static_cast<T>(value_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
