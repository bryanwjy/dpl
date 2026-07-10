// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/mantissa_width.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct min_value_t : broadcastable_base<min_value_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr min_value_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this min_value_t) noexcept {
        if constexpr (signed_integral<T>) {
            constexpr auto shift = char_bit_v * sizeof(T) - 1;
            return static_cast<T>(static_cast<T>(1) << shift);
        } else {
            return 0;
        }
    }

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this min_value_t) noexcept {
        using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
        auto const mask = static_cast<bit_type>(1) << mantissa_width_v<T>;
        return __DPL bit_cast<T>(static_cast<bit_type>(mask));
    }
};

DPL_EXPORT inline constexpr min_value_t min_value{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<min_value_t, T>
inline constexpr auto min_value_v = static_cast<T>(min_value);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
