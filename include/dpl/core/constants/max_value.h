// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/exponent_bits.h"
#include "dpl/core/constants/mantissa_bits.h"
#include "dpl/core/constants/min_value.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT
struct max_value_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr max_value_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this max_value_t self) noexcept {
        return ~static_cast<T>(min_value);
    }

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this max_value_t) noexcept {
        using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
        auto const mantissa = __DPL bit_cast<bit_type>(mantissa_bits_v<T>);
        auto const exponent = __DPL bit_cast<bit_type>(exponent_bits_v<T>) - 1;
        return __DPL bit_cast<T>(static_cast<bit_type>(mantissa | exponent));
    }
};

DPL_EXPORT
inline constexpr max_value_t max_value{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<max_value_t, T>
inline constexpr auto max_value_v = static_cast<T>(max_value);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
