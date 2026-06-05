// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/digits.h"
#include "dpl/core/constants/exponent_bits.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <floating_point T>
struct nexponent_mask_t : broadcastable_base<nexponent_mask_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr nexponent_mask_t() noexcept =
        default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator signed_representation_t<T>(
        this nexponent_mask_t) noexcept {
        using sbit = signed_representation_t<T>;
        return ~__DPL bit_cast<sbit>(exponent_bits_v<T>);
    }
};

template <floating_point T>
struct exponent_mask_t : broadcastable_base<exponent_mask_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr exponent_mask_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator signed_representation_t<T>(
        this exponent_mask_t) noexcept {
        using sbit = signed_representation_t<T>;
        return __DPL bit_cast<sbit>(exponent_bits_v<T>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr signed_representation_t<T> operator>>(
        this exponent_mask_t, digits_t<T>) noexcept {
        using sbit = signed_representation_t<T>;
        return __DPL bit_cast<sbit>(exponent_bits_v<T>) >> mantissa_width_v<T>;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr nexponent_mask_t<T> operator~(this exponent_mask_t) noexcept {
        return nexponent_mask_t<T>{};
    }
};

template <floating_point T>
inline constexpr exponent_mask_t<T> exponent_mask{};

template <floating_point T, integral U>
requires explicitly_convertible_to<exponent_mask_t<T>, U> &&
    (sizeof(T) == sizeof(U))
inline constexpr auto exponent_mask_v = static_cast<U>(exponent_mask<T>);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
