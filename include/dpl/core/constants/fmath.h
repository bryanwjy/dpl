// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/epsilon.h"
#include "dpl/core/constants/exponent_bits.h"
#include "dpl/core/constants/mantissa_bits.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/type_traits/bit_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
DPL_EXPORT
struct toint_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr toint_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this toint_t) noexcept {
        return static_cast<T>(1.0) / epsilon_v<T>;
    }
};

DPL_EXPORT
inline constexpr toint_t toint{};

DPL_EXPORT
struct maxint_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr maxint_t() noexcept = default;

    template <brain_float T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this maxint_t) noexcept {
        return static_cast<T>(
            __DPL bit_cast<internal::bit_type_for_t<T>>(mantissa_bits_v<T>) +
            1);
    }
};

DPL_EXPORT
inline constexpr maxint_t maxint{};

DPL_EXPORT
struct half_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr half_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this half_t) noexcept {
        return 0.5;
    }
};

DPL_EXPORT
inline constexpr half_t half{};

DPL_EXPORT
struct underhalf_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr underhalf_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this underhalf_t) noexcept {
        constexpr auto half_rep =
            __DPL bit_cast<internal::bit_type_for_t<T>>(static_cast<T>(half));
        return __DPL bit_cast<T>(half_rep - 1);
    }
};

DPL_EXPORT
inline constexpr underhalf_t underhalf{};

DPL_EXPORT template <floating_point T>
struct mantissa_of_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_of_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator internal::sbit_type_for_t<T>(
        this mantissa_of_t) noexcept {
        using result = internal::sbit_type_for_t<T>;
        return __DPL bit_cast<result>(mantissa_bits_v<T>);
    }
};

DPL_EXPORT template <floating_point T>
inline constexpr mantissa_of_t<T> mantissa_of{};

DPL_EXPORT template <floating_point T>
struct digits_of_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr digits_of_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator integral auto(this digits_of_t) noexcept {
        using result = internal::sbit_type_for_t<T>;
        return __DPL popcount(
            __DPL to_unsigned(static_cast<result>(mantissa_of<T>)));
    }
};

DPL_EXPORT template <floating_point T>
inline constexpr digits_of_t<T> digits_of{};

DPL_EXPORT template <floating_point T>
struct exp_bias_of_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr exp_bias_of_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator integral auto(this exp_bias_of_t) noexcept {
        constexpr auto exp =
            __DPL bit_cast<internal::bit_type_for_t<T>>(exponent_bits_v<T>);
        return exp >> (__DPL countr_zero(exp) + 1);
    }
};
DPL_EXPORT template <floating_point T>
inline constexpr exp_bias_of_t<T> exp_bias_of{};

DPL_EXPORT template <simd_element T>
struct bit_width_of_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr bit_width_of_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator integral auto(this bit_width_of_t) noexcept {
        return sizeof(T) * char_bit_v;
    }
};
DPL_EXPORT template <floating_point T>
inline constexpr bit_width_of_t<T> bit_width_of{};

DPL_EXPORT
struct rcp_ln2_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr rcp_ln2_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this rcp_ln2_t) noexcept {
        return 1.442695040888963407359924681001892137426645954152985934135449406931;
    }
};

DPL_EXPORT
inline constexpr rcp_ln2_t rcp_ln2{};

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
