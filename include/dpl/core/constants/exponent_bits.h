// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/all_bits.h"
#include "dpl/core/constants/mantissa_width.h"
#include "dpl/core/constants/msb.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct nexponent_bits_t : broadcastable_base<nexponent_bits_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr nexponent_bits_t() noexcept =
        default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this nexponent_bits_t) noexcept {
        using bit_type = bit_type_t<char_bit_v * sizeof(T)>;
        auto const ninf = all_bits_v<bit_type> << mantissa_width_v<T>;
        auto const mask = dpl::to_unsigned(ninf & ~msb_v<bit_type>);
        return __DPL bit_cast<T>(
            static_cast<unsigned_representation_t<T>>(~mask));
    }
};

inline constexpr struct exponent_bits_t : broadcastable_base<nexponent_bits_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr exponent_bits_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this exponent_bits_t) noexcept {
        using bit_type = bit_type_t<char_bit_v * sizeof(T)>;
        auto const ninf = all_bits_v<bit_type> << mantissa_width_v<T>;
        auto const mask = dpl::to_unsigned(ninf & ~msb_v<bit_type>);
        return __DPL bit_cast<T>(
            static_cast<unsigned_representation_t<T>>(mask));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr nexponent_bits_t operator~(this exponent_bits_t) noexcept {
        return nexponent_bits_t{};
    }
} exponent_bits{};

template <typename T>
requires explicitly_convertible_to<exponent_bits_t, T>
inline constexpr auto exponent_bits_v = static_cast<T>(exponent_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
