// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/all_bits.h"
#include "dpl/core/constants/digits.h"
#include "dpl/core/constants/msb.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

inline constexpr struct exponent_bits_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr exponent_bits_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this exponent_bits_t) noexcept {
        using bit_type = bit_type_t<char_bit_v * sizeof(T)>;
        auto const ninf = all_bits_v<bit_type> << digits_v<T>;
        auto const mask = ninf & ~msb_v<bit_type>;
        return __DPL bit_cast<T>(mask);
    }
} exponent_bits{};

template <typename T>
requires explicitly_convertible_to<exponent_bits_t, T>
inline constexpr auto exponent_bits_v = static_cast<T>(exponent_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
