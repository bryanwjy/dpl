// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/digits.h"
#include "dpl/core/constants/exponent_bits.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <floating_point T>
struct exponent_bias_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr exponent_bias_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator int(this exponent_bias_t) noexcept {
        using bit_type = bit_type_t<char_bit_v * sizeof(T)>;
        constexpr auto exp =
            __DPL bit_cast<bit_type>(exponent_bits_v<T>) >> (digits_v<T> + 1);
        return static_cast<int>(exp);
    }
};

DPL_EXPORT template <arithmetic_type T>
inline constexpr exponent_bias_t<T> exponent_bias{};

DPL_EXPORT template <arithmetic_type T>
inline constexpr int exponent_bias_v = exponent_bias<T>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
