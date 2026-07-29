// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/digits.h"
#include "dpl/core/immediate/constants/exponent_bits.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <floating_point_like T>
struct exponent_bias_t : broadcastable_base<exponent_bias_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr exponent_bias_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator int(this exponent_bias_t) noexcept {
        return floating_point_traits<T>::exponent_bias;
    }
};

template <basic_element T>
inline constexpr exponent_bias_t<T> exponent_bias{};

template <basic_element T>
inline constexpr int exponent_bias_v = exponent_bias<T>;

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
