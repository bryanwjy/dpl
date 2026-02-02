// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/ldexp.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/bit_type.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dxi = __DPL datapar::internal;

template <floating_point T, simd_abi A>
struct decomposition {
    basic_simd<T, A> significand;
    basic_simd<dxi::sbit_type_for_t<T>, A> exponent;
};

/**
 * Decompose the argument into a sigficand in the interval [0.75, 1.5] and it's
 * exponent, such that ldexp(significand, exponent) ~ arg
 */
template <simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL decompose(basic_simd<float, A> val) noexcept {
    using simd = basic_simd<float, A>;
    constexpr auto k = dx::broadcast<simd>(static_cast<float>(int64(1) << 32));
    constexpr auto k2 = k * k;
    constexpr auto fourthirds = dx::broadcast<simd>(1.0f / 0.75f);
    auto const mask = val < dx::min_value;
    auto const dval = dx::select(mask, val * k, val);
    auto const exp = fmath::ilogb(compliance::unsafe, dval * fourthirds);
    return decomposition<float, A>{
        .significand = fmath::ldexp(compliance::unsafe, dval, -exp),
        .exponent = dx::select(mask, exp - 64, exp),
    };
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
