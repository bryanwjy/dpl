// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/compliance.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/type_traits/to_integral.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <floating_point T, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ldexp(compliance::unsafe_t, basic_simd<T, A> val,
    basic_simd<dx::to_signed_integral_t<T>, A> exp) noexcept {
    using int_type = dx::to_signed_integral_t<T>;
    return dx::reinterpret<T>(
        dx::reinterpret<int_type>(val) + (exp << imm<dx::mantissa_width_v<T>>));
}

template <floating_point T, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ldexp(compliance::speed_t, basic_simd<T, A> val,
    basic_simd<dx::to_signed_integral_t<T>, A> exp) noexcept {
    using simdi = decltype(exp);
    constexpr auto pow2i = [](simdi exp) {
        return dx::reinterpret<T>(
            (exp + dx::exponent_bias<T>) << imm<dx::mantissa_width_v<T>>);
    };

    auto const hexp = exp >> imm<1>;
    return val * pow2i(hexp) * pow2i(exp - hexp);
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
