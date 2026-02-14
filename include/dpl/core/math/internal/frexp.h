// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/ldexp.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/to_integral.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

template <floating_point T, simd_abi A>
struct decomposition {
    basic_simd<T, A> significand;
    basic_simd<dx::to_signed_integral_t<T>, A> exponent;
};

template <floating_point E>
inline constexpr auto denormalizer = []() {
    if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
        return static_cast<E>(0x1.p64);
    } else {
        return static_cast<E>(0x1.p12);
    }
}();
template <floating_point E>
inline constexpr auto subnormal_offset = []() {
    if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
        return 64;
    } else {
        return 12;
    }
}();

/**
 * Decompose a POSITIVE argument into a sigficand in the interval [0.75, 1.5)
 * and it's exponent, such that ldexp(significand, exponent) ~ arg
 */
template <floating_point E, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL frexp_balanced(basic_simd<E, A> val) noexcept {
    constexpr auto fourthirds = dx::broadcast<E, A>(1.0 / 0.75);
    auto const issubnormal = val < dx::min_value;
    auto const dval = dx::select(issubnormal, val * denormalizer<E>, val);
    auto const exp = fmath::ilogb(compliance::unsafe, dval * fourthirds);
    return decomposition<E, A>{
        .significand = fmath::ldexp(compliance::unsafe, dval, -exp),
        .exponent = dx::select(issubnormal, exp - subnormal_offset<E>, exp),
    };
}

/**
 * Decompose a POSITIVE argument into a sigficand in the interval [0.5, 1.0)
 * and it's exponent, such that ldexp(significand, exponent) ~ arg
 */
template <floating_point E, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL frexp(basic_simd<E, A> val) noexcept {
    using int_type = dx::to_signed_integral_t<E>;
    constexpr auto exp_bits = __DPL bit_cast<int_type>(dx::exponent_bits_v<E>);
    constexpr auto magic = static_cast<int_type>(exponent_bias_v<E> - 1);
    constexpr auto magic_exp = __DPL bit_cast<E>(magic << dx::digits_v<E>);

    auto const issubnormal = val < dx::min_value;
    auto const dval = dx::select(issubnormal, val * denormalizer<E>, val);
    auto exp = dx::reinterpret<int_type>(
        (val & dx::exponent_bits) >> imm<dx::digits_v<E>>);
    exp -= dx::select(exp != dx::zero && exp != exp_bits, magic, dx::zero);
    return decomposition<E, A>{
        .significand = (val & ~exponent_bits) | magic_exp,
        .exponent = dx::select(issubnormal, exp - subnormal_offset<E>, exp),
    };
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
