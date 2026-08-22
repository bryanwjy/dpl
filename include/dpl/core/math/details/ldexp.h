// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#include "dpl/core/math/details/compliance.h" // IWYU pragma: export
#include "dpl/core/math/details/floating_point_simd.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/exponent_bias.h"
#  include "dpl/core/immediate/constants/mantissa_width.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

template <canonical_vector T>
requires binary_layout_floating_point<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ldexp(
    compliance::unsafe_t, T val, exponent_vector_t<T> exp) noexcept {
    using sint_t = simd_element_type_t<exponent_vector_t<T>>;
    constexpr auto shift =
        __DPL countr_zero(floating_point_traits<T>::exponent_mask);
    auto const rhs = dx::bwshift_left(exp, imm<shift>);
    auto const result = dx::add(dx::reinterpret<sint_t>(val), rhs);
    return dx::reinterpret<T>(result);
}

template <canonical_vector T>
requires binary_layout_floating_point<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ldexp(compliance::unsafe_t,
    type_identity_t<T> src, simd_mask_type_t<T> mask, T val,
    exponent_vector_t<T> exp) noexcept {
    using sint_t = simd_element_type_t<exponent_vector_t<T>>;
    constexpr auto shift =
        __DPL countr_zero(floating_point_traits<T>::exponent_mask);
    auto const rhs = dx::bwshift_left(exp, imm<shift>);
    auto const result = dx::add(
        dx::reinterpret<sint_t>(src), mask, dx::reinterpret<sint_t>(val), rhs);
    return dx::reinterpret<T>(result);
}

template <canonical_vector T>
requires binary_layout_floating_point<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ldexp(compliance::unsafe_t,
    simd_mask_type_t<T> mask, T val, exponent_vector_t<T> exp) noexcept {
    using sint_t = simd_element_type_t<exponent_vector_t<T>>;
    constexpr auto shift =
        __DPL countr_zero(floating_point_traits<T>::exponent_mask);
    auto const rhs = dx::bwshift_left(exp, imm<shift>);
    auto const result =
        dx::add(dx::zero, mask, dx::reinterpret<sint_t>(val), rhs);
    return dx::reinterpret<T>(result);
}

template <canonical_vector T>
requires binary_layout_floating_point<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ldexp(
    compliance::speed_t, T val, exponent_vector_t<T> exp) noexcept {
    using E = simd_element_type_t<T>;
    constexpr auto shift =
        __DPL countr_zero(floating_point_traits<T>::exponent_mask);
    constexpr auto pow2i = [](exponent_vector_t<T> exp) {
        return dx::reinterpret<T>(dx::bwshift_left(
            dx::add(exp, floating_point_traits<E>::exponent_bias), imm<shift>));
    };

    auto const hexp = dx::bwshift_right(exp, imm<1>);
    return dx::multiply(
        dx::multiply(val, pow2i(hexp)), pow2i(dx::subtract(exp, hexp)));
}

} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
