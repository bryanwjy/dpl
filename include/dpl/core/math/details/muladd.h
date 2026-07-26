// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/numbers/floating_point_like.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
template <floating_point_like E, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto muladd(
    basic_vector<E, A> a, basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
    using vec_t = basic_vector<E, A>;
    if constexpr (dx::is_simd_canonical_invocable<vec_t, vec_t, vec_t>(
                      dx::fmadd)) {
        return dx::fmadd(a, b, c);
    } else {
        return dx::add(dx::multiply(a, b), c);
    }
}

template <floating_point_like E, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto mulsub(
    basic_vector<E, A> a, basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
    using vec_t = basic_vector<E, A>;
    if constexpr (dx::is_simd_canonical_invocable<vec_t, vec_t, vec_t>(
                      dx::fmadd)) {
        return dx::fmsub(a, b, c);
    } else {
        return dx::subtract(dx::multiply(a, b), c);
    }
}

template <floating_point_like E, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto nmuladd(
    basic_vector<E, A> a, basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
    using vec_t = basic_vector<E, A>;
    if constexpr (dx::is_simd_canonical_invocable<vec_t, vec_t, vec_t>(
                      dx::fmadd)) {
        return dx::fnmadd(a, b, c);
    } else {
        return dx::subtract(c, dx::multiply(a, b));
    }
}

template <floating_point_like E, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto nmulsub(
    basic_vector<E, A> a, basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
    using vec_t = basic_vector<E, A>;
    if constexpr (dx::is_simd_canonical_invocable<vec_t, vec_t, vec_t>(
                      dx::fmadd)) {
        return dx::fnmsub(a, b, c);
    } else {
        return dx::negate(fmath::muladd(a, b, c));
    }
}
} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
