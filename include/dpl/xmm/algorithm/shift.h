// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/algorithm/slide.h"

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/operations/reinterpret.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <size_t N, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL shift_left(
        vector<E> lhs, immediate<N> count = imm<N>) noexcept {
    if constexpr (N == 0) {
        return lhs;
    } else if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::shift_left(xmm::reinterpret<int_t>(lhs), count));
    } else {
        constexpr auto imm8 = static_cast<int>(N * sizeof(E));
        return _mm_srli_si128(+lhs, imm8);
    }
}

template <size_t N, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL shift_right(
        vector<E> lhs, immediate<N> count = imm<N>) noexcept {
    if constexpr (N == 0) {
        return lhs;
    } else if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::shift_right(xmm::reinterpret<int_t>(lhs), count));
    } else {
        constexpr auto imm8 = static_cast<int>(N * sizeof(E));
        return _mm_slli_si128(+lhs, imm8);
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_left(abi_tag, vector<E> lhs, size_t count) noexcept
requires requires { xmm::shift_left(lhs, count); }
{
    // defined in slide.h
    return xmm::shift_left(lhs, count);
}

template <simd_element E, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_left(abi_tag, vector<E> lhs, N count) noexcept
requires requires { xmm::shift_left<N::value>(lhs); }
{
    return xmm::shift_left<N::value>(lhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_left(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, N count) noexcept
requires requires { xmm::slide_left(src, mask, lhs, lhs, count); }
{
    return xmm::slide_left(src, mask, lhs, xmm::broadcast<E>(dx::zero), count);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_left(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, N count) noexcept
requires requires {
    xmm::slide_left(zero, mask, lhs, xmm::broadcast<E>(dx::zero), count);
}
{
    return xmm::slide_left(zero, mask, lhs, xmm::broadcast<E>(dx::zero), count);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_right(abi_tag, vector<E> lhs, size_t count) noexcept
requires requires { xmm::shift_right(lhs, count); }
{
    return xmm::shift_right(lhs, count);
}

template <simd_element E, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_right(abi_tag, vector<E> lhs, N count) noexcept
requires requires { xmm::shift_right<N::value>(lhs); }
{
    return xmm::shift_right<N::value>(lhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_right(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, N) noexcept
requires requires {
    xmm::slide_right(src, mask, xmm::broadcast<E>(dx::zero), lhs,
        imm<vector<E>::size() - N::value>);
}
{
    static_assert(N::value <= vector<E>::size());
    return xmm::slide_right(src, mask, xmm::broadcast<E>(dx::zero), lhs,
        imm<vector<E>::size() - N::value>);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> shift_right(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> lhs, N) noexcept
requires requires {
    xmm::slide_right(zero, mask, xmm::broadcast<E>(dx::zero), lhs,
        imm<vector<E>::size() - N::value>);
}
{
    static_assert(N::value <= vector<E>::size());
    return xmm::slide_right(zero, mask, xmm::broadcast<E>(dx::zero), lhs,
        imm<vector<E>::size() - N::value>);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif // if DPL_SIMD_X86_SSE4_2
