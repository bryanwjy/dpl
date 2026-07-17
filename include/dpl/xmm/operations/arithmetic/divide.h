// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/arithmetic.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL divide(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_div_ps(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL divide(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_div_pd(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512VL
#    if DPL_SIMD_X86_AVX512F

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL divide(vector<float> src, cmask_t<float, M>,
        vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_mask_div_ps(+src, M, +lhs, +rhs);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL divide(vector<double> src, cmask_t<double, M>,
        vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_mask_div_pd(+src, M, +lhs, +rhs);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL divide(dx::zero_t, cmask_t<float, M>, vector<float> lhs,
        vector<float> rhs) noexcept {
    return _mm_maskz_div_ps(M, +lhs, +rhs);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL divide(dx::zero_t, cmask_t<double, M>, vector<double> lhs,
        vector<double> rhs) noexcept {
    return _mm_maskz_div_pd(M, +lhs, +rhs);
}
#    endif

#    if DPL_SIMD_X86_AVX512FP16
inline vector<ext::float16> DPL_VECTORCALL divide(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_div_ph(+lhs, +rhs);
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL divide(vector<ext::float16> src, cmask_t<ext::float16, M>,
        vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_mask_div_ph(src, M, +lhs, +rhs);
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL divide(dx::zero_t zero, cmask_t<ext::float16, M>,
        vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_maskz_div_ph(M, +lhs, +rhs);
}
#    endif
#  endif

template <same_as<ext::bfloat16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16>
    DPL_VECTORCALL divide(vector<E> lhs, vector<E> rhs) noexcept {
    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo = xmm::divide(xmm::to_float(lhs), xmm::to_float(rhs));
    auto const hi = xmm::divide(xmm::to_float(hlhs), xmm::to_float(hrhs));

#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#  else
    auto const packed =
        _mm_shuffle_pd( __DPL bit_cast<__m128d>(+xmm::element_cast<E>(lo)),
            __DPL bit_cast<__m128d>(+xmm::element_cast<E>(hi)), 0);
    return xmm::reinterpret<ext::bfloat16>(vector<double>(packed));
#  endif
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL divide(
    abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::divide(lhs, rhs); }
{
    return xmm::divide(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL divide(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::divide(src, mask, lhs, rhs); }
{
    return xmm::divide(src, mask, lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL divide(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::divide(zero, mask, lhs, rhs); }
{
    return xmm::divide(zero, mask, lhs, rhs);
}
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
